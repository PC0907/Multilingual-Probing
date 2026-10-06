#!/usr/bin/env python3
"""Frozen single-GPU prompted block-state extraction for supported base LMs.

Cache axis 1 contains decoder blocks 1..N, in order, BEFORE the final model
normalization. Embeddings and final-normalized states are deliberately absent.
Consumers must require BOTH metadata.complete and the atomic COMPLETE marker.
Existing output directories are refused; failed partial caches are not resumed.

Each input ``sentence`` is the complete prompt and must end at the final
statement character.  The extractor pools the final *non-special content token*,
so an automatically appended EOS is never mistaken for the statement token.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def reject_nonfinite_json(value):
    raise ValueError('Nonfinite JSON value: ' + value)


def validate_records(records):
    """Preserve IDs and arbitrary extra metadata without coercing them."""
    if not isinstance(records, list) or not records:
        raise ValueError('Input must be a nonempty JSON array of records')
    observed = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict) or not {'id', 'sentence', 'label'} <= row.keys():
            raise ValueError('Record requires id, sentence and label at index ' + str(index))
        if row['id'] is None or row['id'] == '':
            raise ValueError('Missing record ID at index ' + str(index))
        identifier = json.dumps(row['id'], sort_keys=True, ensure_ascii=False, allow_nan=False)
        if identifier in observed:
            raise ValueError('Duplicate record ID at index ' + str(index))
        observed.add(identifier)
        if not isinstance(row['sentence'], str) or not row['sentence'].strip():
            raise ValueError('Sentence must be nonempty text at index ' + str(index))
        if row['label'] not in (0, 1):
            raise ValueError('Label must be 0 or 1 at index ' + str(index))
        json.dumps(row, allow_nan=False)  # Validate extra provenance metadata too.
    return [{key: value for key, value in row.items() if key != 'sentence'} for row in records]


def pooling_layout(input_ids, attention_mask, bos_token_id):
    """CPU helper: right-padded last positions and text-token mean weights."""
    ids, mask = np.asarray(input_ids), np.asarray(attention_mask)
    if ids.ndim != 2 or ids.shape != mask.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError('Input IDs and binary attention mask must be aligned matrices')
    if (mask.sum(axis=1) == 0).any() or (np.diff(mask.astype(int), axis=1) > 0).any():
        raise ValueError('Every sample must have nonempty contiguous right-padded tokens')
    last = mask.sum(axis=1).astype(np.int64) - 1
    mean_weights = mask.astype(np.float32)
    if bos_token_id is not None:
        mean_weights[ids[:, 0] == bos_token_id, 0] = 0
    if (mean_weights.sum(axis=1) == 0).any():
        raise ValueError('No text token remains after excluding the initial BOS')
    return last, mean_weights


def content_end_positions(full_ids, content_ids):
    """Locate one exact content-token subsequence inside special-token-wrapped IDs."""
    positions = []
    for row, content in zip(full_ids, content_ids):
        if not content:
            raise ValueError('Tokenizer produced empty prompt content')
        hits = [start for start in range(len(row) - len(content) + 1)
                if row[start:start + len(content)] == content]
        if len(hits) != 1:
            raise ValueError('Prompt content must occur exactly once inside special-token-wrapped IDs')
        positions.append(hits[0] + len(content) - 1)
    return positions


def tokenizer_fingerprints(directory):
    directory = Path(directory)
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('config.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
                         'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json')
            if (directory / name).is_file()}


def gpu_snapshot():
    try:
        return {'csv': subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=15).strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {'unavailable': str(error)}


def cache_metadata(records, source_hash, model_path, model_config, token_lengths,
                   target_positions, tokenizer_hashes):
    blocks, width = int(model_config['num_hidden_layers']), int(model_config['hidden_size'])
    return {'schema_version': 1, 'complete': False, 'status': 'initialized',
            'input_sha256': source_hash, 'model': str(Path(model_path).resolve()),
            'model_config': model_config, 'configuration_and_tokenizer_sha256': tokenizer_hashes,
            'source_ids': [row['id'] for row in records], 'record_metadata': validate_records(records),
            'n': len(records), 'shape': [len(records), blocks, width],
            'layer_numbers': list(range(1, blocks + 1)),
            'layer_convention': 'tensor[:,j,:] = decoder block j+1 output BEFORE final model normalization; no embedding or final-normalized state included',
            'pooling': {'last': 'final non-special content token; input prompt ends at final statement character',
                        'mean': 'mean of non-padding tokens, excluding only an initial BOS when present; internal special tokens retained'},
            'token_lengths': token_lengths, 'target_token_positions': target_positions,
            'target_position_rule': 'unique exact match of tokenizer(prompt, add_special_tokens=False) inside add_special_tokens=True IDs; take its final token',
            'max_length': max(token_lengths), 'truncated': 0,
            'dtype': 'bfloat16 frozen forward; float32 pooled cache', 'n_completed_samples': 0}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_disk_space(directory, shape, n_poolings=2):
    cache_bytes = n_poolings * int(shape[0]) * int(shape[1]) * int(shape[2]) * np.dtype(np.float32).itemsize
    reserve_bytes = 6 * 1024**3
    free_bytes = int(shutil.disk_usage(directory).free)
    if free_bytes < cache_bytes + reserve_bytes:
        raise RuntimeError('Insufficient disk space: need ' + str(cache_bytes) + ' cache bytes plus fixed '
                           + str(reserve_bytes) + ' reserve bytes; only ' + str(free_bytes) + ' bytes free')
    return {'estimated_cache_bytes': cache_bytes, 'required_reserve_bytes': reserve_bytes,
            'observed_free_bytes': free_bytes, 'required_free_bytes': cache_bytes + reserve_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', required=True, help='Existing local checkpoint directory')
    parser.add_argument('--output', required=True, help='New cache directory; never reused')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=512, help='Refuse longer examples; never truncate')
    parser.add_argument('--limit', type=int, help='Pilot prefix only; use a separate output directory')
    parser.add_argument('--poolings', choices=['last', 'last,mean'], default='last', help='Explicit storage scope; new study primary last only')
    parser.add_argument('--hash-caches', action='store_true', help='SHA256 full final NPY files after extraction')
    args = parser.parse_args()
    poolings = args.poolings.split(',')
    if args.batch_size < 1 or args.max_seq_length < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size, maximum length and optional limit must be positive')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible or ',' in visible:
        raise RuntimeError('Expose exactly one GPU UUID/index using CUDA_VISIBLE_DEVICES')
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise ValueError('--model must be an existing local checkpoint directory')
    source = Path(args.input).read_bytes()
    original_records = json.loads(source, parse_constant=reject_nonfinite_json)
    validate_records(original_records)
    records = original_records if args.limit is None else original_records[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = {'complete': False, 'status': 'initializing', 'input_sha256': hashlib.sha256(source).hexdigest(),
                'input': str(Path(args.input).resolve()), 'model': str(model_path.resolve()),
                'cuda_visible_devices': visible, 'n_raw': len(original_records), 'n': len(records),
                'pilot': args.limit is not None, 'requested_limit': args.limit,
                'batch_size': args.batch_size, 'requested_max_seq_length': args.max_seq_length,
                'local_files_only': True, 'trust_remote_code': False, 'device': 'cuda:0',
                'gpu_before': gpu_snapshot()}
    atomic_json(out / 'metadata.json', metadata)
    handles, arrays = [], {}
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA device must be visible')
        torch.cuda.set_device(0)
        torch.cuda.init()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('The selected GPU does not support required BF16 execution')
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        supported_model_types = ('gemma', 'gemma2', 'qwen3', 'llama', 'apertus', 'mistral', 'gpt_neox', 'olmo2')
        if config.model_type not in supported_model_types:
            raise ValueError('Supported model types are ' + ', '.join(supported_model_types) +
                             '; received ' + config.model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            vocab = tokenizer.get_vocab()
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                metadata['padding_fallback'] = 'eos'
            elif '<|padding|>' in vocab:
                # Early Pythia tags ship an empty special-token map; GPT-NeoX's own
                # padding token is in the vocabulary. Right padding is masked and the
                # pooled position is the last content token, so states are unaffected.
                tokenizer.pad_token = '<|padding|>'
                metadata['padding_fallback'] = 'gpt_neox_padding_token'
            else:
                raise ValueError('Tokenizer has neither a padding nor EOS token')
        prompts = [row['sentence'] for row in records]
        all_ids = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=True)['input_ids']
        content_ids = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=False)['input_ids']
        target_positions = content_end_positions(all_ids, content_ids)
        lengths = [len(ids) for ids in all_ids]
        max_length = min(args.max_seq_length, int(config.max_position_embeddings))
        offenders = [{'id': row['id'], 'tokens': length} for row, length in zip(records, lengths) if length > max_length]
        if offenders:
            metadata['overlong_examples'] = offenders
            raise ValueError(str(len(offenders)) + ' examples exceed ' + str(max_length) + ' tokens; refusing truncation')
        if any(length == 0 for length in lengths):
            raise ValueError('Tokenizer produced an empty sequence')
        initial_metadata = cache_metadata(records, hashlib.sha256(source).hexdigest(), model_path,
                                          config.to_dict(), lengths, target_positions,
                                          tokenizer_fingerprints(model_path))
        metadata.update(initial_metadata)
        metadata.update({'effective_max_seq_length': max_length, 'tokenizer_class': type(tokenizer).__name__,
                         'pad_token_id': tokenizer.pad_token_id, 'bos_token_id': tokenizer.bos_token_id,
                         'eos_token_id': tokenizer.eos_token_id, 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'gpu_name': torch.cuda.get_device_name(0),
                         'model_weight_files': [{'name': path.name, 'bytes': path.stat().st_size}
                                                for path in sorted(model_path.iterdir()) if path.suffix in ('.safetensors', '.bin')],
                         'weight_provenance_note': 'Local weight filenames/sizes recorded; cache/model acquisition manifests should supply full weight hashes.'})
        metadata['extracted_poolings'] = poolings
        metadata['extractor_script_sha256'] = sha256_file(Path(__file__))
        metadata['pooling'] = {key: value for key, value in metadata['pooling'].items() if key in poolings}
        metadata['disk_preflight'] = check_disk_space(out, metadata['shape'], len(poolings))
        atomic_json(out / 'metadata.json', metadata)
        torch.cuda.reset_peak_memory_stats(0)
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                         torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        model.eval().requires_grad_(False).to('cuda:0')
        blocks = getattr(model, 'layers', None)
        final_norm = getattr(model, 'norm', None)
        if final_norm is None:
            final_norm = getattr(model, 'final_layer_norm', None)
        if blocks is None or len(blocks) != config.num_hidden_layers or final_norm is None:
            raise RuntimeError('Unexpected base-model layout; refusing ambiguous layer extraction')
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError('Model is not frozen')
        for pooling in poolings:
            arrays[pooling] = np.lib.format.open_memmap(out / (pooling + '.partial.npy'), mode='w+',
                                                       dtype=np.float32, shape=tuple(metadata['shape']))
        state = {}

        def collect(hidden, layer_index):
            if hidden.ndim != 3 or hidden.shape[0] != state['size'] or hidden.shape[2] != config.hidden_size:
                raise RuntimeError('Unexpected decoder block output shape')
            if layer_index in state['seen_layers']:
                raise RuntimeError('Decoder block was executed more than once per batch')
            state['seen_layers'].add(layer_index)
            if state['start'] == 0 and layer_index == len(blocks) - 1:
                state['final_block_for_check'] = hidden.detach().clone()
            batch_rows = torch.arange(state['size'], device=hidden.device)
            last = hidden[batch_rows, state['last']].float()
            mean = (torch.bmm(state['mean_mask'].unsqueeze(1), hidden.float()).squeeze(1) / state['mean_count']) if 'mean' in poolings else None
            if not bool(torch.isfinite(last).all()) or (mean is not None and not bool(torch.isfinite(mean).all())):
                raise ValueError('Nonfinite pooled activation')
            start, stop = state['start'], state['start'] + state['size']
            arrays['last'][start:stop, layer_index] = last.cpu().numpy()
            if mean is not None:
                arrays['mean'][start:stop, layer_index] = mean.cpu().numpy()

        for layer_index, block in enumerate(blocks):
            def hook(module, inputs, output, layer_index=layer_index):
                collect(output[0] if isinstance(output, tuple) else output, layer_index)
            handles.append(block.register_forward_hook(hook))
        metadata.update({'status': 'extracting', 'model_load_and_preflight_seconds': time.time() - started})
        atomic_json(out / 'metadata.json', metadata)
        forward_started = time.time()
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                ids = all_ids[start:start + args.batch_size]
                batch = tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                                      padding=True, return_tensors='pt')
                _, mean_mask = pooling_layout(batch['input_ids'].numpy(), batch['attention_mask'].numpy(), tokenizer.bos_token_id)
                last = np.asarray(target_positions[start:start + len(ids)], dtype=np.int64)
                if (last >= batch['attention_mask'].sum(dim=1).numpy()).any():
                    raise AssertionError('Target token falls outside non-padding input')
                batch = batch.to('cuda:0')
                weights = torch.as_tensor(mean_mask, dtype=torch.float32, device='cuda:0')
                state.update(start=start, size=len(ids), last=torch.as_tensor(last, device='cuda:0'),
                             mean_mask=weights, mean_count=weights.sum(dim=1, keepdim=True), seen_layers=set())
                result = model(**batch, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
                if len(state['seen_layers']) != len(blocks):
                    raise RuntimeError('Missing decoder block outputs')
                if start == 0:
                    raw_final = torch.as_tensor(np.array(arrays['last'][start:start + len(ids), -1]),
                                                dtype=torch.bfloat16, device='cuda:0')
                    # RMSNorm reduction kernels may round differently for a
                    # gathered 2D tensor than the original 3D batch. Verify the
                    # exact live block output and same-shape model operation.
                    live_final = state.pop('final_block_for_check')
                    live_last = live_final[torch.arange(len(ids), device='cuda:0'), state['last']]
                    if not torch.equal(raw_final, live_last):
                        raise AssertionError('Cached final-block state differs from live pooled block output')
                    predicted_full = final_norm(live_final).float()
                    predicted_final = predicted_full[torch.arange(len(ids), device='cuda:0'), state['last']]
                    returned_final = result.last_hidden_state[
                        torch.arange(len(ids), device='cuda:0'), state['last']].float()
                    difference = float(torch.max(torch.abs(predicted_final - returned_final)).item())
                    if not torch.equal(predicted_full, result.last_hidden_state.float()):
                        raise AssertionError('Final block hook plus actual final norm disagrees with returned model state')
                    gathered_norm_difference = float(torch.max(torch.abs(final_norm(raw_final).float() - returned_final)).item())
                    metadata['final_block_runtime_verification'] = {
                        'passed': True, 'first_batch_samples': len(ids), 'maximum_absolute_difference': difference,
                        'gathered_2D_norm_max_absolute_difference': gathered_norm_difference,
                        'cached_state_matches_live_pooled_output_exactly': True,
                        'description': 'Cached last state equals live block pool exactly; actual final norm applied once to full live 3D block matches entire model return exactly. Gathered 2D RMSNorm rounding is separately recorded.'}
                    del raw_final, predicted_final, returned_final, predicted_full, live_final, live_last
                del result, batch
                done = start + len(ids)
                metadata['n_completed_samples'] = done
                if start == 0 or done % 128 == 0 or done == len(records):
                    for array in arrays.values():
                        array.flush()
                    metadata.update({'seconds_so_far': time.time() - started,
                                     'forward_seconds': time.time() - forward_started,
                                     'peak_cuda_gib': torch.cuda.max_memory_allocated(0) / 2**30})
                    atomic_json(out / 'metadata.json', metadata)
                    print(json.dumps({'done': done, 'total': len(records), 'seconds': metadata['seconds_so_far'],
                                      'peak_cuda_gib': metadata['peak_cuda_gib']}), flush=True)
        for handle in handles:
            handle.remove()
        handles.clear()
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()
        gc.collect()
        for pooling in poolings:
            (out / (pooling + '.partial.npy')).replace(out / (pooling + '.npy'))
        metadata['cache_files'] = {pooling: {'filename': pooling + '.npy', 'bytes': (out / (pooling + '.npy')).stat().st_size}
                                   for pooling in poolings}
        if args.hash_caches:
            for value in metadata['cache_files'].values():
                value['sha256'] = sha256_file(out / value['filename'])
        metadata.update({'complete': True, 'status': 'complete', 'seconds': time.time() - started,
                         'peak_cuda_gib': torch.cuda.max_memory_allocated(0) / 2**30, 'gpu_after': gpu_snapshot()})
        atomic_json(out / 'metadata.json', metadata)
        marker = out / 'COMPLETE.tmp'
        marker.write_text('All pre-final-normalization decoder block states extracted successfully.\n')
        marker.replace(out / 'COMPLETE')
        print(json.dumps({'complete': str(out), 'shape': metadata['shape'], 'seconds': metadata['seconds']}), flush=True)
    except BaseException as error:
        metadata.update({'complete': False, 'status': 'failed', 'error_type': type(error).__name__,
                         'error': str(error), 'seconds': time.time() - started})
        atomic_json(out / 'metadata.json', metadata)
        raise
    finally:
        for handle in handles:
            handle.remove()
        for array in arrays.values():
            array.flush()


if __name__ == '__main__':
    main()
