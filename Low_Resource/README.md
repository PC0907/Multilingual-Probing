# Truth directions in low-resource languages and scripts

Code and figures for an extension of *"Shared facts raise cosine, not transfer"* (truth-direction
transfer across languages) to five low-resource Indic languages (Urdu, Marathi, Nepali, Gujarati,
Punjabi) and seven script interventions (Urdu in Devanagari and Roman, Punjabi in Shahmukhi and
Devanagari, Gujarati in Devanagari, Marathi in Gujarati script, Roman Hindi), across Gemma-7B,
Qwen3-8B-Base, Apertus-8B-2509 and Mistral-7B-v0.3.

Main findings: sharing a script gives no transfer benefit (script-match difference-in-differences
about +0.005 AUROC); how familiar the target text is to the model (bits per character) is the dominant
predictor of transfer; script and relatedness change direction cosine much more than transfer AUROC.

## Layout

| path | contents |
|---|---|
| `code/config.py` | models, conditions, prompts, paths, seed |
| `code/check_chunks.py`, `code/build_data.py` | translation validation, dataset building, Brahmic transliteration |
| `code/extract.py`, `code/queue_worker.sh` | residual-stream extraction (all blocks, final statement token), token counts, NLL |
| `code/layer_selection.py` | frozen block per model (validation AUROC on the original six languages) |
| `code/common.py` | mass-mean probe, metrics, dependency-group bootstrap |
| `code/analysis_core.py` | 18x18 transfer matrices, zero-overlap / disattenuated cosine, 1,000-draw bootstrap |
| `code/features.py`, `code/quality.py` | fertility, bits per character, token overlap, lang2vec distances; LaBSE quality checks |
| `code/analysis_samplesize.py`, `code/analysis_layers.py` | RQ6 sample-size curves; RQ11/12 layerwise analysis |
| `code/analysis_rq.py` | all research-question tables and figures |
| `code/build_report_html.py` | report rendering |
| `figures/` | result figures |

`data/` holds all 18 conditions (2,000 claims each, same ids, labels, dependency groups and split), the source CSV, and the raw translation chunks in `data/translation_work/`. Activation caches (~90 GB) and result files are not included.

## Reproduce

```bash
cd code
python build_data.py ur pa mr gu ne hiLatn translit
python extract.py --model <model> --conds <c1,c2,...> --gpus 0,1,2,3
python layer_selection.py
python analysis_core.py && python features.py && python quality.py
python analysis_samplesize.py && python analysis_layers.py
python analysis_rq.py
```
