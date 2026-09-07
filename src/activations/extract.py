"""Activation extraction and caching.

One forward pass yields every layer at once, so a layer sweep costs disk, not
compute. The whole project is forward passes only — no gradients, no base-model
training — which is why compute is not the bottleneck and data is.

Three decisions are made explicit here rather than buried in defaults, because
each of them can manufacture or destroy a result.

TOKEN POSITION
--------------
An activation is [n_tokens, d] but a probe needs [d]. Which token?

Orgad et al. (ICLR 2025) show truth information is not uniformly spread: it
concentrates on specific tokens, notably the last token of the answer entity, and
mean-pooling over the sequence dilutes it. The default here is `last`, matching
Marks & Tegmark, but the choice interacts badly with multilinguality:

  * `last` on a right-branching language and `last` on a left-branching one are
    not the same linguistic position. Hindi and Tamil are verb-final; English is
    not. "Last token" may be a verb in one language and an object in another.
  * Tokenizer fertility means the last token of a Tamil word may be a
    word-internal fragment carrying little semantic content, while the English
    last token is a whole word.

This is a real confound for RQ1 and there is no clean fix. The honest response is
to run the geometry at more than one pooling choice and report whether the
conclusion survives. `entity_last` exists for that: it pools at the final token
of a marked span, which is linguistically comparable across word orders in a way
that sequence-final is not. Building the datasets with entity spans recorded
costs nothing at generation time and cannot be recovered afterwards.

PRECISION
---------
Activations are cached as float32 even when the model runs in bfloat16. bf16 has
roughly 8 bits of mantissa; a mass-mean direction is a difference of two nearly
equal means, which is exactly the operation that loses precision catastrophically
when the inputs are coarse. The cast costs disk and buys correctness.

NORMALISATION
-------------
Nothing is normalised here. Layer-norm scaling, mean-centring and whitening are
analysis choices that belong to the probe, not the cache, and baking one in makes
the cache unusable for the variants that need a different one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np

__all__ = [
    "ExtractionConfig",
    "ActivationCache",
    "pool_tokens",
    "extract_activations",
]

Pooling = Literal["last", "mean", "entity_last", "all"]


@dataclass(frozen=True)
class ExtractionConfig:
    """Everything that must match between two caches for them to be comparable."""

    model_id: str
    pooling: Pooling = "last"
    layers: tuple[int, ...] | None = None      # None = every layer
    max_length: int = 128
    batch_size: int = 16
    dtype: str = "float32"
    inference_dtype: str = "unknown"
    add_bos: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def assert_compatible(self, other: "ExtractionConfig") -> None:
        """Refuse to compare caches built differently.

        Cross-lingual alignment between a cache pooled at `last` and one pooled
        at `mean` is a comparison of two different quantities. It will still
        produce a number.
        """
        for field in ("model_id", "pooling", "max_length", "add_bos", "inference_dtype"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ValueError(
                    f"incompatible caches: {field} is {mine!r} vs {theirs!r}. "
                    "Directions from differently-extracted activations are not "
                    "comparable."
                )


def pool_tokens(
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    pooling: Pooling = "last",
    entity_spans: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Reduce [batch, seq, d] to [batch, d].

    Args:
        hidden: [batch, seq, d] hidden states for one layer.
        attention_mask: [batch, seq], 1 for real tokens. Required — with
            left-padded batches the sequence-final position is padding, and
            pooling there yields the pad embedding for every example, which
            produces a probe that is confidently reading nothing.
        pooling: strategy.
        entity_spans: [(start, end)] per example, required for `entity_last`.
            End is exclusive.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    attention_mask = np.asarray(attention_mask)

    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, seq, d], got {hidden.shape}")
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"mask {attention_mask.shape} does not match hidden {hidden.shape[:2]}"
        )
    if not attention_mask.any(axis=1).all():
        raise ValueError("at least one sequence is entirely padding")

    if pooling == "mean":
        weights = attention_mask[..., None].astype(np.float32)
        return (hidden * weights).sum(axis=1) / weights.sum(axis=1)

    if pooling == "last":
        # LEFT padding means the final real token is the final column, for every
        # row. Do NOT use sum(mask) - 1: that is right-padding logic, and under
        # left padding it lands squarely inside the padding, returning the pad
        # embedding for every example. The probe then trains on a constant and
        # reports whatever the label prior is.
        if not attention_mask[:, -1].all():
            raise ValueError(
                "the last column contains padding, so this batch is not "
                "left-padded. Set tokenizer.padding_side = 'left'."
            )
        return hidden[:, -1]

    if pooling == "entity_last":
        if entity_spans is None:
            raise ValueError(
                "entity_last pooling needs entity_spans. Record them when the "
                "statements are generated — they cannot be recovered later."
            )
        if len(entity_spans) != hidden.shape[0]:
            raise ValueError("one entity span required per example")
        idx = np.array([end - 1 for _, end in entity_spans])
        rows = np.arange(hidden.shape[0])
        if (idx < 0).any() or (idx >= hidden.shape[1]).any():
            raise ValueError("an entity span index is outside the sequence length")
        # Padding-agnostic: check the chosen position is a real token. Spans must
        # be indices into the PADDED sequence, which for left padding means they
        # shift per example with the pad count.
        if not attention_mask[rows, idx].all():
            raise ValueError(
                "an entity span points at padding — spans must be indices into "
                "the padded batch, not the unpadded sequence, and must be token "
                "indices rather than character offsets. Also check that "
                "truncation at max_length has not cut the entity off."
            )
        return hidden[rows, idx]

    raise ValueError(f"unknown pooling {pooling!r}")


class ActivationCache:
    """On-disk activations for one (model, language, dataset) triple.

    Layout::

        cache_dir/
          meta.json            config + provenance
          layer_00.npy         [n, d] float32
          layer_01.npy
          ...
          index.parquet        statement ids, labels, polarity, groups

    Layers are separate files so that a layer sweep can be memory-mapped one
    layer at a time. A 7B model over 5000 statements is roughly 30 GB across all
    layers in float32; loading it whole is the difference between the analysis
    running and the machine swapping.
    """

    META = "meta.json"

    def __init__(self, cache_dir: str | Path, config: ExtractionConfig | None = None):
        self.dir = Path(cache_dir)
        self._config = config

    # -- writing ----------------------------------------------------------

    def write_layer(self, layer: int, activations: np.ndarray) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"layer_{layer:02d}.npy"
        np.save(path, np.asarray(activations, dtype=np.float32))
        return path

    def write_meta(self, extra: dict | None = None) -> Path:
        if self._config is None:
            raise ValueError("no config to write")
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"config": self._config.to_dict(), **(extra or {})}
        path = self.dir / self.META
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    # -- reading ----------------------------------------------------------

    @property
    def config(self) -> ExtractionConfig:
        if self._config is None:
            meta = json.loads((self.dir / self.META).read_text())
            self._config = ExtractionConfig(**meta["config"])
        return self._config

    def available_layers(self) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1]) for p in self.dir.glob("layer_*.npy")
        )

    def load_layer(self, layer: int, mmap: bool = True) -> np.ndarray:
        path = self.dir / f"layer_{layer:02d}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; available layers: {self.available_layers()}"
            )
        return np.load(path, mmap_mode="r" if mmap else None)

    def iter_layers(self, mmap: bool = True) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (layer, activations) one at a time, never holding two."""
        for layer in self.available_layers():
            yield layer, self.load_layer(layer, mmap=mmap)


def extract_activations(
    statements: Sequence[str],
    model,
    tokenizer,
    config: ExtractionConfig,
    cache: ActivationCache | None = None,
    entity_spans: Sequence[tuple[int, int]] | None = None,
) -> dict[int, np.ndarray]:
    """Run forward passes and pool one vector per statement per layer.

    Args:
        statements: raw text, already in the target language.
        model: a HF causal LM. Must be in eval mode; caller's responsibility.
        tokenizer: matching tokenizer. Left-padded — with right padding the
            `last` position is a pad token.
        config: extraction settings.
        cache: if given, layers are written as they are produced rather than
            accumulated in memory.
        entity_spans: token-index spans, required for `entity_last` pooling.

    Returns:
        {layer_index: [n, d] float32}. Empty if `cache` was supplied, since the
        point of caching is not to hold everything at once.
    """
    import torch  # local import: the geometry modules must not need torch

    if tokenizer.padding_side != "left":
        raise ValueError(
            "tokenizer must be left-padded, or `last` pooling reads a pad token"
        )
    if model.training:
        raise ValueError("model is in training mode; call model.eval()")

    collected: dict[int, list[np.ndarray]] = {}
    n = len(statements)

    for start in range(0, n, config.batch_size):
        batch = list(statements[start : start + config.batch_size])
        spans = (
            entity_spans[start : start + config.batch_size]
            if entity_spans is not None
            else None
        )

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
            add_special_tokens=config.add_bos,
        ).to(model.device)

        with torch.inference_mode():
            out = model(**encoded, output_hidden_states=True)

        mask = encoded["attention_mask"].cpu().numpy()
        layers = (
            range(len(out.hidden_states))
            if config.layers is None
            else config.layers
        )

        for layer in layers:
            # float32 before pooling: the mean of bf16 values accumulates error
            # that a difference-of-means then amplifies.
            hidden = out.hidden_states[layer].to(torch.float32).cpu().numpy()
            pooled = pool_tokens(hidden, mask, config.pooling, spans)
            collected.setdefault(layer, []).append(pooled)

        del out

    stacked = {layer: np.concatenate(parts) for layer, parts in collected.items()}

    if cache is not None:
        for layer, arr in stacked.items():
            cache.write_layer(layer, arr)
        cache.write_meta({"n_statements": n})
        return {}
    return stacked
