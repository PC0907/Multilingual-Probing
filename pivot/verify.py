"""RQ3 prerequisite: does the latent-English pivot exist in these models?

Wendler et al. (ACL 2024) report that multilingual models appear to route through
an English-like latent space in middle layers: when a non-English next token is
being predicted, decoding intermediate layers with the output embedding yields
English tokens before the target-language token wins near the end.

RQ3 asks whether cross-lingual truth-probe transfer is MEDIATED by that pivot. If
the pivot does not exist in the models under study, RQ3 has no object and should
be cut. Wendler characterised it on Llama-2 specifically, and newer tokenizers
and training mixes may not reproduce it. Schut et al. dispute the interpretation
even where the phenomenon replicates.

So this is a preliminary result in its own right, and cheap: it needs no probes,
no labelled data, and no truth statements — just parallel word pairs and forward
passes.

THE METHOD (logit lens on a cloze task)
----------------------------------------
Build a few-shot prompt that establishes a translation pattern, then leave the
target word blank::

    Français: "fleur" - Tamil: "பூ"
    Français: "montagne" - Tamil: "மலை"
    Français: "chat" - Tamil: "

The model must produce the Tamil word. Decode every intermediate layer through
the unembedding matrix and ask, at each depth, whether probability mass sits on
the English word ("cat"), the target-language word, or neither.

A pivot looks like: English probability rising in middle layers, then falling as
target-language probability takes over near the output.

FOUR WAYS THIS MEASUREMENT LIES, AND WHAT IS DONE ABOUT THEM
-------------------------------------------------------------
1.  NO LAYER NORM. Applying the unembedding to a raw residual stream without the
    model's final layer norm gives distorted logits, and the distortion is
    depth-dependent — exactly the axis being measured. `decode_layer` requires
    the norm module and refuses to guess.

2.  MULTI-TOKEN WORDS. "cat" may be one token while "பூ" is four. Comparing the
    probability of a single token against the first token of a four-token word is
    not a like-for-like comparison, and Tamil will look artificially weak because
    its first token is a fragment shared with many other words. `token_id_sets`
    reports token counts so single-token pairs can be selected, which is the only
    clean comparison available.

3.  THE ENGLISH BASELINE IS NOT NEUTRAL. English tokens are more frequent in
    training and have larger unembedding norms, so they attract probability mass
    at every layer for reasons unrelated to pivoting. Comparing English against
    the target language alone will find "English" even in a model that does not
    pivot. `control_languages` adds a third language that is neither the source
    nor the target: if French rises in middle layers just as English does, the
    effect is a frequency artefact, not a pivot.

4.  THE PROMPT IS IN A LANGUAGE. If the few-shot examples are written in English,
    an English-shaped intermediate state may be the prompt rather than a pivot.
    Use a non-English source language in the prompt, as Wendler did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "WordPair",
    "PivotResult",
    "build_cloze_prompt",
    "token_id_sets",
    "decode_layer",
    "measure_pivot",
    "summarise_pivot",
]


@dataclass(frozen=True)
class WordPair:
    """One concept in several languages. Keys are language codes."""

    forms: dict[str, str]

    def __getitem__(self, language: str) -> str:
        return self.forms[language]


@dataclass
class PivotResult:
    """Per-layer probability mass by language, for one prompt."""

    layer_probs: dict[str, np.ndarray]     # language -> [n_layers]
    n_layers: int
    target_language: str
    source_language: str
    control_languages: tuple[str, ...]

    def peak_layer(self, language: str) -> int:
        return int(np.argmax(self.layer_probs[language]))

    def pivot_score(self, pivot_language: str = "en") -> float:
        """How much the pivot language leads the target, in middle layers.

        Positive means the pivot language dominates before the target does —
        the signature Wendler reports. Computed over the middle third only,
        because early layers carry the prompt and late layers necessarily
        favour the target (that is the answer).
        """
        n = self.n_layers
        middle = slice(n // 3, 2 * n // 3)
        pivot = self.layer_probs[pivot_language][middle].mean()
        target = self.layer_probs[self.target_language][middle].mean()
        return float(pivot - target)

    def control_adjusted_score(self, pivot_language: str = "en") -> float:
        """Pivot score minus the best control language's score.

        This is the number to report. An unadjusted score finds "English" in any
        model where English tokens are simply more probable everywhere. If a
        control language scores as highly as English, there is no pivot — there
        is a frequency effect.
        """
        if not self.control_languages:
            return float("nan")
        n = self.n_layers
        middle = slice(n // 3, 2 * n // 3)
        target = self.layer_probs[self.target_language][middle].mean()
        controls = [
            self.layer_probs[c][middle].mean() - target
            for c in self.control_languages
        ]
        return self.pivot_score(pivot_language) - float(max(controls))


def build_cloze_prompt(
    examples: list[WordPair],
    query: WordPair,
    source_language: str,
    target_language: str,
    source_name: str,
    target_name: str,
) -> str:
    """Few-shot translation prompt with the answer left open.

    `source_language` should NOT be English. If the prompt is English, an
    English-looking intermediate state may simply be the prompt persisting rather
    than the model routing through English.
    """
    if source_language == "en":
        raise ValueError(
            "the prompt's source language must not be English, or an "
            "English-shaped intermediate state is uninterpretable — it could be "
            "the prompt rather than a pivot"
        )

    lines = [
        f'{source_name}: "{pair[source_language]}" - {target_name}: "{pair[target_language]}"'
        for pair in examples
    ]
    lines.append(f'{source_name}: "{query[source_language]}" - {target_name}: "')
    return "\n".join(lines)


def token_id_sets(
    word_pair: WordPair,
    tokenizer,
    languages: list[str],
    with_leading_space: bool = False,
) -> dict[str, dict]:
    """First-token ids and token counts for each language's form of a word.

    Returns counts so multi-token words can be filtered out. Comparing a
    single-token English word against the first token of a four-token Tamil word
    is not a comparison of languages; it is a comparison of tokenizer
    fragmentation, and it will manufacture a pivot in every high-fertility
    language.
    """
    out = {}
    for language in languages:
        form = word_pair.forms.get(language)
        if form is None:
            continue
        text = (" " + form) if with_leading_space else form
        ids = tokenizer.encode(text, add_special_tokens=False)
        out[language] = {
            "first_token_id": ids[0] if ids else None,
            "n_tokens": len(ids),
            "form": form,
            "single_token": len(ids) == 1,
        }
    return out


def decode_layer(
    hidden: np.ndarray,
    unembedding: np.ndarray,
    final_norm,
) -> np.ndarray:
    """Logit-lens decode of one layer's residual stream.

    Args:
        hidden: [d] residual stream at the final position of one layer.
        unembedding: [vocab, d] output embedding matrix.
        final_norm: the model's final layer-norm module, applied before
            unembedding. Required, not optional: skipping it produces logits
            whose distortion grows with depth, and depth is the axis under study.
            Passing None raises rather than silently producing a curve that looks
            like a pivot.

    Returns:
        [vocab] softmax probabilities.
    """
    if final_norm is None:
        raise ValueError(
            "final_norm is required. Decoding a raw residual stream without the "
            "model's final layer norm yields depth-dependent distortion, which "
            "is indistinguishable from the depth-dependent effect being measured."
        )

    import torch

    with torch.inference_mode():
        tensor = torch.as_tensor(hidden, dtype=torch.float32)
        normed = final_norm(tensor)
        logits = normed @ torch.as_tensor(unembedding, dtype=torch.float32).T
        return torch.softmax(logits, dim=-1).cpu().numpy()


def measure_pivot(
    prompt: str,
    word_pair: WordPair,
    model,
    tokenizer,
    target_language: str,
    source_language: str,
    control_languages: tuple[str, ...] = ("fr",),
    require_single_token: bool = True,
) -> PivotResult:
    """Per-layer probability of each language's form of the answer word.

    Args:
        control_languages: languages that are neither the prompt's source nor the
            target. Their curves are the null hypothesis: if a control rises in
            middle layers exactly as English does, the "pivot" is token frequency.
        require_single_token: skip word pairs where any language's form is
            multi-token. Strict, and it will discard many candidate words —
            which is the cost of a comparison that means something.
    """
    import torch

    languages = [target_language, "en", *control_languages]
    tokens = token_id_sets(word_pair, tokenizer, languages)

    missing = [l for l in languages if l not in tokens]
    if missing:
        raise ValueError(f"word pair has no form for {missing}")

    if require_single_token:
        multi = {l: t["n_tokens"] for l, t in tokens.items() if not t["single_token"]}
        if multi:
            raise ValueError(
                f"multi-token forms {multi}; comparing first tokens across "
                "languages with different fragmentation measures the tokenizer, "
                "not the model. Choose words that are single tokens in every "
                "language, or accept a much smaller word list."
            )

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True)

    unembedding = model.get_output_embeddings().weight.detach()
    final_norm = getattr(getattr(model, "model", model), "norm", None)
    if final_norm is None:
        raise ValueError(
            "could not locate the model's final layer norm; pass it explicitly "
            "rather than decoding without it"
        )

    n_layers = len(outputs.hidden_states)
    layer_probs = {language: np.zeros(n_layers) for language in languages}

    for layer in range(n_layers):
        hidden = outputs.hidden_states[layer][0, -1].to(torch.float32)
        probabilities = decode_layer(
            hidden.cpu().numpy(), unembedding.cpu().numpy(), final_norm.cpu()
        )
        for language in languages:
            token_id = tokens[language]["first_token_id"]
            layer_probs[language][layer] = probabilities[token_id]

    return PivotResult(
        layer_probs=layer_probs,
        n_layers=n_layers,
        target_language=target_language,
        source_language=source_language,
        control_languages=control_languages,
    )


def summarise_pivot(results: list[PivotResult], pivot_language: str = "en") -> dict:
    """Aggregate across word pairs and decide whether the pivot replicates.

    The verdict is deliberately conservative. RQ3 is an expensive experiment and
    should not be launched on a marginal effect; a negative here is a cheap,
    reportable preliminary result and a legitimate reason to cut RQ3.
    """
    if not results:
        raise ValueError("no results to summarise")

    raw = np.array([r.pivot_score(pivot_language) for r in results])
    adjusted = np.array([r.control_adjusted_score(pivot_language) for r in results])
    adjusted = adjusted[~np.isnan(adjusted)]

    mean_adjusted = float(adjusted.mean()) if adjusted.size else float("nan")
    # Fraction of words where the effect points the same way — a mean driven by a
    # few outlier words is not a pivot.
    consistency = float((adjusted > 0).mean()) if adjusted.size else float("nan")

    replicates = bool(
        adjusted.size and mean_adjusted > 0.01 and consistency > 0.7
    )

    return {
        "n_words": len(results),
        "mean_raw_score": float(raw.mean()),
        "mean_control_adjusted_score": mean_adjusted,
        "consistency": consistency,
        "replicates": replicates,
        "verdict": (
            "Pivot replicates. RQ3's ablation experiment has an object."
            if replicates else
            "Pivot does NOT replicate above the control-language baseline in "
            "this model. Report as a preliminary negative result — Wendler "
            "characterised it on Llama-2 and it may not survive newer training "
            "mixes — and cut RQ3 rather than ablating a subspace that is not "
            "doing the work."
        ),
    }
