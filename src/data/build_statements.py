"""Per-language statement templates.

Turns an EntityFact into labelled declarative statements. Three things happen
here that cannot be fixed later:

1.  THE 2x2 IS GENERATED, NOT JUST TRUE/FALSE PAIRS.
2.  ENTITY SPANS ARE RECORDED at render time.
3.  SURFACE FORM IS MATCHED between the classes.

THE 2x2 (and why getting it wrong silently breaks RQ1)
------------------------------------------------------
Negation flips the truth value::

    affirmative + correct object   -> TRUE   ("Chennai is in Tamil Nadu")
    affirmative + wrong object     -> FALSE  ("Chennai is in Kerala")
    negated     + correct object   -> FALSE  ("Chennai is not in Tamil Nadu")
    negated     + wrong object     -> TRUE   ("Chennai is not in Kerala")

    label = correct_object XOR negated

All four cells are required. The common shortcut — negate only the true
statements — makes `negated` and `label` perfectly correlated, so the polarity
axis and the truth axis become the same vector. `subspace.orthonormalise` will
then raise on rank deficiency, which is the good outcome; the bad outcome is a
dataset where they are merely *highly* correlated, which yields an
ill-conditioned plane and a second principal angle that is pure noise while still
printing a plausible number.

`check_cell_balance` verifies all four cells are populated and roughly equal.

ENTITY SPANS
------------
`entity_last` pooling needs to know where the object entity sits. Spans are
recorded as CHARACTER offsets into the rendered string, because token offsets
depend on the tokenizer and this module is model-agnostic. Convert them with the
tokenizer's `offset_mapping` at extraction time (see `char_span_to_token_span`).

These cannot be recovered afterwards without re-parsing, so they are recorded
even when the current run uses `last` pooling.

MORPHOLOGY — THE REAL LIMIT OF TEMPLATING
------------------------------------------
Agglutinative and case-marking languages attach suffixes to the slot filler:
Tamil locative -il, Hindi postposition placement, Urdu izafat. A template that
concatenates a bare label produces text that is wrong or unidiomatic, and
unidiomatic target-language text is itself a confound — the probe may be reading
"this sentence is malformed" rather than "this proposition is false", and
malformedness will correlate with whichever class got the awkward filler.

The `suffix` field is a crude accommodation, not a solution. **Every template
below requires native-speaker review before generation.** The ones written here
are placeholders by a non-native author and should be treated as a schema to be
filled in, not as validated linguistic material. This is the cheapest possible
use of annotator time and the highest-leverage: a few hours reviewing templates
protects thousands of generated statements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "Statement",
    "Template",
    "TEMPLATES",
    "render",
    "generate_2x2",
    "check_cell_balance",
    "surface_asymmetry",
    "char_span_to_token_span",
]

AFFIRMATIVE = 0
NEGATED = 1


@dataclass(frozen=True)
class Statement:
    """One rendered statement with everything the pipeline needs downstream."""

    text: str
    label: int                       # 1 = true
    polarity: int                    # 0 = affirmative, 1 = negated
    language: str
    group_id: str                    # shared across all four cells of one fact
    template_id: str
    object_char_span: tuple[int, int]
    subject_char_span: tuple[int, int]
    subject_qid: str = ""
    object_qid: str = ""

    def __repr__(self) -> str:
        pol = "neg" if self.polarity else "aff"
        return f"Statement({self.text!r}, {'T' if self.label else 'F'}, {pol})"


@dataclass(frozen=True)
class Template:
    """A language-specific frame with subject and object slots.

    `affirmative` and `negated` must differ ONLY by the negation marker. If the
    negated form also reorders the sentence or changes the verb stem beyond
    negation, the polarity axis picks up word-order information as well and stops
    being interpretable as polarity.
    """

    template_id: str
    language: str
    property_pid: str
    affirmative: str                 # uses {subject} and {object}
    negated: str
    object_suffix: str = ""          # case marker glued to the object label
    subject_suffix: str = ""
    requires_native_review: bool = True

    def __post_init__(self) -> None:
        for form in (self.affirmative, self.negated):
            for slot in ("{subject}", "{object}"):
                if slot not in form:
                    raise ValueError(
                        f"{self.template_id}: {form!r} is missing {slot}"
                    )
        if self.affirmative == self.negated:
            raise ValueError(f"{self.template_id}: negated form is identical")


# ---------------------------------------------------------------------------
# PLACEHOLDER TEMPLATES — schema only. Native review required before use.
# ---------------------------------------------------------------------------
TEMPLATES: dict[str, list[Template]] = {
    "en": [
        Template("en.located.1", "en", "P131",
                 "{subject} is located in {object}.",
                 "{subject} is not located in {object}."),
    ],
    "es": [
        Template("es.located.1", "es", "P131",
                 "{subject} está en {object}.",
                 "{subject} no está en {object}."),
    ],
    "hi": [
        # SOV; negation particle before the verb.
        Template("hi.located.1", "hi", "P131",
                 "{subject} {object} में स्थित है।",
                 "{subject} {object} में स्थित नहीं है।"),
    ],
    "ur": [
        # Same syntax as Hindi, different script — the pair the design rests on.
        # Keeping the templates structurally parallel is what makes the hi/ur
        # contrast a script contrast rather than a template contrast.
        Template("ur.located.1", "ur", "P131",
                 "{subject} {object} میں واقع ہے۔",
                 "{subject} {object} میں واقع نہیں ہے۔"),
    ],
    "ta": [
        # Agglutinative: the locative attaches to the object. `object_suffix`
        # handles the simple case; sandhi and stem changes do not fit here and
        # are exactly what native review must catch.
        Template("ta.located.1", "ta", "P131",
                 "{subject} {object} அமைந்துள்ளது.",
                 "{subject} {object} அமைந்திருக்கவில்லை.",
                 object_suffix="ல்"),
    ],
}


def render(
    template: Template,
    subject_label: str,
    object_label: str,
    negated: bool,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Render a statement and return it with character spans for both entities.

    Spans are computed by locating the filled slots rather than by searching for
    the label text, so a label that also occurs elsewhere in the frame cannot
    produce a wrong span.
    """
    frame = template.negated if negated else template.affirmative
    subject_text = subject_label + template.subject_suffix
    object_text = object_label + template.object_suffix

    subject_start = frame.index("{subject}")
    text = frame.replace("{subject}", subject_text, 1)
    subject_span = (subject_start, subject_start + len(subject_text))

    object_start = text.index("{object}")
    text = text.replace("{object}", object_text, 1)
    object_span = (object_start, object_start + len(object_text))

    if text[slice(*object_span)] != object_text:
        raise RuntimeError(f"object span mismatch in {template.template_id}")
    if text[slice(*subject_span)] != subject_text:
        raise RuntimeError(f"subject span mismatch in {template.template_id}")

    return text, subject_span, object_span


def generate_2x2(
    template: Template,
    subject_label: str,
    correct_object: str,
    wrong_object: str,
    group_id: str,
    subject_qid: str = "",
    correct_qid: str = "",
    wrong_qid: str = "",
) -> list[Statement]:
    """All four truth x polarity cells for one fact.

    `wrong_object` must come from `wikidata.sample_distractors` — same class,
    comparable prominence. A randomly chosen object turns this into a
    plausibility dataset.

    All four statements share `group_id`, so the grouped split in
    `baselines.split_half_ceiling` keeps them together. Splitting them apart
    leaks near-identical surface forms across the two halves and inflates the
    ceiling.
    """
    if correct_object == wrong_object:
        raise ValueError(
            f"distractor equals the true object ({correct_object!r}); the false "
            "cells would be true"
        )

    statements = []
    for negated in (False, True):
        for correct in (True, False):
            obj = correct_object if correct else wrong_object
            text, subj_span, obj_span = render(template, subject_label, obj, negated)
            statements.append(
                Statement(
                    text=text,
                    label=int(correct != negated),      # XOR
                    polarity=NEGATED if negated else AFFIRMATIVE,
                    language=template.language,
                    group_id=group_id,
                    template_id=template.template_id,
                    subject_char_span=subj_span,
                    object_char_span=obj_span,
                    subject_qid=subject_qid,
                    object_qid=correct_qid if correct else wrong_qid,
                )
            )
    return statements


def check_cell_balance(statements: Sequence[Statement], tolerance: float = 0.1) -> dict:
    """Verify all four truth x polarity cells exist and are roughly equal.

    An empty or thin cell makes the truth and polarity axes collinear, which
    makes the second principal angle meaningless. Run this before extraction, not
    after the geometry comes out strange.
    """
    cells = {(lab, pol): 0 for lab in (0, 1) for pol in (AFFIRMATIVE, NEGATED)}
    for s in statements:
        cells[(s.label, s.polarity)] += 1

    total = sum(cells.values())
    expected = total / 4 if total else 0
    empty = [k for k, v in cells.items() if v == 0]
    skewed = [
        k for k, v in cells.items()
        if expected and abs(v - expected) / expected > tolerance
    ]

    return {
        "cells": {f"label={k[0]},polarity={k[1]}": v for k, v in cells.items()},
        "total": total,
        "empty_cells": empty,
        "skewed_cells": skewed,
        "ok": not empty and not skewed,
        "message": (
            "OK" if not empty and not skewed
            else f"empty={empty} skewed={skewed} — the truth and polarity axes "
                 "will be correlated and the 2D subspace ill-conditioned"
        ),
    }


def surface_asymmetry(statements: Sequence[Statement]) -> dict:
    """Measure whether true and false statements differ in surface form.

    Templates guarantee the frames match, but the swapped entity labels can
    differ in length, and length is a cue a probe can exploit. This cannot be
    eliminated — real place names differ in length — but it must be measured and
    reported. A large asymmetry means some of the probe's accuracy is reading
    string length rather than truth, and that artefact will not transfer across
    languages for reasons unrelated to representation.
    """
    true_lens = [len(s.text) for s in statements if s.label == 1]
    false_lens = [len(s.text) for s in statements if s.label == 0]
    if not true_lens or not false_lens:
        return {"ok": False, "message": "one class is empty"}

    mean_true = sum(true_lens) / len(true_lens)
    mean_false = sum(false_lens) / len(false_lens)
    pooled = (mean_true + mean_false) / 2
    relative = abs(mean_true - mean_false) / pooled if pooled else 0.0

    return {
        "mean_chars_true": mean_true,
        "mean_chars_false": mean_false,
        "relative_difference": relative,
        "ok": relative < 0.02,
        "message": (
            "OK" if relative < 0.02
            else f"{relative:.1%} length difference between classes — report this, "
                 "and consider length-matching the distractor pool"
        ),
    }


def char_span_to_token_span(
    offset_mapping: Sequence[tuple[int, int]],
    char_span: tuple[int, int],
) -> tuple[int, int]:
    """Map a character span to token indices using a tokenizer offset mapping.

    Call with `tokenizer(..., return_offsets_mapping=True)`. Special tokens have
    offsets (0, 0) and are skipped.

    Raises if no token overlaps the span — which happens when truncation cut the
    entity off, and would otherwise pool at an arbitrary position.
    """
    start_char, end_char = char_span
    indices = [
        i for i, (s, e) in enumerate(offset_mapping)
        if not (s == 0 and e == 0) and s < end_char and e > start_char
    ]
    if not indices:
        raise ValueError(
            f"no tokens overlap character span {char_span}; the entity was "
            "probably truncated at max_length"
        )
    return indices[0], indices[-1] + 1        # end exclusive
