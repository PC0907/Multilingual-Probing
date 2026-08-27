"""Wikidata SPARQL access for locality-filtered, natively-labelled entities.

This is where the native-data claim becomes real or stays aspirational. The
argument in the README is that content-disjoint statements can be generated at
probe scale without annotators, by querying entities *by locality* and reading
their labels *in the target language*. Both halves matter:

  * Locality filtering (P131 / P17) gives facts that are salient in the target
    language's region rather than facts translated from English sources.
  * Native labels mean the statement contains the form a speaker would actually
    write, not a transliteration of the English name.

An entity with no label in the target language is not usable. Dropping those
silently is the single easiest way to reintroduce exactly the bias this pipeline
exists to avoid: Wikidata's label coverage is itself skewed toward globally
prominent entities, so the surviving set drifts back toward the internationally
famous. `coverage_report` exists to make that drift visible and quantified, so it
can be stated in the paper rather than discovered by a reviewer.

DISTRACTOR SELECTION IS THE HARD PART
-------------------------------------
Generating true statements is easy. Generating *false* ones that isolate truth is
not.

If the false statement pairs a Kerala city with a randomly chosen region from
anywhere on Earth, the probe can separate the classes by noticing that the two
entities are implausible together — a type or plausibility signal, not a truth
signal. It will score well, transfer poorly, and the failure will be attributed
to language when it was a dataset artefact.

Distractors must therefore be drawn from the *same class and comparable
prominence* as the true value: another Indian state, not a Norwegian county.
`sample_distractors` enforces this and refuses when the candidate pool is too
small, rather than quietly widening it.

ENDPOINT ETIQUETTE
------------------
query.wikidata.org requires a descriptive User-Agent with contact details and
throttles aggressively. Results are cached to disk keyed by query hash; a rerun
of the same generation should hit the cache, not the endpoint. Set
WIKIDATA_USER_AGENT before use.

Note: the public endpoint is not reachable from every sandbox. Query
construction, parsing and distractor logic are all independently testable
offline; only `run_query` needs the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "EntityFact",
    "SPARQL_ENDPOINT",
    "build_locality_query",
    "parse_bindings",
    "run_query",
    "coverage_report",
    "sample_distractors",
]

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Properties used for statement generation. Keep this list explicit rather than
# accepting arbitrary PIDs: each property needs a hand-written template per
# language, so the set of usable properties is bounded by translation effort.
PROPERTIES = {
    "located_in": "P131",     # located in administrative territorial entity
    "country": "P17",
    "headquarters": "P159",
    "inception": "P571",
    "birthplace": "P19",
    "deathplace": "P20",
    "occupation": "P106",
    "population": "P1082",
}


@dataclass(frozen=True)
class EntityFact:
    """One (subject, property, object) triple with native-language labels.

    `subject_label` and `object_label` are in the TARGET language. If either is
    missing the fact is unusable, because a statement mixing a Tamil subject with
    an English object label is neither native nor a fair test — the probe could
    read the script switch instead of the truth value.
    """

    subject_qid: str
    subject_label: str
    property_pid: str
    object_qid: str
    object_label: str
    language: str
    sitelinks: int = 0          # prominence proxy, for matched distractors
    object_class_qid: str | None = None   # for same-class distractor sampling

    @property
    def group_id(self) -> str:
        """Stable id shared by every statement derived from this fact.

        Used by the grouped split in baselines.split_half_ceiling: the
        affirmative, its negation and any template variants must never land on
        opposite sides of a split.
        """
        return f"{self.subject_qid}:{self.property_pid}"


def build_locality_query(
    property_pid: str,
    region_qids: Sequence[str],
    language: str,
    limit: int = 2000,
    require_object_label: bool = True,
) -> str:
    """SPARQL for entities in given regions, with labels in `language`.

    Args:
        property_pid: e.g. "P131".
        region_qids: administrative regions or countries to restrict to. These
            define locality — the whole point of the query.
        language: BCP-47 code for the labels ("ta", "hi", "ur").
        limit: rows. Ask for more than needed; label coverage will cut it down.
        require_object_label: if False, rows missing an object label are returned
            so `coverage_report` can quantify what would have been lost. Set
            False for the audit pass, True for generation.

    The label service is deliberately NOT used. `wikibase:label` silently falls
    back to English when the requested language is missing, which would fill the
    dataset with English labels wearing a Tamil language tag — precisely the
    contamination this module exists to prevent. Explicit rdfs:label with a
    language filter fails loudly instead.
    """
    if not region_qids:
        raise ValueError("at least one region QID is required; an unrestricted "
                         "query defeats the purpose of locality filtering")
    for qid in region_qids:
        if not qid.startswith("Q") or not qid[1:].isdigit():
            raise ValueError(f"malformed QID: {qid!r}")
    if not property_pid.startswith("P") or not property_pid[1:].isdigit():
        raise ValueError(f"malformed PID: {property_pid!r}")

    regions = " ".join(f"wd:{q}" for q in region_qids)
    object_label_clause = (
        f'?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{language}")'
    )
    if not require_object_label:
        object_label_clause = f"OPTIONAL {{ {object_label_clause} }}"

    return f"""
SELECT ?subject ?subjectLabel ?object ?objectLabel ?objectClass ?sitelinks WHERE {{
  VALUES ?region {{ {regions} }}
  ?subject wdt:{property_pid} ?object .
  ?subject wdt:P131*/wdt:P17? ?region .

  ?subject rdfs:label ?subjectLabel .
  FILTER(LANG(?subjectLabel) = "{language}")
  {object_label_clause}

  OPTIONAL {{ ?object wdt:P31 ?objectClass . }}
  OPTIONAL {{ ?subject wikibase:sitelinks ?sitelinks . }}
}}
LIMIT {limit}
""".strip()


def parse_bindings(payload: dict, property_pid: str, language: str) -> list[EntityFact]:
    """Turn a SPARQL JSON response into EntityFacts, skipping unusable rows."""
    facts: list[EntityFact] = []
    for row in payload.get("results", {}).get("bindings", []):
        try:
            subject_uri = row["subject"]["value"]
            object_uri = row["object"]["value"]
            subject_label = row["subjectLabel"]["value"]
            object_label = row["objectLabel"]["value"]
        except KeyError:
            continue        # missing native label; counted by coverage_report

        object_class = row.get("objectClass", {}).get("value")
        facts.append(
            EntityFact(
                subject_qid=subject_uri.rsplit("/", 1)[-1],
                subject_label=subject_label,
                property_pid=property_pid,
                object_qid=object_uri.rsplit("/", 1)[-1],
                object_label=object_label,
                language=language,
                sitelinks=int(row.get("sitelinks", {}).get("value", 0) or 0),
                object_class_qid=object_class.rsplit("/", 1)[-1] if object_class else None,
            )
        )
    return facts


def run_query(
    query: str,
    cache_dir: str | Path = "data/raw/wikidata_cache",
    user_agent: str | None = None,
    max_retries: int = 3,
    timeout: int = 120,
) -> dict:
    """Execute a SPARQL query with disk caching and backoff.

    The cache key is the query hash, so an unchanged generation run costs no
    requests. Delete the cache directory to force a refresh — and note that
    Wikidata changes, so a regenerated dataset is not byte-identical to an older
    one. Record the cache alongside results for reproducibility.
    """
    import requests

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    cached = cache_dir / f"{key}.json"

    if cached.exists():
        return json.loads(cached.read_text())

    agent = user_agent or os.environ.get("WIKIDATA_USER_AGENT")
    if not agent:
        raise ValueError(
            "set WIKIDATA_USER_AGENT to something like "
            "'crosslingual-truth/0.1 (you@university.edu)'. The endpoint blocks "
            "anonymous bulk traffic and this is a condition of use, not a "
            "formality."
        )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": agent, "Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 2 ** (attempt + 3)))
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            cached.write_text(json.dumps(payload))
            return payload
        except Exception as exc:            # noqa: BLE001 — retry any transport error
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"query failed after {max_retries} attempts") from last_error


def coverage_report(
    payload: dict,
    property_pid: str,
    language: str,
) -> dict:
    """Quantify what native-label filtering discards.

    Run the audit query (`require_object_label=False`) through this before
    generating. The numbers belong in the paper: if Tamil label coverage is 40%
    and the surviving entities are systematically more prominent than the
    discarded ones, the "native" dataset is quietly biased toward
    internationally-known facts, and that limits what a null result can mean.
    """
    rows = payload.get("results", {}).get("bindings", [])
    kept, dropped = [], []
    for row in rows:
        has_labels = "subjectLabel" in row and "objectLabel" in row
        sitelinks = int(row.get("sitelinks", {}).get("value", 0) or 0)
        (kept if has_labels else dropped).append(sitelinks)

    total = len(rows)
    return {
        "language": language,
        "property": property_pid,
        "rows_returned": total,
        "usable": len(kept),
        "coverage": len(kept) / total if total else 0.0,
        "median_sitelinks_kept": float(np.median(kept)) if kept else 0.0,
        "median_sitelinks_dropped": float(np.median(dropped)) if dropped else 0.0,
        # If kept entities are markedly more prominent than dropped ones, label
        # coverage is acting as a prominence filter and the "local" dataset is
        # less local than it looks.
        "prominence_skew": (
            float(np.median(kept) - np.median(dropped)) if kept and dropped else 0.0
        ),
    }


def sample_distractors(
    fact: EntityFact,
    candidates: Iterable[EntityFact],
    n: int = 1,
    prominence_tolerance: float = 3.0,
    rng: np.random.Generator | None = None,
) -> list[EntityFact]:
    """Pick plausible wrong objects for a fact, matched on class and prominence.

    A distractor must be the same KIND of thing as the true object and of
    comparable prominence. Otherwise the probe learns to detect implausible
    pairings rather than false propositions, and the resulting "truth direction"
    is a plausibility direction that will not transfer for reasons having nothing
    to do with language.

    Args:
        fact: the true fact.
        candidates: pool to draw from — normally other facts retrieved by the
            same query, so they share class and locality by construction.
        n: distractors wanted.
        prominence_tolerance: allowed ratio between candidate and true sitelink
            counts, in either direction.
        rng: for reproducibility.

    Raises:
        ValueError: if fewer than `n` matched candidates exist. Deliberate — the
            alternative is silently relaxing the constraint, which produces a
            dataset that looks fine and measures the wrong thing.
    """
    rng = rng or np.random.default_rng(0)
    base = max(fact.sitelinks, 1)

    pool = [
        c for c in candidates
        if c.object_qid != fact.object_qid                       # actually wrong
        and c.subject_qid != fact.subject_qid                    # not the same subject
        and c.object_class_qid == fact.object_class_qid          # same kind of thing
        and c.language == fact.language
        and (1 / prominence_tolerance) <= (max(c.sitelinks, 1) / base) <= prominence_tolerance
    ]

    # Unique object labels only: two candidates naming the same place give
    # duplicate false statements, which inflates one region in the negatives.
    seen, unique = set(), []
    for c in pool:
        if c.object_label not in seen:
            seen.add(c.object_label)
            unique.append(c)

    if len(unique) < n:
        raise ValueError(
            f"only {len(unique)} matched distractors for {fact.subject_label} "
            f"({fact.subject_qid}); need {n}. Widen the region set or retrieve "
            "more candidates — do NOT relax the class or prominence match, which "
            "would let the probe separate on plausibility instead of truth."
        )

    picked = rng.choice(len(unique), size=n, replace=False)
    return [unique[i] for i in picked]
