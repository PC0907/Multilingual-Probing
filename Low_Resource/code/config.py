"""Shared configuration for the low-resource / script extension (research_n).

Everything that defines a condition lives here: the surface form (language + script),
the judgment prompt used in that surface form, and the metadata used by the RQ2
regressions. The models, revisions, split, groups and probe are the ones used in
nabeel_naacl.pdf; only the language/script conditions are new.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ACTS = ROOT / "acts"            # symlink to /mount/studenten-temp1/.../research_n/acts
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
HF_HOME = "/mount/studenten-temp1/users/shahidmm/.hfcache"

# ---------------------------------------------------------------- models (as in the paper)
# Gemma: google/gemma-7b is gated and no HF token exists on this machine; unsloth/gemma-7b
# ships the same four safetensors shards (byte-identical sizes to google/gemma-7b@ff6768d).
MODELS = {
    "gemma-7b":        dict(repo="unsloth/gemma-7b", revision="main", paper_block=18),
    "qwen3-8b-base":   dict(repo="Qwen/Qwen3-8B-Base", revision="49e3418", paper_block=21),
    "apertus-8b-2509": dict(repo="swiss-ai/Apertus-8B-2509", revision="3162c99", paper_block=15),
    "mistral-7b-v0.3": dict(repo="mistralai/Mistral-7B-v0.3", revision="caa1feb", paper_block=15),
}
MODEL_LABEL = {"gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B-Base",
               "apertus-8b-2509": "Apertus-8B-2509", "mistral-7b-v0.3": "Mistral-7B-v0.3"}

# ---------------------------------------------------------------- conditions
# lang: ISO 639-1 of the underlying language; script: ISO 15924; kind:
#   native   = translation written in the language's normal script
#   translit = the *same sentence* as a native condition, re-written in another script
#   (base = the native condition it is a transliteration of)
CONDITIONS = {
    # original six (paper)
    "en":      dict(lang="en", script="Latn", kind="native"),
    "de":      dict(lang="de", script="Latn", kind="native"),
    "fr":      dict(lang="fr", script="Latn", kind="native"),
    "es":      dict(lang="es", script="Latn", kind="native"),
    "ar":      dict(lang="ar", script="Arab", kind="native"),
    "hi":      dict(lang="hi", script="Deva", kind="native"),
    # new low-resource Indic languages
    "ur":      dict(lang="ur", script="Arab", kind="native"),
    "mr":      dict(lang="mr", script="Deva", kind="native"),
    "ne":      dict(lang="ne", script="Deva", kind="native"),
    "gu":      dict(lang="gu", script="Gujr", kind="native"),
    "pa":      dict(lang="pa", script="Guru", kind="native"),
    # script interventions (same sentence, different script)
    "ur_Deva": dict(lang="ur", script="Deva", kind="translit", base="ur"),
    "ur_Latn": dict(lang="ur", script="Latn", kind="translit", base="ur"),
    "pa_Arab": dict(lang="pa", script="Arab", kind="translit", base="pa"),
    "pa_Deva": dict(lang="pa", script="Deva", kind="translit", base="pa"),
    "gu_Deva": dict(lang="gu", script="Deva", kind="translit", base="gu"),
    "mr_Gujr": dict(lang="mr", script="Gujr", kind="translit", base="mr"),
    "hi_Latn": dict(lang="hi", script="Latn", kind="translit", base="hi"),
}
ORIGINAL6 = ["en", "de", "ar", "hi", "fr", "es"]
NATIVE = [c for c, m in CONDITIONS.items() if m["kind"] == "native"]

# ---------------------------------------------------------------- judgment prompts
# One judgment prompt per condition, written in that condition's language *and script*
# (a transliterated condition gets the transliterated prompt, so the whole input changes
# script). The representation is read at the final token of the statement.
PROMPTS = {
    "en": "Decide whether the following statement is true or false.\nStatement: ",
    "de": "Entscheide, ob die folgende Aussage wahr oder falsch ist.\nAussage: ",
    "fr": "Décide si l'affirmation suivante est vraie ou fausse.\nAffirmation : ",
    "es": "Decide si la siguiente afirmación es verdadera o falsa.\nAfirmación: ",
    "ar": "حدّد ما إذا كانت العبارة التالية صحيحة أم خاطئة.\nالعبارة: ",
    "hi": "तय कीजिए कि निम्नलिखित कथन सही है या ग़लत।\nकथन: ",
    "hi_Latn": "Tay kijiye ki nimnalikhit kathan sahi hai ya galat.\nKathan: ",
    "ur": "فیصلہ کیجیے کہ درج ذیل بیان درست ہے یا غلط۔\nبیان: ",
    "ur_Deva": "फ़ैसला कीजिए कि दर्ज ज़ैल बयान दुरुस्त है या ग़लत।\nबयान: ",
    "ur_Latn": "Faisla kijiye ke darj zail bayan durust hai ya ghalat.\nBayan: ",
    "mr": "पुढील विधान खरे आहे की खोटे ते ठरवा.\nविधान: ",
    "ne": "तलको कथन सत्य हो वा असत्य, निर्णय गर्नुहोस्।\nकथन: ",
    "gu": "નીચેનું વિધાન સાચું છે કે ખોટું તે નક્કી કરો.\nવિધાન: ",
    "pa": "ਫ਼ੈਸਲਾ ਕਰੋ ਕਿ ਹੇਠ ਲਿਖਿਆ ਕਥਨ ਸੱਚ ਹੈ ਜਾਂ ਝੂਠ।\nਕਥਨ: ",
    "pa_Arab": "فیصلہ کرو کہ ہیٹھ لکھیا کتھن سچ ہے یاں جھوٹھ۔\nکتھن: ",
    # deterministic Brahmic transliterations of the base prompts (indic_transliteration)
    "pa_Deva": None, "gu_Deva": None, "mr_Gujr": None,
}


def _fill_translit_prompts():
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    PROMPTS["pa_Deva"] = transliterate(PROMPTS["pa"], sanscript.GURMUKHI, sanscript.DEVANAGARI)
    PROMPTS["gu_Deva"] = transliterate(PROMPTS["gu"], sanscript.GUJARATI, sanscript.DEVANAGARI)
    PROMPTS["mr_Gujr"] = transliterate(PROMPTS["mr"], sanscript.DEVANAGARI, sanscript.GUJARATI)


_fill_translit_prompts()

SEED = 20261006
