"""Failure classifier: tags a test failure from its *visible* messages.

This is the build/test-error explainer role: it reads only what the model
is allowed to see (expected-vs-got messages, crash text) -- never the hidden
convention tags -- and assigns a coarse category plus domain tags. The loop
stores the category on each distilled lesson and passes the tags to the
distiller, so lessons are searchable by the kind of failure that produced
them and the dashboard can group the graveyard by failure class.
"""

import re

_DATE_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_US = re.compile(r"\d{2}/\d{2}/\d{4}")
_ID_OK = re.compile(r"\b[a-z]{2,4}_\d{6}\b")
_ID_BAD = re.compile(r"\b[A-Z]{2,4}-\d+\b")
_MONEY_FLOAT = re.compile(r"\$\d+\.\d+|\b\d+\.\d+\b")


def classify(feedback):
    """Return {"category", "tags", "explanation"} for one failure's feedback."""
    text = feedback or ""
    low = text.lower()
    tags = []

    if "timed out" in low or "timeout" in low:
        category, why = "timeout", "the solution did not finish within the time limit"
    elif "failed to load" in low or "test run crashed" in low:
        category, why = "load_error", "the solution did not load or the test run crashed"
    elif "not a valid error code" in low:
        category, why = "error_code", "a failure result carried a non UPPER_SNAKE error code"
    elif "raised" in low or "expected a failure result" in low or "{'ok': false" in low:
        category, why = "envelope", "the function raised or returned the wrong result shape"
    else:
        category, why = "value_mismatch", "the returned value did not match the expected value"

    # Domain tags inferred from shapes visible in expected-vs-got text.
    if _DATE_US.search(text) or (_DATE_ISO.search(text) and category == "value_mismatch"):
        tags.append("dates")
    if _ID_BAD.search(text):
        tags += ["ids", "id_format"]
    elif _ID_OK.search(text):
        tags.append("ids")
    if _MONEY_FLOAT.search(text) or "$" in text or "amount" in low or "cents" in low:
        tags.append("money")
    if "expected {'ok'" in low or "expected a failure result" in low:
        tags.append("envelope")
    if "'error'" in low or "error code" in low:
        tags.append("errors")
    # A quoted multi-word string whose expected form is Title Case hints names.
    if re.search(r"'[A-Z][a-z]+ [A-Z][a-z]+'", text):
        tags.append("names")
    if re.search(r"\['[a-z]{2,4}_\d{6}'", text):
        tags.append("sorting")

    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return {"category": category, "tags": uniq[:4], "explanation": why}
