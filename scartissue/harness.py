"""Test harness prepended to every task's test file.

Each check records a failure with:
  - msg:  human-readable, shown to the LLM ("expected X, got Y")
  - conv: hidden tag naming the codebase convention that was violated. The real
          LLM never sees it; the offline mock agent uses it, and metrics use it
          to report which conventions have been learned.
"""

HARNESS = r'''
import json, re, sys

_FAILS = []
_TOTAL = 0


def _load(path):
    ns = {}
    try:
        exec(compile(open(path).read(), "solution.py", "exec"), ns)
    except Exception as e:  # syntax errors, import errors, ...
        print(json.dumps({"passed": 0, "total": 1, "fails": [
            {"label": "load", "msg": f"solution failed to load: {type(e).__name__}: {e}", "conv": []}]}))
        sys.exit(1)
    return ns


S = _load(sys.argv[1])


def _fn(name):
    f = S.get(name)
    if not callable(f):
        raise NameError(f"function {name!r} is not defined")
    return f


def _same(a, b):
    """Type-aware equality: 1250 != 1250.0, True != 1."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _call(thunk):
    try:
        return True, thunk()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _fail(label, msg, conv):
    _FAILS.append({"label": label, "msg": msg, "conv": sorted(set(conv))})


def check(label, thunk, value, conv=()):
    """Expect a successful result whose value equals `value`."""
    global _TOTAL
    _TOTAL += 1
    want = {"ok": True, "value": value}
    ok, got = _call(thunk)
    if not ok:
        _fail(label, f"{label} raised {got}; expected {want!r}", ["result_envelope", *conv])
        return
    if _same(got, want):
        return
    tags = []
    if isinstance(got, dict) and "ok" in got:
        if got.get("ok") is not True:
            tags.append("result_envelope")
        if not _same(got.get("value"), value):
            tags += list(conv)
    else:
        tags.append("result_envelope")
        if not _same(got, value):
            tags += list(conv)
    _fail(label, f"{label}: expected {want!r}, got {got!r}", tags or list(conv))


def check_error(label, thunk):
    """Expect a failure result (never an exception)."""
    global _TOTAL
    _TOTAL += 1
    ok, got = _call(thunk)
    if not ok:
        _fail(label, f"{label} raised {got}; expected {{'ok': False, 'error': <code>}}", ["result_envelope"])
        return
    if not (isinstance(got, dict) and got.get("ok") is False and "error" in got):
        _fail(label, f"{label}: expected a failure result {{'ok': False, 'error': <code>}}, got {got!r}", ["result_envelope"])
        return
    if not re.fullmatch(r"[A-Z][A-Z0-9_]+", str(got["error"])):
        _fail(label, f"{label}: error {got['error']!r} is not a valid error code (e.g. 'INVALID_INPUT')", ["error_codes"])


def done():
    print(json.dumps({"passed": _TOTAL - len({f['label'] for f in _FAILS}), "total": _TOTAL, "fails": _FAILS}))
    sys.exit(0 if not _FAILS else 1)
'''
