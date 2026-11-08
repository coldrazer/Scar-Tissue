"""The coding agent: solve tasks with injected lessons, and distill new lessons from failures.

Two backends with the same interface:
  LLMAgent  - real models over OpenAI-compatible APIs (Meta Llama API, Gemini, SubQ, ...)
  MockAgent - an offline simulation so the whole loop (and the demo) runs without keys.
"""

import hashlib
import json
import random
import re

from . import benchmark, classifier, llm

# ---------------------------------------------------------------------------
# Real LLM agent
# ---------------------------------------------------------------------------

SOLVER_SYSTEM = """You are a coding agent working inside "Acme Ledger", an existing Python billing codebase.
Implement exactly what the task asks, following the codebase's conventions.
Reply with ONE ```python code block containing the complete implementation (imports + function).
No explanations, no tests, no example usage."""

LESSONS_HEADER = ("Lessons learned from past mistakes in THIS codebase. Follow them; they override "
                  "generic best practices:")

DISTILL_SYSTEM = """You maintain the memory of a coding agent that works in one codebase ("Acme Ledger").
The agent just failed the codebase's tests. Extract the codebase house rule(s) it violated: rules that
would prevent similar failures on OTHER tasks in the same codebase.

Rules for lessons:
- One or two sentences each, concrete, with a tiny example. No vague advice ("be careful", "add tests").
- Generalize beyond this task: do not name this task's function.
- Only state what the evidence (expected vs got) supports.
- scope "global" if it applies to essentially every function (return shape, error conventions);
  "specific" if it applies to a kind of data (money, dates, ids, names, lists, splitting...).
- At most 2 lessons. Do not repeat lessons the agent already had.

Reply with JSON only:
{"lessons": [{"text": "...", "scope": "global" | "specific", "tags": ["2-4 short keywords"]}]}"""


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def lessons_block(lessons):
    if not lessons:
        return ""
    return LESSONS_HEADER + "\n" + "\n".join(f"{i}. {l.text}" for i, l in enumerate(lessons, 1))


class LLMAgent:
    def __init__(self, cfg):
        self.cfg = cfg

    def solve(self, task, lessons, history, seed_key=""):
        """history: [(code, feedback)] from earlier attempts on this task."""
        system = SOLVER_SYSTEM
        block = lessons_block(lessons)
        if block:
            system += "\n\n" + block
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]
        for code, feedback in history:
            msgs.append({"role": "assistant", "content": f"```python\n{code}```"})
            msgs.append({"role": "user", "content": "Your solution failed the codebase's tests:\n"
                         f"{feedback}\nFix it. Reply with the full corrected code block."})
        out = llm.chat(self.cfg.solver, msgs, temperature=self.cfg.temperature)
        return extract_code(out)

    def distill(self, task, lessons, failed_code, feedback, fixed_code=None, seed_key=""):
        known = "\n".join(f"- {l.text}" for l in lessons) or "(none)"
        user = (f"TASK:\n{task.prompt}\n\nLESSONS THE AGENT ALREADY HAD:\n{known}\n\n"
                f"FAILING CODE:\n```python\n{failed_code}```\n\nTEST FAILURES:\n{feedback}\n")
        if fixed_code:
            user += f"\nCODE THAT EVENTUALLY PASSED:\n```python\n{fixed_code}```\n"
        hint = classifier.classify(feedback)
        user += f"\nFAILURE CLASSIFIER: category={hint['category']} ({hint['explanation']}); tags={hint['tags']}\n"
        out = llm.chat(self.cfg.distiller, [{"role": "system", "content": DISTILL_SYSTEM},
                                            {"role": "user", "content": user}],
                       temperature=0.2, max_tokens=600)
        rows = extract_json(out).get("lessons", [])
        res = []
        for r in rows[:2]:
            if not isinstance(r, dict) or not str(r.get("text", "")).strip():
                continue
            scope = "global" if str(r.get("scope", "")).lower() == "global" else "specific"
            tags = [str(t)[:24] for t in (r.get("tags") or [])][:4]
            res.append({"text": str(r["text"]).strip()[:400], "scope": scope, "tags": tags})
        return res


# ---------------------------------------------------------------------------
# Offline mock agent
# ---------------------------------------------------------------------------

# What a lesson must mention for the mock model to "understand" a convention.
KEYWORDS = {
    "result_envelope": ["envelope"],
    "error_codes": ["upper_snake"],
    "money_cents": ["integer cents"],
    "iso_dates": ["yyyy-mm-dd"],
    "sorted_ids": ["ascending"],
    "id_format": ["zero-padded"],
    "split_remainder": ["remainder"],
    "name_normalize": ["title case"],
}

CANON = {
    "result_envelope": ("global", ["return", "envelope", "errors"], [
        "Functions in this codebase never raise. Wrap results in an envelope: {'ok': True, 'value': v} on "
        "success and {'ok': False, 'error': CODE} on failure.",
        "Return a result envelope instead of raising: success is {'ok': True, 'value': ...}; failures are "
        "{'ok': False, 'error': ...}.",
    ]),
    "error_codes": ("global", ["error", "codes"], [
        "Error codes are UPPER_SNAKE constants such as 'INVALID_AMOUNT', never human-readable sentences.",
        "Use UPPER_SNAKE error codes (e.g. 'NEGATIVE_TERMS') in failure results.",
    ]),
    "money_cents": ("specific", ["money", "amount", "cents"], [
        "Money amounts are integer cents ($12.50 -> 1250), for inputs and outputs. Parse amount strings with "
        "Decimal and return int cents, never float dollars.",
        "Every amount, total, balance or discount is integer cents; only divide by 100 when formatting an "
        "amount for display.",
    ]),
    "iso_dates": ("specific", ["dates", "iso"], [
        "Dates are ISO-8601 strings 'YYYY-MM-DD' in and out; never return date objects or US-style "
        "'MM/DD/YYYY' dates.",
        "Format every returned date (start, end, due date) with date.isoformat(), i.e. YYYY-MM-DD.",
    ]),
    "sorted_ids": ("specific", ["ids", "sorting"], [
        "Any list of ids (accounts, invoices, customers) is returned sorted ascending by id, regardless of "
        "input or ranking order.",
        "Return the ids of matching records in ascending id order, not in the order they were found or ranked.",
    ]),
    "id_format": ("specific", ["ids", "sequence"], [
        "New ids are a lowercase prefix plus a 6-digit zero-padded sequence number: 'inv_000042', "
        "'cus_000007', 'pay_000003'.",
        "Build the id for a sequence number as f'{prefix}_{seq:06d}' (zero-padded, lowercase prefix such as "
        "inv, cus, pay).",
    ]),
    "split_remainder": ("specific", ["split", "shares"], [
        "When splitting a total evenly into shares (bills, tips), give the leftover remainder to the first "
        "shares, one each: 1000 split 3 ways -> [334, 333, 333].",
        "Even splits put the remainder on the earliest shares in the list, never on the last share.",
    ]),
    "name_normalize": ("specific", ["names", "customers"], [
        "Customer names typed into forms are normalized: collapse whitespace and use Title Case "
        "('  ada   LOVELACE ' -> 'Ada Lovelace').",
        "Normalize a customer name with title case and single spaces: ' '.join(w.capitalize() for w in "
        "name.split()).",
    ]),
}

# Plausible-sounding but useless lessons. Real distillers produce these too;
# ablation is what gets rid of them.
NOISE = [
    ("specific", "Add docstrings to every function describing the invoice, account or customer fields it uses."),
    ("specific", "Prefer list comprehensions over loops when building lists of ids, amounts or names."),
    ("specific", "Validate that dict inputs contain the expected keys before reading amount, date or id fields."),
    ("specific", "Use type hints on parameters such as amount, date, seq and name."),
    ("specific", "Keep functions short: move amount parsing and date formatting into small helpers."),
    ("global", "Always handle edge cases carefully in billing code."),
    ("global", "Write defensive code: guard against None inputs in every function."),
    ("global", "Stay consistent with the existing codebase style."),
]


# Misreadings: a distiller looks at a failure and writes the WRONG rule. When a
# wrong lesson and the right one are both in the prompt, the model has to pick;
# with only the wrong one, it follows it. Ablation catches these as "hurt".
WRONG = {
    "result_envelope": ("global", "raise valueerror", "Raise ValueError on invalid input so callers notice errors immediately."),
    "error_codes": ("global", "friendly sentences", "Error messages should be friendly sentences a customer can read."),
    "money_cents": ("specific", "rounded to 2 decimals", "Amounts are dollars: return them as floats rounded to 2 decimals, e.g. 12.5."),
    "iso_dates": ("specific", "the way users read them", "Return dates the way users read them on invoices, like 02/14/2026."),
    "sorted_ids": ("specific", "callers rely on input order", "Keep returned ids in the order found; callers rely on input order."),
    "id_format": ("specific", "uppercase with a dash", "Ids are uppercase with a dash, like 'INV-42' or 'CUS-7'."),
    "split_remainder": ("specific", "stays tidy", "Put any leftover from a split onto the last share so the list stays tidy."),
    "name_normalize": ("specific", "exactly as typed", "Keep customer names exactly as typed apart from trimming the ends."),
}


def misled(lessons):
    text = " ".join(l.text.lower() for l in lessons)
    return {c for c, (_s, kw, _t) in WRONG.items() if kw in text}


def _rng(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def understood(lessons):
    text = " ".join(l.text.lower() for l in lessons)
    return {c for c, kws in KEYWORDS.items() if any(k in text for k in kws)}


class MockAgent:
    """Simulates an LLM: applies a convention if a lesson in its prompt explains it
    (or the test feedback just revealed it); otherwise it writes the 'obvious' code."""

    P_APPLY = 0.99     # follows a lesson it was given
    P_LUCKY = 0.05     # stumbles on an unknown convention by chance
    P_CONFLICT = 0.5   # right and wrong lesson both present: coin flip
    P_NOISE = 0.45     # distillation also emits a useless lesson
    P_WRONG = 0.15     # distillation misreads a failure into a wrong rule

    def __init__(self, cfg):
        self.cfg = cfg
        self._revealed = {}  # feedback text -> conventions it revealed

    def solve(self, task, lessons, history, seed_key=""):
        known, wrong = understood(lessons), misled(lessons)
        fresh = set()
        for _code, feedback in history:
            fresh |= self._revealed.get(feedback, set())
        rng = _rng(self.cfg.seed, seed_key, task.id, len(history))
        K = set()
        for c in task.convs:
            r = rng.random()
            if c in fresh:                      # the test failure is right in front of it
                p = self.P_APPLY
            elif c in known and c in wrong:
                p = self.P_CONFLICT
            elif c in known:
                p = self.P_APPLY
            elif c in wrong:
                p = 0.0
            else:
                p = self.P_LUCKY
            if r < p:
                K.add(c)
        return benchmark.mock_solution(task, K)

    # The mock "reads" a failure through its hidden tags (a real model reads the messages).
    def note_feedback(self, result):
        self._revealed[result.feedback()] = result.conv_tags()

    def distill(self, task, lessons, failed_code, feedback, fixed_code=None, seed_key=""):
        rng = _rng(self.cfg.seed, "distill", seed_key, task.id)
        known = understood(lessons)
        out = []
        for conv in sorted(self._revealed.get(feedback, set()) - known):
            if rng.random() < self.P_WRONG:
                scope, _kw, text = WRONG[conv]
                out.append({"text": text, "scope": scope, "tags": []})
                continue
            scope, tags, texts = CANON[conv]
            out.append({"text": rng.choice(texts), "scope": scope, "tags": tags})
        if rng.random() < self.P_NOISE:
            scope, text = rng.choice(NOISE)
            out.append({"text": text, "scope": scope, "tags": []})
        return out[:2] if len(out) > 2 and rng.random() < 0.5 else out


class CompositeAgent:
    """Mixed backends: solver and distiller chosen independently.

    E.g. a real solver with the offline distiller, or the mock solver with a
    real distiller. Feedback notes are forwarded to whichever side keeps them.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._solver = MockAgent(cfg) if cfg.solver.provider == "mock" else LLMAgent(cfg)
        self._distiller = MockAgent(cfg) if cfg.distiller.provider == "mock" else LLMAgent(cfg)

    def solve(self, task, lessons, history, seed_key=""):
        return self._solver.solve(task, lessons, history, seed_key=seed_key)

    def distill(self, task, lessons, failed_code, feedback, fixed_code=None, seed_key=""):
        return self._distiller.distill(task, lessons, failed_code, feedback, fixed_code, seed_key=seed_key)

    def note_feedback(self, result):
        for side in (self._solver, self._distiller):
            if hasattr(side, "note_feedback"):
                side.note_feedback(result)


def make_agent(cfg):
    if cfg.solver.provider == "mock" and cfg.distiller.provider == "mock":
        return MockAgent(cfg)
    if cfg.solver.provider != "mock" and cfg.distiller.provider != "mock":
        return LLMAgent(cfg)
    return CompositeAgent(cfg)
