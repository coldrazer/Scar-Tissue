"""The Acme Ledger benchmark.

Acme Ledger is a fictional billing codebase with eight *unwritten* house
conventions. Task prompts never state them — like a real codebase, the only
way to discover them is to ship code, fail review (the tests), and remember
why. That is exactly what Scar Tissue is built to do.

  result_envelope   functions never raise; they return {"ok": True, "value": v}
                    or {"ok": False, "error": CODE}
  error_codes       error codes are UPPER_SNAKE_CASE strings
  money_cents       money is always integer cents, never float dollars
  iso_dates         dates are ISO-8601 strings, YYYY-MM-DD
  sorted_ids        lists of ids are returned sorted ascending
  id_format         ids are <lowercase prefix>_<6-digit zero-padded number>
  split_remainder   when splitting cents, leftover cents go to the first shares
  name_normalize    names: whitespace collapsed, Title Case

Train tasks are what the agent learns from. Eval tasks are held out: the agent
never gets feedback on them, so the eval pass rate measures transfer across
functions that share the same house conventions (not memorized solutions).
"""

from dataclasses import dataclass, field
from typing import Callable

CONVENTIONS = {
    "result_envelope": "Functions never raise: return {'ok': True, 'value': v} or {'ok': False, 'error': CODE}",
    "error_codes": "Error codes are UPPER_SNAKE_CASE strings",
    "money_cents": "Money is integer cents, never float dollars",
    "iso_dates": "Dates are ISO-8601 strings (YYYY-MM-DD)",
    "sorted_ids": "Lists of ids are returned sorted ascending",
    "id_format": "Ids are <prefix>_<6-digit zero-padded number>",
    "split_remainder": "Leftover cents from a split go to the first shares",
    "name_normalize": "Names are whitespace-collapsed and Title Case",
}


@dataclass
class Task:
    id: str
    split: str  # "train" | "eval"
    prompt: str
    tests: str
    convs: list = field(default_factory=list)
    mock: Callable = None  # mock(K: set[str]) -> solution source (offline mode only)


# ---------------------------------------------------------------------------
# Offline mock solutions: what a model "would write" given the conventions K
# it has been told about. Only used by --mock mode and `selfcheck`.
# ---------------------------------------------------------------------------

def prelude(K):
    env = "result_envelope" in K
    codes = "error_codes" in K
    s = "def _ok(v):\n"
    s += "    return {'ok': True, 'value': v}\n\n" if env else "    return v\n\n"
    s += "def _fail(code, msg):\n"
    if env and codes:
        s += "    return {'ok': False, 'error': code}\n\n"
    elif env:
        s += "    return {'ok': False, 'error': msg}\n\n"
    else:
        s += "    raise ValueError(msg)\n\n"
    return s


def _m_parse_amount(K):
    ret = "int((d * 100).to_integral_value())" if "money_cents" in K else "float(d)"
    return f"""
from decimal import Decimal, InvalidOperation

def parse_amount(text):
    t = text.strip().lstrip('$')
    try:
        d = Decimal(t)
    except InvalidOperation:
        return _fail('INVALID_AMOUNT', 'invalid amount')
    return _ok({ret})
"""


def _m_format_amount(K):
    if "money_cents" in K:
        return """
def format_amount(amount):
    sign = '-' if amount < 0 else ''
    a = abs(amount)
    return _ok(f"{sign}${a // 100}.{a % 100:02d}")
"""
    return """
def format_amount(amount):
    sign = '-' if amount < 0 else ''
    return _ok(f"{sign}${abs(amount):.2f}")
"""


def _m_due(K):
    fmt = "'%Y-%m-%d'" if "iso_dates" in K else "'%m/%d/%Y'"
    return f"""
from datetime import date, timedelta

def invoice_due_date(issued, days):
    if days < 0:
        return _fail('NEGATIVE_TERMS', 'negative terms')
    d = date.fromisoformat(issued) + timedelta(days=days)
    return _ok(d.strftime({fmt}))
"""


def _m_active(K):
    s = "sorted(ids)" if "sorted_ids" in K else "ids"
    return f"""
def active_accounts(accounts):
    ids = [a['id'] for a in accounts if a['active']]
    return _ok({s})
"""


def _m_id(name, prefix):
    def gen(K):
        fmt = f"f'{prefix}_{{seq:06d}}'" if "id_format" in K else f"f'{prefix.upper()}-{{seq}}'"
        return f"""
def {name}(seq):
    if seq <= 0:
        return _fail('INVALID_SEQUENCE', 'sequence must be positive')
    return _ok({fmt})
"""
    return gen


def _m_split(name, a, b):
    def gen(K):
        if "split_remainder" in K:
            body = f"base, rem = divmod({a}, {b})\n    return _ok([base + (1 if i < rem else 0) for i in range({b})])"
        else:
            body = f"base, rem = divmod({a}, {b})\n    shares = [base] * {b}\n    shares[-1] += rem\n    return _ok(shares)"
        return f"""
def {name}({a}, {b}):
    if {b} <= 0:
        return _fail('INVALID_PARTY_COUNT', 'need at least one person')
    {body}
"""
    return gen


def _norm(K):
    return "' '.join(w.capitalize() for w in s.split())" if "name_normalize" in K else "s.strip()"


def _m_normalize(K):
    return f"""
def normalize_name(name):
    s = name
    return _ok({_norm(K)})
"""


def _m_discount(K):
    ret = "round(amount * (100 - percent) / 100)" if "money_cents" in K else "amount * (100 - percent) / 100"
    return f"""
def apply_discount(amount, percent):
    if percent < 0 or percent > 100:
        return _fail('INVALID_PERCENT', 'percent out of range')
    return _ok({ret})
"""


def _m_overdue(K):
    s = "sorted(ids)" if "sorted_ids" in K else "ids"
    return f"""
def overdue_invoices(invoices, today):
    ids = [i['id'] for i in invoices if i['due'] < today]
    return _ok({s})
"""


def _m_month(K):
    fmt = "'%Y-%m-%d'" if "iso_dates" in K else "'%m/%d/%Y'"
    return f"""
import calendar
from datetime import date

def month_bounds(year, month):
    if not 1 <= month <= 12:
        return _fail('INVALID_MONTH', 'invalid month')
    last = calendar.monthrange(year, month)[1]
    return _ok({{'start': date(year, month, 1).strftime({fmt}), 'end': date(year, month, last).strftime({fmt})}})
"""


def _m_customer(K):
    idf = "f'cus_{seq:06d}'" if "id_format" in K else "f'CUS-{seq}'"
    return f"""
def new_customer(name, seq):
    s = name
    return _ok({{'id': {idf}, 'name': {_norm(K)}}})
"""


def _m_ledger(K):
    conv = "int((Decimal(e['amount'].lstrip('$')) * 100).to_integral_value())" if "money_cents" in K \
        else "float(e['amount'].lstrip('$'))"
    zero = "0" if "money_cents" in K else "0.0"
    return f"""
from decimal import Decimal

def ledger_balance(entries):
    total = {zero}
    for e in entries:
        amt = {conv}
        if e['type'] == 'credit':
            total += amt
        elif e['type'] == 'debit':
            total -= amt
        else:
            return _fail('UNKNOWN_ENTRY_TYPE', 'unknown entry type')
    return _ok(total)
"""


def _m_total(K):
    conv = "int((d * 100).to_integral_value())" if "money_cents" in K else "float(d)"
    zero = "0" if "money_cents" in K else "0.0"
    return f"""
from decimal import Decimal, InvalidOperation

def total_of(amounts):
    total = {zero}
    for a in amounts:
        try:
            d = Decimal(a.strip().lstrip('$'))
        except InvalidOperation:
            return _fail('INVALID_AMOUNT', 'invalid amount')
        total += {conv}
    return _ok(total)
"""


def _m_period(K):
    fmt = "'%Y-%m-%d'" if "iso_dates" in K else "'%m/%d/%Y'"
    return f"""
from datetime import date, timedelta

def statement_period(start):
    s = date.fromisoformat(start)
    e = s + timedelta(days=29)
    return _ok({{'start': s.strftime({fmt}), 'end': e.strftime({fmt})}})
"""


def _m_top(K):
    s = "sorted(top)" if "sorted_ids" in K else "top"
    return f"""
def top_spenders(customers, k):
    ranked = sorted(customers, key=lambda c: -c['spend'])
    top = [c['id'] for c in ranked[:k]]
    return _ok({s})
"""


def _m_add_tax(K):
    ret = "round(amount * (100 + percent) / 100)" if "money_cents" in K else "amount * (100 + percent) / 100"
    return f"""
def add_tax(amount, percent):
    if percent < 0:
        return _fail('INVALID_PERCENT', 'percent out of range')
    return _ok({ret})
"""


def _m_display_name(K):
    return f"""
def display_name(name):
    s = name
    return _ok({_norm(K)})
"""


def _m_customer_ids(K):
    srt = "sorted(ids)" if "sorted_ids" in K else "ids"
    return f"""
def customer_ids(customers):
    ids = [c['id'] for c in customers]
    return _ok({srt})
"""


def _m_cycle(K):
    fmt = "'%Y-%m-%d'" if "iso_dates" in K else "'%m/%d/%Y'"
    return f"""
from datetime import date, timedelta

def billing_cycle(start):
    s = date.fromisoformat(start)
    e = s + timedelta(days=29)
    return _ok({{'start': s.strftime({fmt}), 'end': e.strftime({fmt})}})
"""


def _m_unique(K):
    return f"""
def unique_names(names):
    out = []
    for s in names:
        n = {_norm(K)}
        if n not in out:
            out.append(n)
    return _ok(out)
"""


def mock_solution(task, K):
    return prelude(K) + task.mock(K).lstrip("\n")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    # ------------------------------ train ---------------------------------
    Task("parse_amount", "train",
         "Write `parse_amount(text)` that parses a user-entered amount string such as '$12.50', '7' "
         "or '$0.05' and returns the amount. Invalid input such as 'abc' should be reported as an error.",
         """
check("parse_amount('$12.50')", lambda: _fn("parse_amount")("$12.50"), 1250, ["money_cents"])
check("parse_amount('7')", lambda: _fn("parse_amount")("7"), 700, ["money_cents"])
check("parse_amount('$0.05')", lambda: _fn("parse_amount")("$0.05"), 5, ["money_cents"])
check_error("parse_amount('abc')", lambda: _fn("parse_amount")("abc"))
""", ["money_cents", "result_envelope", "error_codes"], _m_parse_amount),

    Task("format_amount", "train",
         "Write `format_amount(amount)` that formats an amount for display on an invoice, like '$12.50'. "
         "Negative amounts are shown like '-$3.00'.",
         """
check("format_amount(1250)", lambda: _fn("format_amount")(1250), "$12.50", ["money_cents"])
check("format_amount(5)", lambda: _fn("format_amount")(5), "$0.05", ["money_cents"])
check("format_amount(-300)", lambda: _fn("format_amount")(-300), "-$3.00", ["money_cents"])
""", ["money_cents", "result_envelope"], _m_format_amount),

    Task("invoice_due_date", "train",
         "Write `invoice_due_date(issued, days)` that returns the due date of an invoice issued on the date "
         "`issued` (e.g. '2026-01-15') with payment terms of `days` days. Negative terms are an error.",
         """
check("invoice_due_date('2026-01-15', 30)", lambda: _fn("invoice_due_date")("2026-01-15", 30), "2026-02-14", ["iso_dates"])
check("invoice_due_date('2026-12-20', 15)", lambda: _fn("invoice_due_date")("2026-12-20", 15), "2027-01-04", ["iso_dates"])
check_error("invoice_due_date('2026-01-15', -1)", lambda: _fn("invoice_due_date")("2026-01-15", -1))
""", ["iso_dates", "result_envelope", "error_codes"], _m_due),

    Task("active_accounts", "train",
         "Write `active_accounts(accounts)` where `accounts` is a list of dicts with keys 'id' and 'active'. "
         "Return the ids of the active accounts.",
         """
A = [{"id": "acct_000009", "active": True}, {"id": "acct_000002", "active": False},
     {"id": "acct_000005", "active": True}, {"id": "acct_000001", "active": True}]
check("active_accounts(<4 accounts>)", lambda: _fn("active_accounts")(A), ["acct_000001", "acct_000005", "acct_000009"], ["sorted_ids"])
check("active_accounts([])", lambda: _fn("active_accounts")([]), [], ["sorted_ids"])
""", ["sorted_ids", "result_envelope"], _m_active),

    Task("new_invoice_id", "train",
         "Write `new_invoice_id(seq)` that builds the id for the invoice with sequence number `seq` "
         "(a positive integer). Non-positive sequence numbers are an error.",
         """
check("new_invoice_id(42)", lambda: _fn("new_invoice_id")(42), "inv_000042", ["id_format"])
check("new_invoice_id(7)", lambda: _fn("new_invoice_id")(7), "inv_000007", ["id_format"])
check_error("new_invoice_id(0)", lambda: _fn("new_invoice_id")(0))
""", ["id_format", "result_envelope", "error_codes"], _m_id("new_invoice_id", "inv")),

    Task("split_bill", "train",
         "Write `split_bill(total, people)` that splits a bill `total` evenly between `people` people and "
         "returns each person's share in a list. The shares must add up to the total. "
         "Zero or negative people is an error.",
         """
check("split_bill(1000, 3)", lambda: _fn("split_bill")(1000, 3), [334, 333, 333], ["split_remainder"])
check("split_bill(1001, 4)", lambda: _fn("split_bill")(1001, 4), [251, 250, 250, 250], ["split_remainder"])
check("split_bill(900, 3)", lambda: _fn("split_bill")(900, 3), [300, 300, 300], ["split_remainder"])
check_error("split_bill(500, 0)", lambda: _fn("split_bill")(500, 0))
""", ["split_remainder", "result_envelope", "error_codes"], _m_split("split_bill", "total", "people")),

    Task("normalize_name", "train",
         "Write `normalize_name(name)` that cleans up a customer name typed into a form, e.g. '  ada   LOVELACE '.",
         """
check("normalize_name('  ada   LOVELACE ')", lambda: _fn("normalize_name")("  ada   LOVELACE "), "Ada Lovelace", ["name_normalize"])
check("normalize_name('grace hopper')", lambda: _fn("normalize_name")("grace hopper"), "Grace Hopper", ["name_normalize"])
check("normalize_name('  ALAN TURING')", lambda: _fn("normalize_name")("  ALAN TURING"), "Alan Turing", ["name_normalize"])
""", ["name_normalize", "result_envelope"], _m_normalize),

    Task("apply_discount", "train",
         "Write `apply_discount(amount, percent)` that applies a percentage discount to `amount` and returns "
         "the discounted amount. A percent outside 0-100 is an error.",
         """
check("apply_discount(1999, 10)", lambda: _fn("apply_discount")(1999, 10), 1799, ["money_cents"])
check("apply_discount(1000, 25)", lambda: _fn("apply_discount")(1000, 25), 750, ["money_cents"])
check("apply_discount(500, 0)", lambda: _fn("apply_discount")(500, 0), 500, ["money_cents"])
check_error("apply_discount(100, 150)", lambda: _fn("apply_discount")(100, 150))
""", ["money_cents", "result_envelope", "error_codes"], _m_discount),

    Task("overdue_invoices", "train",
         "Write `overdue_invoices(invoices, today)` where each invoice is a dict with 'id' and 'due' "
         "(a date string in the same format as `today`). Return the ids of the invoices that are past due.",
         """
I = [{"id": "inv_000012", "due": "2026-03-01"}, {"id": "inv_000003", "due": "2026-01-10"},
     {"id": "inv_000007", "due": "2026-05-01"}, {"id": "inv_000001", "due": "2026-02-28"}]
check("overdue_invoices(<4 invoices>, '2026-03-01')", lambda: _fn("overdue_invoices")(I, "2026-03-01"), ["inv_000001", "inv_000003"], ["sorted_ids"])
check("overdue_invoices(<4 invoices>, '2026-01-01')", lambda: _fn("overdue_invoices")(I, "2026-01-01"), [], ["sorted_ids"])
""", ["sorted_ids", "result_envelope"], _m_overdue),

    Task("month_bounds", "train",
         "Write `month_bounds(year, month)` that returns a dict with the 'start' and 'end' dates of the given "
         "month. An invalid month is an error.",
         """
check("month_bounds(2026, 2)", lambda: _fn("month_bounds")(2026, 2), {"start": "2026-02-01", "end": "2026-02-28"}, ["iso_dates"])
check("month_bounds(2028, 2)", lambda: _fn("month_bounds")(2028, 2), {"start": "2028-02-01", "end": "2028-02-29"}, ["iso_dates"])
check("month_bounds(2026, 12)", lambda: _fn("month_bounds")(2026, 12), {"start": "2026-12-01", "end": "2026-12-31"}, ["iso_dates"])
check_error("month_bounds(2026, 13)", lambda: _fn("month_bounds")(2026, 13))
""", ["iso_dates", "result_envelope", "error_codes"], _m_month),

    Task("new_customer", "train",
         "Write `new_customer(name, seq)` that creates a customer record (a dict with keys 'id' and 'name') "
         "from a name typed into a form and the customer's sequence number.",
         """
check("new_customer('  ada   LOVELACE ', 7)", lambda: _fn("new_customer")("  ada   LOVELACE ", 7), {"id": "cus_000007", "name": "Ada Lovelace"}, ["id_format", "name_normalize"])
check("new_customer('grace hopper', 12)", lambda: _fn("new_customer")("grace hopper", 12), {"id": "cus_000012", "name": "Grace Hopper"}, ["id_format", "name_normalize"])
""", ["id_format", "name_normalize", "result_envelope"], _m_customer),

    Task("ledger_balance", "train",
         "Write `ledger_balance(entries)` where each entry is a dict with 'amount' (an amount string like "
         "'$12.50') and 'type' ('credit' or 'debit'). Return the balance: credits minus debits. "
         "Any other entry type is an error.",
         """
E = [{"amount": "$12.50", "type": "credit"}, {"amount": "$2.25", "type": "debit"}, {"amount": "$0.10", "type": "credit"}]
check("ledger_balance(<3 entries>)", lambda: _fn("ledger_balance")(E), 1035, ["money_cents"])
check("ledger_balance([])", lambda: _fn("ledger_balance")([]), 0, ["money_cents"])
check_error("ledger_balance([{'amount': '$1', 'type': 'refund'}])", lambda: _fn("ledger_balance")([{"amount": "$1", "type": "refund"}]))
""", ["money_cents", "result_envelope", "error_codes"], _m_ledger),

    # ------------------------------ eval ----------------------------------
    Task("total_of", "eval",
         "Write `total_of(amounts)` that adds up a list of user-entered amount strings like '$3.10' and "
         "returns the total. Any invalid amount is an error.",
         """
check("total_of(['$3.10', '$4.20', '5'])", lambda: _fn("total_of")(["$3.10", "$4.20", "5"]), 1230, ["money_cents"])
check("total_of([])", lambda: _fn("total_of")([]), 0, ["money_cents"])
check_error("total_of(['$1', 'x'])", lambda: _fn("total_of")(["$1", "x"]))
""", ["money_cents", "result_envelope", "error_codes"], _m_total),

    Task("statement_period", "eval",
         "Write `statement_period(start)` returning a dict with the 'start' and 'end' dates of a 30-day "
         "billing statement that begins on the date `start` (e.g. '2026-01-01'); the end is the 30th day, inclusive.",
         """
check("statement_period('2026-01-01')", lambda: _fn("statement_period")("2026-01-01"), {"start": "2026-01-01", "end": "2026-01-30"}, ["iso_dates"])
check("statement_period('2026-02-15')", lambda: _fn("statement_period")("2026-02-15"), {"start": "2026-02-15", "end": "2026-03-16"}, ["iso_dates"])
""", ["iso_dates", "result_envelope"], _m_period),

    Task("top_spenders", "eval",
         "Write `top_spenders(customers, k)` where each customer is a dict with 'id' and 'spend'. "
         "Return the ids of the `k` customers with the highest spend.",
         """
C = [{"id": "cus_000004", "spend": 500}, {"id": "cus_000002", "spend": 900},
     {"id": "cus_000009", "spend": 700}, {"id": "cus_000001", "spend": 100}]
check("top_spenders(<4 customers>, 3)", lambda: _fn("top_spenders")(C, 3), ["cus_000002", "cus_000004", "cus_000009"], ["sorted_ids"])
check("top_spenders(<4 customers>, 1)", lambda: _fn("top_spenders")(C, 1), ["cus_000002"], ["sorted_ids"])
""", ["sorted_ids", "result_envelope"], _m_top),

    Task("new_payment_id", "eval",
         "Write `new_payment_id(seq)` that builds the id for the payment with sequence number `seq` "
         "(a positive integer). Non-positive sequence numbers are an error.",
         """
check("new_payment_id(3)", lambda: _fn("new_payment_id")(3), "pay_000003", ["id_format"])
check("new_payment_id(120)", lambda: _fn("new_payment_id")(120), "pay_000120", ["id_format"])
check_error("new_payment_id(-1)", lambda: _fn("new_payment_id")(-1))
""", ["id_format", "result_envelope", "error_codes"], _m_id("new_payment_id", "pay")),

    Task("split_tip", "eval",
         "Write `split_tip(tip, staff)` that divides a `tip` evenly among `staff` staff members, returning "
         "each share in a list. The shares must add up to the tip. No staff is an error.",
         """
check("split_tip(700, 3)", lambda: _fn("split_tip")(700, 3), [234, 233, 233], ["split_remainder"])
check("split_tip(5, 2)", lambda: _fn("split_tip")(5, 2), [3, 2], ["split_remainder"])
check("split_tip(1000, 4)", lambda: _fn("split_tip")(1000, 4), [250, 250, 250, 250], ["split_remainder"])
check_error("split_tip(100, 0)", lambda: _fn("split_tip")(100, 0))
""", ["split_remainder", "result_envelope", "error_codes"], _m_split("split_tip", "tip", "staff")),

    Task("add_tax", "eval",
         "Write `add_tax(amount, percent)` that returns `amount` with a tax of `percent` percent added. "
         "A negative percent is an error.",
         """
check("add_tax(1000, 10)", lambda: _fn("add_tax")(1000, 10), 1100, ["money_cents"])
check("add_tax(1999, 25)", lambda: _fn("add_tax")(1999, 25), 2499, ["money_cents"])
check_error("add_tax(100, -5)", lambda: _fn("add_tax")(100, -5))
""", ["money_cents", "result_envelope", "error_codes"], _m_add_tax),

    Task("new_order_id", "eval",
         "Write `new_order_id(seq)` that builds the id for the order with sequence number `seq` "
         "(a positive integer). Non-positive sequence numbers are an error.",
         """
check("new_order_id(9)", lambda: _fn("new_order_id")(9), "ord_000009", ["id_format"])
check("new_order_id(123)", lambda: _fn("new_order_id")(123), "ord_000123", ["id_format"])
check_error("new_order_id(0)", lambda: _fn("new_order_id")(0))
""", ["id_format", "result_envelope", "error_codes"], _m_id("new_order_id", "ord")),

    Task("split_rent", "eval",
         "Write `split_rent(rent, people)` that splits a rent payment evenly between `people` people and "
         "returns each person's share in a list. The shares must add up to the rent. "
         "Zero or negative people is an error.",
         """
check("split_rent(1000, 3)", lambda: _fn("split_rent")(1000, 3), [334, 333, 333], ["split_remainder"])
check("split_rent(501, 2)", lambda: _fn("split_rent")(501, 2), [251, 250], ["split_remainder"])
check_error("split_rent(500, 0)", lambda: _fn("split_rent")(500, 0))
""", ["split_remainder", "result_envelope", "error_codes"], _m_split("split_rent", "rent", "people")),

    Task("display_name", "eval",
         "Write `display_name(name)` that cleans up a customer name typed into a form for display, "
         "e.g. '  ada   LOVELACE '.",
         """
check("display_name('  ada   LOVELACE ')", lambda: _fn("display_name")("  ada   LOVELACE "), "Ada Lovelace", ["name_normalize"])
check("display_name('GRACE hopper')", lambda: _fn("display_name")("GRACE hopper"), "Grace Hopper", ["name_normalize"])
""", ["name_normalize", "result_envelope"], _m_display_name),

    Task("customer_ids", "eval",
         "Write `customer_ids(customers)` where `customers` is a list of dicts with key 'id'. "
         "Return the ids of the customers.",
         """
C = [{"id": "cus_000008"}, {"id": "cus_000002"}, {"id": "cus_000005"}]
check("customer_ids(<3 customers>)", lambda: _fn("customer_ids")(C), ["cus_000002", "cus_000005", "cus_000008"], ["sorted_ids"])
check("customer_ids([])", lambda: _fn("customer_ids")([]), [], ["sorted_ids"])
""", ["sorted_ids", "result_envelope"], _m_customer_ids),

    Task("billing_cycle", "eval",
         "Write `billing_cycle(start)` returning a dict with the 'start' and 'end' dates of a 30-day "
         "billing cycle that begins on the date `start` (e.g. '2026-01-01'); the end is the 30th day, inclusive.",
         """
check("billing_cycle('2026-01-01')", lambda: _fn("billing_cycle")("2026-01-01"), {"start": "2026-01-01", "end": "2026-01-30"}, ["iso_dates"])
check("billing_cycle('2026-06-10')", lambda: _fn("billing_cycle")("2026-06-10"), {"start": "2026-06-10", "end": "2026-07-09"}, ["iso_dates"])
""", ["iso_dates", "result_envelope"], _m_cycle),

    Task("unique_names", "eval",
         "Write `unique_names(names)` that takes customer names typed into forms and returns the list of "
         "distinct customers, in order of first appearance.",
         """
check("unique_names(['ada lovelace', '  Ada  Lovelace ', 'GRACE HOPPER'])", lambda: _fn("unique_names")(["ada lovelace", "  Ada  Lovelace ", "GRACE HOPPER"]), ["Ada Lovelace", "Grace Hopper"], ["name_normalize"])
check("unique_names([])", lambda: _fn("unique_names")([]), [], ["name_normalize"])
""", ["name_normalize", "result_envelope"], _m_unique),
]


def tasks(split=None):
    return [t for t in TASKS if split is None or t.split == split]


def get(task_id):
    return next(t for t in TASKS if t.id == task_id)
