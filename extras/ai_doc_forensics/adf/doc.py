"""Document content: genuine/AI-generated construction, AI edits, element texts and OCR extraction noise."""
import copy
import numpy as np
from . import names


def money(v):
    return f"{v:,.2f}"


def _r(x):
    return float(np.round(x, 2))


# ----------------------------------------------------------------------------------------------- construction
def _amount(rng, uniform_digits):
    if uniform_digits:                       # not Benford-like: leading digits roughly uniform
        return _r(rng.uniform(30, 9000))
    return _r(np.exp(rng.uniform(np.log(20), np.log(9000))))   # log-uniform => Benford-like leading digits


def _statement(rng, cfg, generated):
    uni = generated and rng.random() < cfg.p_gen_benford_off
    opening = _r(np.clip(np.exp(rng.normal(8.0, 1.0)), 500, 80000))
    n = int(rng.integers(10, 19))
    month = int(rng.integers(1, 13))
    days = np.sort(rng.integers(1, 28, n))
    rows, bal = [], opening
    for i in range(n):
        credit = debit = 0.0
        if rng.random() < 0.22:
            credit = _amount(rng, uni) * (3 if rng.random() < 0.3 else 1)
            credit = _r(credit)
        else:
            debit = _amount(rng, uni)
        bal = _r(bal + credit - debit)
        rows.append(dict(date=f"{int(days[i]):02d}/{month:02d}", desc=str(rng.choice(names.MERCHANTS)),
                         debit=debit, credit=credit, balance=bal))
    summ = dict(opening=opening, total_debits=_r(sum(r["debit"] for r in rows)),
                total_credits=_r(sum(r["credit"] for r in rows)), closing=bal)
    return dict(type="statement", holder=f"{rng.choice(names.FIRST)} {rng.choice(names.LAST)}",
                acct=f"{int(rng.integers(10**9, 10**10))}", period=f"01/{month:02d} - 28/{month:02d}",
                rows=rows, summary=summ)


def _paye(g):
    t = 0.18 * min(g, 20000) + 0.26 * max(0, min(g, 35000) - 20000) + 0.31 * max(0, g - 35000) - 2500
    return _r(max(0.0, t))


def _uif(g):
    return _r(min(0.01 * g, 177.12))


def _payslip(rng, cfg, generated):
    uni = generated and rng.random() < cfg.p_gen_benford_off
    basic = _r(np.clip(np.exp(rng.normal(10.0, 0.4)), 6000, 90000))
    earn = [("Basic salary", basic)]
    if rng.random() < 0.6:
        earn.append(("Transport allowance", _amount(rng, uni) if uni else _r(rng.uniform(300, 2500))))
    if rng.random() < 0.3:
        earn.append(("Overtime", _amount(rng, uni)))
    gross = _r(sum(v for _, v in earn))
    paye, uif = _paye(gross), _uif(gross)
    if generated and rng.random() < cfg.p_gen_benford_off:      # fabricated deductions do not follow the formulas
        paye = _r(gross * rng.uniform(0.10, 0.34)); uif = _r(rng.uniform(40, 260))
    ded = [("PAYE", paye), ("UIF", uif), ("Pension fund", _r(0.075 * basic)), ("Medical aid", _r(rng.uniform(800, 2500)))]
    total_ded = _r(sum(v for _, v in ded))
    return dict(type="payslip", employee=f"{rng.choice(names.FIRST)} {rng.choice(names.LAST)}",
                empno=f"E{int(rng.integers(10000, 99999))}", month=f"2026-{int(rng.integers(1, 13)):02d}",
                earnings=[list(e) for e in earn], deductions=[list(d) for d in ded],
                gross=gross, total_ded=total_ded, net=_r(gross - total_ded))


def make_document(rng, cfg, doc_type, generated=False):
    d = _statement(rng, cfg, generated) if doc_type == "statement" else _payslip(rng, cfg, generated)
    return d


def corrupt_arithmetic(rng, d, mild=False):
    """Break the internal arithmetic. mild=True gives a small rounding-size flaw (real documents have these too)."""
    d = copy.deepcopy(d)
    if d["type"] == "statement":
        k = int(rng.integers(0, len(d["rows"])))
        if mild:
            d["rows"][k]["balance"] = _r(d["rows"][k]["balance"] + rng.choice([-1, 1]) * rng.uniform(0.01, 0.10))
            return d
        mode = rng.choice(["balances", "totals", "closing"])
        if mode == "balances":
            for kk in rng.choice(len(d["rows"]), size=int(rng.integers(1, 4)), replace=False):
                d["rows"][kk]["balance"] = _r(d["rows"][kk]["balance"] + rng.uniform(-900, 900))
        elif mode == "totals":
            key = rng.choice(["total_debits", "total_credits"])
            d["summary"][key] = _r(d["summary"][key] + rng.uniform(-1500, 1500))
        else:
            d["summary"]["closing"] = _r(d["summary"]["closing"] + rng.uniform(-1500, 1500))
    else:
        if mild:
            d["net"] = _r(d["net"] + rng.choice([-1, 1]) * rng.uniform(0.01, 0.10)); return d
        mode = rng.choice(["net", "gross", "total_ded"])
        d[mode] = _r(d[mode] + rng.uniform(-1800, 1800))
    return d


# ---------------------------------------------------------------------------------------------------- edits
def _recalc_statement(d, from_row=0):
    rows = d["rows"]
    prev = d["summary"]["opening"] if from_row == 0 else rows[from_row - 1]["balance"]
    changed = []
    for i in range(from_row, len(rows)):
        prev = _r(prev + rows[i]["credit"] - rows[i]["debit"])
        if rows[i]["balance"] != prev:
            rows[i]["balance"] = prev; changed.append(f"r{i}_balance")
    d["summary"]["total_debits"] = _r(sum(r["debit"] for r in rows))
    d["summary"]["total_credits"] = _r(sum(r["credit"] for r in rows))
    d["summary"]["closing"] = rows[-1]["balance"]
    return changed + ["s_debits", "s_credits", "s_closing"]


def apply_edit(rng, cfg, d):
    """AI/editor tampering of a genuine document. Returns (edited_doc, changed_element_ids, manipulation_type)."""
    d = copy.deepcopy(d); recalc = rng.random() < cfg.p_edit_recalc
    if d["type"] == "statement":
        kind = rng.choice(["amount", "closing", "name", "date"], p=[0.45, 0.25, 0.15, 0.15])
        rows = d["rows"]
        if kind == "amount":
            i = int(rng.integers(0, len(rows)))
            side = "credit" if rows[i]["credit"] > 0 else "debit"
            rows[i][side] = _r(rows[i][side] * rng.uniform(1.15, 3.0) + rng.uniform(20, 400))
            ch = [f"r{i}_{side}"]
            if recalc:
                ch += _recalc_statement(d, i)
            return d, ch, "amount_edit" + ("_recalculated" if recalc else "")
        if kind == "closing":
            old = d["summary"]["closing"]; d["summary"]["closing"] = _r(old + rng.uniform(1500, 40000))
            ch = ["s_closing"]
            if recalc:                                     # add a consistent fake credit on the last row
                add = _r(d["summary"]["closing"] - old); rows[-1]["credit"] = _r(rows[-1]["credit"] + add)
                rows[-1]["balance"] = d["summary"]["closing"]; d["summary"]["total_credits"] = _r(d["summary"]["total_credits"] + add)
                ch += [f"r{len(rows)-1}_credit", f"r{len(rows)-1}_balance", "s_credits"]
            return d, ch, "closing_balance_edit" + ("_recalculated" if recalc else "")
        if kind == "name":
            d["holder"] = f"{rng.choice(names.FIRST)} {rng.choice(names.LAST)}"; return d, ["holder"], "name_edit"
        i = int(rng.integers(0, len(rows))); rows[i]["date"] = f"{int(rng.integers(1, 28)):02d}/{rows[i]['date'][3:]}"
        return d, [f"r{i}_date"], "date_edit"
    kind = rng.choice(["basic", "net", "name"], p=[0.5, 0.3, 0.2])
    if kind == "basic":
        d["earnings"][0][1] = _r(d["earnings"][0][1] * rng.uniform(1.2, 2.2)); ch = ["e0_amount"]
        if recalc:
            d["gross"] = _r(sum(v for _, v in d["earnings"])); d["net"] = _r(d["gross"] - d["total_ded"]); ch += ["gross", "net"]
        return d, ch, "salary_edit" + ("_recalculated" if recalc else "")
    if kind == "net":
        d["net"] = _r(d["net"] * rng.uniform(1.15, 2.0)); return d, ["net"], "net_pay_edit"
    d["employee"] = f"{rng.choice(names.FIRST)} {rng.choice(names.LAST)}"; return d, ["employee"], "name_edit"


# --------------------------------------------------------------------------------------------- element texts
def element_texts(d):
    """id -> displayed string, for every variable element of the document."""
    t = {}
    if d["type"] == "statement":
        t["holder"] = d["holder"]; t["acct"] = d["acct"]; t["period"] = d["period"]
        for i, r in enumerate(d["rows"]):
            t[f"r{i}_date"] = r["date"]; t[f"r{i}_desc"] = r["desc"]
            t[f"r{i}_debit"] = money(r["debit"]) if r["debit"] else ""
            t[f"r{i}_credit"] = money(r["credit"]) if r["credit"] else ""
            t[f"r{i}_balance"] = money(r["balance"])
        s = d["summary"]; t["s_opening"] = money(s["opening"]); t["s_debits"] = money(s["total_debits"])
        t["s_credits"] = money(s["total_credits"]); t["s_closing"] = money(s["closing"])
    else:
        t["employee"] = d["employee"]; t["empno"] = d["empno"]; t["month"] = d["month"]
        for i, (lab, v) in enumerate(d["earnings"]):
            t[f"e{i}_label"] = lab; t[f"e{i}_amount"] = money(v)
        for i, (lab, v) in enumerate(d["deductions"]):
            t[f"d{i}_label"] = lab; t[f"d{i}_amount"] = money(v)
        t["gross"] = money(d["gross"]); t["total_ded"] = money(d["total_ded"]); t["net"] = money(d["net"])
    return t


# ------------------------------------------------------------------------------------------------ OCR noise
def _misread(rng, s, rate):
    out = []
    for ch in s:
        if ch.isdigit() and rng.random() < rate:
            ch = str(int(rng.integers(0, 10)))
        out.append(ch)
    return "".join(out)


def _num(s):
    s = s.replace(",", "").strip()
    return float(s) if s else 0.0


def ocr_extract(rng, d, rate):
    """What a text-extraction step would read from the page: digits are occasionally misread (rate)."""
    t = {k: _misread(rng, v, rate) for k, v in element_texts(d).items()}
    if d["type"] == "statement":
        rows = [dict(date=t[f"r{i}_date"], desc=t[f"r{i}_desc"], debit=_num(t[f"r{i}_debit"]),
                     credit=_num(t[f"r{i}_credit"]), balance=_num(t[f"r{i}_balance"])) for i in range(len(d["rows"]))]
        return dict(type="statement", rows=rows, summary=dict(opening=_num(t["s_opening"]), total_debits=_num(t["s_debits"]),
                                                              total_credits=_num(t["s_credits"]), closing=_num(t["s_closing"])))
    return dict(type="payslip", earnings=[[t[f"e{i}_label"], _num(t[f"e{i}_amount"])] for i in range(len(d["earnings"]))],
                deductions=[[t[f"d{i}_label"], _num(t[f"d{i}_amount"])] for i in range(len(d["deductions"]))],
                gross=_num(t["gross"]), total_ded=_num(t["total_ded"]), net=_num(t["net"]))
