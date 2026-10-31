"""Render documents to page images, plus the image-level effects that stand in for capture, editing and generation.

Fonts come from matplotlib's bundled DejaVu files, so no system fonts are needed (works on Windows/Anaconda).
"""
import os, functools
import numpy as np
import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from . import names, doc as D

W, H = 660, 900
BG = (252, 252, 250)
SIG_POS, SIG_SIZE = (40, H - 138), (160, 65)


@functools.lru_cache(maxsize=None)
def get_font(family, bold, size):
    base = {"sans": "DejaVuSans", "serif": "DejaVuSerif"}[family] + ("-Bold" if bold else "") + ".ttf"
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", base)
    return ImageFont.truetype(path, int(size))


def _it(id_, text, x, y, anchor="la", bold=False, size=12):
    return dict(id=id_, text=text, x=x, y=y, anchor=anchor, bold=bold, size=size)


def layout(d):
    t = D.element_texts(d); items = []
    if d["type"] == "statement":
        items += [_it(None, names.BANK, 40, 22, bold=True, size=22), _it(None, "STATEMENT OF ACCOUNT", 40, 88, bold=True, size=16),
                  _it(None, "Account holder", 40, 124, size=10), _it("holder", t["holder"], 40, 140, size=14),
                  _it(None, "Account number", 360, 124, size=10), _it("acct", t["acct"], 360, 140, size=14),
                  _it(None, "Statement period", 40, 170, size=10), _it("period", t["period"], 150, 168, size=12)]
        for lab, x, a in [("Date", 40, "la"), ("Description", 110, "la"), ("Debit", 440, "ra"), ("Credit", 520, "ra"), ("Balance", 620, "ra")]:
            items.append(_it(None, lab, x, 205, a, True, 12))
        n = len(d["rows"])
        for i in range(n):
            y = 232 + 22 * i
            items += [_it(f"r{i}_date", t[f"r{i}_date"], 40, y), _it(f"r{i}_desc", t[f"r{i}_desc"][:28], 110, y),
                      _it(f"r{i}_debit", t[f"r{i}_debit"], 440, y, "ra"), _it(f"r{i}_credit", t[f"r{i}_credit"], 520, y, "ra"),
                      _it(f"r{i}_balance", t[f"r{i}_balance"], 620, y, "ra")]
        y0 = 232 + 22 * n + 24
        for k, (lab, key) in enumerate([("Opening balance", "s_opening"), ("Total debits", "s_debits"),
                                        ("Total credits", "s_credits"), ("Closing balance", "s_closing")]):
            items += [_it(None, lab, 380, y0 + 22 * k, bold=(k == 3)), _it(key, t[key], 620, y0 + 22 * k, "ra", k == 3)]
    else:
        items += [_it(None, names.EMPLOYER, 40, 22, bold=True, size=20), _it(None, "PAYSLIP", 40, 88, bold=True, size=16),
                  _it(None, "Employee", 40, 124, size=10), _it("employee", t["employee"], 40, 140, size=14),
                  _it(None, "Employee no.", 360, 124, size=10), _it("empno", t["empno"], 360, 140, size=14),
                  _it(None, "Pay month", 40, 170, size=10), _it("month", t["month"], 150, 168, size=12),
                  _it(None, "Earnings", 40, 205, bold=True, size=12), _it(None, "Deductions", 340, 205, bold=True, size=12)]
        for i in range(len(d["earnings"])):
            y = 232 + 24 * i
            items += [_it(f"e{i}_label", t[f"e{i}_label"], 40, y), _it(f"e{i}_amount", t[f"e{i}_amount"], 300, y, "ra")]
        for i in range(len(d["deductions"])):
            y = 232 + 24 * i
            items += [_it(f"d{i}_label", t[f"d{i}_label"], 340, y), _it(f"d{i}_amount", t[f"d{i}_amount"], 620, y, "ra")]
        items += [_it(None, "Gross pay", 40, 380, bold=True), _it("gross", t["gross"], 300, 380, "ra", True),
                  _it(None, "Total deductions", 340, 380, bold=True), _it("total_ded", t["total_ded"], 620, 380, "ra", True),
                  _it(None, "NET PAY", 340, 430, bold=True, size=16), _it("net", t["net"], 620, 430, "ra", True, 16)]
    items.append(_it(None, "Authorised signature", 40, H - 154, size=10))
    items.append(_it(None, f"{names.BANK if d['type']=='statement' else names.EMPLOYER} - fictional sample for research use", 40, H - 40, size=9))
    return items


def draw_item(draw, it, fam, size_delta, ink, wobble, rng, dx=0, dy=0):
    font = get_font(fam, it["bold"], it["size"] + size_delta)
    text, x, y, anc = it["text"], it["x"] + dx, it["y"] + dy, it["anchor"]
    if not text:
        return (x, y, x, y)
    if wobble <= 0:
        draw.text((x, y), text, font=font, fill=ink, anchor=anc)
        return tuple(draw.textbbox((x, y), text, font=font, anchor=anc))
    adv = [font.getlength(ch) + rng.normal(0, wobble * 0.45) for ch in text]
    total = sum(adv); x0 = x - total if anc[0] == "r" else x; cx = x0
    for ch, a in zip(text, adv):
        draw.text((cx, y + rng.normal(0, wobble * 0.6)), ch, font=font, fill=ink, anchor="l" + anc[1]); cx += a
    return (x0, y, x0 + total, y + font.size)


def render_page(d, style, rng, wobble=0.0, signature=None):
    """Returns (RGB uint8 array, {element id: box}, items). `style`: family, accent, ink."""
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 70], fill=style["accent"])
    dr.rectangle([W - 90, 14, W - 40, 56], outline=(255, 255, 255), width=3)          # abstract fictional logo mark
    dr.line([40, 225, W - 40, 225], fill=(150, 150, 150), width=1)
    items = layout(d); boxes = {}
    for it in items:
        col = (255, 255, 255) if it["y"] < 60 else style["ink"]
        b = draw_item(dr, it, style["family"], 0, col, wobble, rng)
        if it["id"]:
            boxes[it["id"]] = b
    arr = np.array(im)
    if signature is not None:
        paste_signature(arr, signature)
    x, y = SIG_POS; boxes["signature"] = (x, y, x + SIG_SIZE[0], y + SIG_SIZE[1])
    return arr, boxes, items


def paste_signature(arr, sig_gray):
    x, y = SIG_POS; s = cv2.resize(sig_gray, SIG_SIZE, interpolation=cv2.INTER_AREA)
    reg = arr[y:y + SIG_SIZE[1], x:x + SIG_SIZE[0]]
    arr[y:y + SIG_SIZE[1], x:x + SIG_SIZE[0]] = np.minimum(reg, np.stack([s] * 3, 2)).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- image effects
def effect_generated(arr, rng):
    """Smooth, texture-like finish typical of generated pages (assumption): mild blur + low-frequency + fine noise."""
    x = arr.astype(np.float32)
    x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.35, 0.8)))
    lf = rng.normal(0, 1, (9, 7)).astype(np.float32)
    lf = cv2.resize(lf, (W, H), interpolation=cv2.INTER_CUBIC) * rng.uniform(1.5, 4.5)
    x += lf[..., None] + rng.normal(0, rng.uniform(0.6, 1.8), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def effect_rescan(arr, rng):
    """Print-and-scan / phone-photo capture of a genuine document: slight rotation, grain, softening."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.normal(0, 0.35)), 1.0)
    x = cv2.warpAffine(arr, M, (W, H), borderValue=BG, flags=cv2.INTER_LINEAR).astype(np.float32)
    x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.3, 0.6)))
    x = x * rng.uniform(0.96, 1.02) + rng.normal(0, rng.uniform(1.2, 3.0), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def jpeg_roundtrip(arr, quality):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf, cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def patch_elements(arr, boxes_old, items_new, ids, edit_style, rng, seam, halo, new_signature=None):
    """Overwrite the changed elements the way an editor would. edit_style: family, size_delta, jitter(px), ink."""
    im = Image.fromarray(arr.copy()); dr = ImageDraw.Draw(im)
    bg = tuple(int(np.clip(c + (rng.integers(-3, 4) if seam else 0), 0, 255)) for c in BG)
    by_id = {it["id"]: it for it in items_new if it["id"]}; regions = []
    for i in ids:
        if i == "signature":
            continue
        x0, y0, x1, y1 = boxes_old[i]; dr.rectangle([x0 - 2, y0 - 1, x1 + 2, y1 + 1], fill=bg)
        dx = int(rng.integers(-edit_style["jitter"], edit_style["jitter"] + 1)) if edit_style["jitter"] else 0
        dy = int(rng.integers(-edit_style["jitter"], edit_style["jitter"] + 1)) if edit_style["jitter"] else 0
        nb = draw_item(dr, by_id[i], edit_style["family"], edit_style["size_delta"], edit_style["ink"], 0.0, rng, dx, dy)
        regions.append((min(x0, nb[0]) - 3, min(y0, nb[1]) - 3, max(x1, nb[2]) + 3, max(y1, nb[3]) + 3))
    out = np.array(im)
    if new_signature is not None:
        x, y = SIG_POS; out[y:y + SIG_SIZE[1], x:x + SIG_SIZE[0]] = BG
        paste_signature(out, new_signature); regions.append((x, y, x + SIG_SIZE[0], y + SIG_SIZE[1]))
    if halo:
        for x0, y0, x1, y1 in regions:
            x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), float(rng.uniform(0.5, 0.9)))
    return out
