"""Linked-case generator: one case_id ties together a page image, a metadata record, extracted numbers and a signature.

Labels: genuine | ai_generated | ai_edited, with the manipulation type recorded. Genuine and manipulated cases
overlap by design (see config.py), so a perfect score is not achievable.
"""
import json, os, copy
import numpy as np
import pandas as pd
from . import doc as D, render as R, signature as S
from .config import GenConfig

BANK_SYS = ["KTB StatementEngine 4.2", "KTB StatementEngine 5.0", "PayrollSuite 11", "PayrollSuite 12"]
SCANNER = ["ScanPro Mobile 3.1", "OfficeScan 8.0"]
EDITOR = ["PDF Editor Pro 9", "PixelStudio 24", "DocCanvas Online", "QuickPDF Editor"]
GENERATIVE = ["ImageSynth v2", "python-pdf-lib", "Unknown", ""]


def _clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))


def make_metadata(rng, cfg, label, doc_type, rescanned, laundered, clean_font, size_delta, benign_tool):
    base_fonts, base_sizes = 3, 6
    m = dict(has_text_layer=True, producer="", creator="", create_day=float(rng.uniform(0, 400)), mod_delta_days=0.0,
             n_fonts=base_fonts, n_font_sizes=base_sizes, xmp_present=True, n_meta_fields=int(rng.integers(8, 13)),
             has_edit_history=False)
    if label == "genuine":
        odd = rng.random() < cfg.p_genuine_odd_meta
        if rescanned:
            m["producer"] = str(rng.choice(SCANNER)); m["creator"] = m["producer"]
            m["has_text_layer"] = bool(rng.random() < 0.25); ocr_mu = rng.normal(0.945, 0.03)
        else:
            m["producer"] = str(rng.choice(BANK_SYS)); m["creator"] = m["producer"]
            m["has_text_layer"] = bool(rng.random() < 0.88); ocr_mu = rng.normal(0.972, 0.014)
        if odd:                                           # legitimate but unusual provenance
            kind = rng.choice(["editor_resave", "stripped"])
            if kind == "editor_resave":
                m["producer"] = str(rng.choice(EDITOR)); m["has_edit_history"] = bool(rng.random() < 0.3)
                m["mod_delta_days"] = float(rng.uniform(0.02, 20)); m["n_fonts"] = base_fonts + int(rng.integers(0, 2))
            else:
                m["producer"] = str(rng.choice(GENERATIVE)); m["creator"] = ""; m["n_meta_fields"] = int(rng.integers(0, 5)); m["xmp_present"] = False
        elif rng.random() < 0.22:
            m["mod_delta_days"] = float(rng.uniform(0, 25))
        if m["has_text_layer"]:
            m["n_fonts"] = int(rng.choice([2, 3, 4], p=[0.15, 0.7, 0.15])) if not (odd and m["n_fonts"] > base_fonts) else m["n_fonts"]
        else:
            m["n_fonts"] = 0
    elif label == "ai_edited":
        benign = benign_tool or (rng.random() < cfg.p_edit_meta_reset)
        m["creator"] = str(rng.choice(BANK_SYS))
        m["producer"] = str(rng.choice(BANK_SYS + SCANNER)) if benign else str(rng.choice(EDITOR))
        if benign: m["creator"] = m["producer"]
        m["has_text_layer"] = bool(rng.random() < 0.75)
        extra = 0 if clean_font else int(rng.integers(1, 3))
        m["n_fonts"] = (int(rng.choice([2, 3, 4], p=[0.15, 0.7, 0.15])) + extra) if m["has_text_layer"] else 0
        m["n_font_sizes"] = base_sizes + (0 if size_delta == 0 else 1)
        m["mod_delta_days"] = float(rng.uniform(0.02, 30)) if rng.random() < 0.55 else 0.0
        m["has_edit_history"] = bool((not benign) and rng.random() < 0.4)
        m["n_meta_fields"] = int(rng.integers(6, 12))
        ocr_mu = rng.normal(0.962 if not laundered else 0.945, 0.02)
    else:  # ai_generated
        clean = rng.random() < cfg.p_gen_meta_clean
        if clean:
            m["producer"] = str(rng.choice(BANK_SYS)); m["creator"] = m["producer"]
        else:
            m["producer"] = str(rng.choice(GENERATIVE)); m["creator"] = str(rng.choice(["", "Unknown", m["producer"]]))
        m["has_text_layer"] = bool(rng.random() < cfg.p_gen_text_layer)
        m["n_fonts"] = int(rng.integers(3, 8)) if m["has_text_layer"] else 0
        m["n_font_sizes"] = int(rng.integers(4, 9))
        m["xmp_present"] = bool(rng.random() < 0.5) if not clean else True
        m["n_meta_fields"] = int(rng.integers(8, 13)) if clean else int(rng.integers(0, 8))
        ocr_mu = rng.normal(0.958, 0.03)
    mean_c = _clip(ocr_mu, 0.80, 0.999); sd = abs(rng.normal(0.02, 0.008)) + 0.004
    m["ocr_mean_conf"] = mean_c; m["ocr_conf_std"] = _clip(sd, 0.002, 0.2)
    m["ocr_min_conf"] = _clip(mean_c - abs(rng.normal(3.0 * sd, sd)), 0.3, mean_c)
    m["ocr_low_conf_frac"] = _clip(abs(rng.normal(max(0.0, (0.985 - mean_c)) * 1.3, 0.012)), 0.0, 0.6)
    return m


def make_case(i, cfg: GenConfig, identities, out_dir):
    rng = np.random.default_rng([cfg.seed, i])
    label = str(rng.choice(["genuine", "ai_generated", "ai_edited"], p=[cfg.p_genuine, cfg.p_ai_generated, cfg.p_ai_edited]))
    doc_type = "statement" if rng.random() < cfg.p_statement else "payslip"
    ident_id = int(rng.integers(cfg.n_identities)); ident = identities[ident_id]
    case_id = f"C{i:06d}"

    # ---- content
    if label == "ai_generated":
        truth = D.make_document(rng, cfg, doc_type, generated=True)
        if rng.random() < cfg.p_gen_arith_error:
            truth = D.corrupt_arithmetic(rng, truth)
    else:
        truth = D.make_document(rng, cfg, doc_type)
        if rng.random() < cfg.p_genuine_arith_flaw:
            truth = D.corrupt_arithmetic(rng, truth, mild=True)
    shown, changed, manip = truth, [], "none"
    if label == "ai_edited":
        shown, changed, manip = D.apply_edit(rng, cfg, truth)
    if label == "ai_generated":
        manip = "full_generation"

    # ---- signature
    reference = S.render_signature(ident, rng, sigma=1.5)
    forged_sig = label == "ai_generated" or (label == "ai_edited" and rng.random() < cfg.p_sig_edit_swapped)
    if forged_sig:
        sg = float(rng.uniform(*cfg.sig_forger_sigma_range))
        claimed = S.render_signature(ident, rng, sigma=sg, pen_uniform=bool(rng.random() < 0.6), pen_noise=bool(rng.random() < 0.5))
        if label == "ai_edited":
            changed = changed + ["signature"]; manip = manip + "+signature_swap"
    else:
        claimed = S.render_signature(ident, rng, sigma=1.5)

    # ---- image
    style = dict(family=str(rng.choice(["sans", "serif"])), accent=tuple(int(v) for v in rng.choice([[24, 62, 98], [20, 90, 70], [90, 30, 60], [60, 60, 70]])),
                 ink=(int(rng.integers(12, 40)),) * 3)
    rescanned = laundered = False; clean_font = benign = False; size_delta = 0
    if label == "genuine":
        arr, _, _ = R.render_page(truth, style, rng, 0.0, claimed)
        rescanned = bool(rng.random() < cfg.p_genuine_rescanned)
        if rescanned:
            arr = R.effect_rescan(arr, rng); _, arr = R.jpeg_roundtrip(arr, rng.integers(72, 90))   # first save
            q = int(rng.integers(70, 90)) if rng.random() < 0.5 else int(rng.integers(86, 95))      # often re-saved
        else:
            q = int(rng.integers(88, 96))
        buf, _ = R.jpeg_roundtrip(arr, q)
    elif label == "ai_generated":
        wob = float(rng.uniform(*cfg.gen_wobble_range))
        arr, _, _ = R.render_page(truth, style, rng, wob, claimed)
        arr = R.effect_generated(arr, rng); buf, _ = R.jpeg_roundtrip(arr, int(rng.integers(85, 96)))
    else:
        orig_sig = S.render_signature(ident, rng, sigma=1.5) if forged_sig else claimed   # the page originally carries a genuine signature
        arr, boxes, _ = R.render_page(truth, style, rng, 0.0, orig_sig)
        q1 = int(rng.integers(85, 96)); _, arr1 = R.jpeg_roundtrip(arr, q1)
        _, boxes_old, _ = R.render_page(truth, style, np.random.default_rng([cfg.seed, i, 7]), 0.0, None)
        _, _, items_new = R.render_page(shown, style, np.random.default_rng([cfg.seed, i, 7]), 0.0, None)
        clean_font = bool(rng.random() < cfg.p_edit_clean_font); benign = bool(rng.random() < cfg.p_edit_benign_tool)
        if clean_font:
            es = dict(family=style["family"], size_delta=0, jitter=0, ink=style["ink"])
        else:
            kind = rng.choice(["family", "size", "jitter"])
            es = dict(family=("serif" if style["family"] == "sans" else "sans") if kind == "family" else style["family"],
                      size_delta=int(rng.choice([-1, 1])) if kind == "size" else 0,
                      jitter=int(rng.integers(1, 3)) if kind == "jitter" else 0, ink=tuple(min(255, v + 12) for v in style["ink"]))
        size_delta = es["size_delta"]
        seam = bool(rng.random() < 0.5); halo = bool(rng.random() > cfg.p_edit_no_halo)
        edited = R.patch_elements(arr1, boxes_old, items_new, changed, es, rng, seam, halo,
                                  new_signature=claimed if forged_sig else None)
        laundered = bool(rng.random() < cfg.p_edit_laundered)
        if laundered:
            edited = R.effect_rescan(edited, rng)
        q2 = q1 if rng.random() < cfg.p_edit_same_quality else int(rng.integers(76, 94))
        buf, _ = R.jpeg_roundtrip(edited, q2)

    img_path = os.path.join(out_dir, "images", f"{case_id}.jpg"); buf.tofile(img_path)
    import cv2
    cv2.imwrite(os.path.join(out_dir, "signatures", f"{case_id}_claimed.png"), claimed)
    cv2.imwrite(os.path.join(out_dir, "signatures", f"{case_id}_reference.png"), reference)

    # ---- metadata + extracted numbers
    meta = make_metadata(rng, cfg, label, doc_type, rescanned, laundered, clean_font, size_delta, benign)
    if rescanned or laundered:
        rate = float(rng.uniform(0.004, 0.02))
    elif label == "genuine":
        rate = float(rng.uniform(0.0, 0.003))
    else:
        rate = float(rng.uniform(0.0, 0.010))
    extracted = D.ocr_extract(rng, shown, rate)

    return dict(case_id=case_id, label=label, y=int(label != "genuine"), manipulation=manip, doc_type=doc_type,
                identity_id=ident_id, image=f"images/{case_id}.jpg", sig_claimed=f"signatures/{case_id}_claimed.png",
                sig_reference=f"signatures/{case_id}_reference.png", ocr_rate=rate, metadata=meta, extracted=extracted,
                rescanned=rescanned, laundered=laundered, clean_font=clean_font, benign_tool=benign)


def generate_dataset(cfg: GenConfig, out_dir: str, verbose=True) -> pd.DataFrame:
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True); os.makedirs(os.path.join(out_dir, "signatures"), exist_ok=True)
    rid = np.random.default_rng([cfg.seed, 10 ** 6])
    identities = [S.make_identity(rid) for _ in range(cfg.n_identities)]
    recs = []
    for i in range(cfg.n_cases):
        recs.append(make_case(i, cfg, identities, out_dir))
        if verbose and (i + 1) % 300 == 0:
            print(f"  generated {i + 1}/{cfg.n_cases} cases", flush=True)
    with open(os.path.join(out_dir, "cases.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "generator_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return pd.DataFrame([{k: r[k] for k in ("case_id", "label", "y", "manipulation", "doc_type", "identity_id")} for r in recs])
