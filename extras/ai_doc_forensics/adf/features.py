"""Feature extraction for the four branches. Everything is computed from the stored artefacts of each case
(the JPEG, the metadata record, the extracted numbers and the two signature images), as a deployed system would."""
import json, os, math
import numpy as np
import pandas as pd
import cv2
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from . import generator as G, signature as S, doc as D


# ------------------------------------------------------------------------------------------------ image branch
def _blocks(a, gy=6, gx=8):
    h, w = a.shape[:2]; ys = np.linspace(0, h, gy + 1, dtype=int); xs = np.linspace(0, w, gx + 1, dtype=int)
    return [[a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] for j in range(gx)] for i in range(gy)]


def image_features(path):
    bgr = cv2.imread(path); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = {}
    ela = {}
    for q in (75, 90, 95):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q]); rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ela[q] = np.abs(bgr.astype(np.float32) - rec.astype(np.float32)).mean(2)
        f[f"ela{q}_mean"] = float(ela[q].mean())
    e = ela[90]
    f["ela90_std"] = float(e.std()); f["ela90_p99"] = float(np.percentile(e, 99)); f["ela90_max"] = float(e.max())
    f["ela_ratio_75_95"] = f["ela75_mean"] / (f["ela95_mean"] + 1e-6)
    bm = np.array([[b.mean() for b in row] for row in _blocks(e)]).ravel()
    med = np.median(bm); mad = np.median(np.abs(bm - med)) + 1e-6
    f["ela_blk_cv"] = float(bm.std() / (bm.mean() + 1e-6)); f["ela_blk_maxratio"] = float(bm.max() / (med + 1e-6))
    f["ela_blk_outlier_frac"] = float(np.mean(bm > med + 4 * mad))
    res = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    f["res_std"] = float(res.std()); f["res_kurt"] = float(kurtosis(res.ravel()))
    rs = np.array([[b.std() for b in row] for row in _blocks(res)]).ravel()
    f["res_blk_cv"] = float(rs.std() / (rs.mean() + 1e-6)); f["res_blk_maxratio"] = float(rs.max() / (np.median(rs) + 1e-6))
    k = np.ones((5, 5), np.uint8)
    flat = ((cv2.dilate(gray, k) - cv2.erode(gray, k)) < 10) & (gray > 215)
    f["bg_std"] = float(gray[flat].std()) if flat.sum() > 100 else 0.0
    lf = cv2.GaussianBlur(gray, (0, 0), 25)
    f["bg_lowfreq_std"] = float(lf[flat].std()) if flat.sum() > 100 else 0.0
    F = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))); h, w = F.shape
    yy, xx = np.ogrid[:h, :w]; r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)
    tot = (F ** 2).sum() + 1e-9
    f["hf_energy_frac"] = float((F[r > 0.30] ** 2).sum() / tot); f["mf_energy_frac"] = float((F[(r > 0.08) & (r <= 0.30)] ** 2).sum() / tot)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    f["lap_mean"] = float(lap.mean())
    lb = np.array([[b.mean() for b in row] for row in _blocks(lap)]).ravel(); lb = lb[lb > np.percentile(lb, 30)]
    f["lap_blk_cv"] = float(lb.std() / (lb.mean() + 1e-6))
    d = np.abs(np.diff(gray, axis=1)); cols = np.arange(d.shape[1])
    f["blockiness"] = float(d[:, cols % 8 == 7].mean() / (d[:, cols % 8 != 7].mean() + 1e-6))
    return f


# ---------------------------------------------------------------------------------------- metadata/OCR/font branch
def _family(p):
    if p in G.BANK_SYS: return "bank"
    if p in G.SCANNER: return "scanner"
    if p in G.EDITOR: return "editor"
    return "other"


def metadata_features(m):
    fam = _family(m["producer"])
    f = {f"meta_prod_{k}": float(fam == k) for k in ("bank", "scanner", "editor", "other")}
    f.update(meta_producer_empty=float(m["producer"] == ""), meta_creator_mismatch=float(m["creator"] != m["producer"]),
             meta_has_text=float(m["has_text_layer"]), meta_n_fonts=m["n_fonts"], meta_n_font_sizes=m["n_font_sizes"],
             meta_xmp=float(m["xmp_present"]), meta_n_fields=m["n_meta_fields"], meta_edit_history=float(m["has_edit_history"]),
             meta_mod_delta_log=math.log1p(m["mod_delta_days"]), meta_mod_changed=float(m["mod_delta_days"] > 0),
             ocr_mean=m["ocr_mean_conf"], ocr_min=m["ocr_min_conf"], ocr_std=m["ocr_conf_std"], ocr_low_frac=m["ocr_low_conf_frac"])
    return f


# ------------------------------------------------------------------------------------------------ arithmetic branch
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _benford_chi2(vals):
    v = [abs(x) for x in vals if abs(x) >= 1]
    if len(v) < 5: return 0.0
    d = np.array([int(str(int(x))[0]) for x in v]); obs = np.bincount(d, minlength=10)[1:10]
    exp = BENFORD * len(v)
    return float(((obs - exp) ** 2 / (exp + 1e-9)).sum() / len(v))


def arithmetic_features(x):
    f = dict(is_payslip=float(x["type"] == "payslip"), recon_err_count=0., recon_err_frac=0., recon_err_max=0., recon_err_small=0.,
             recon_err_large=0., totals_diff_max=0., closing_vs_last=0., closing_expected=0., date_violations=0.,
             gross_diff=0., ded_diff=0., net_diff=0., paye_dev=0., uif_dev=0., benford_chi2=0., round_frac=0., dup_frac=0.)
    if x["type"] == "statement":
        rows, s = x["rows"], x["summary"]; prev = s["opening"]; errs = []
        for r in rows:
            errs.append(abs(prev + r["credit"] - r["debit"] - r["balance"])); prev = r["balance"]
        errs = np.array(errs)
        f.update(recon_err_count=float((errs > 0.005).sum()), recon_err_frac=float((errs > 0.005).mean()), recon_err_max=float(errs.max()),
                 recon_err_small=float(((errs > 0.005) & (errs <= 1.0)).sum()), recon_err_large=float((errs > 1.0).sum()))
        td = abs(sum(r["debit"] for r in rows) - s["total_debits"]); tc = abs(sum(r["credit"] for r in rows) - s["total_credits"])
        f["totals_diff_max"] = float(max(td, tc)); f["closing_vs_last"] = float(abs(s["closing"] - rows[-1]["balance"]))
        f["closing_expected"] = float(abs(s["opening"] + s["total_credits"] - s["total_debits"] - s["closing"]))
        days = []
        for r in rows:
            try: days.append(int(r["date"][:2]))
            except ValueError: days.append(0)
        f["date_violations"] = float(sum(1 for a, b in zip(days, days[1:]) if b < a))
        amounts = [r["debit"] for r in rows if r["debit"]] + [r["credit"] for r in rows if r["credit"]]
    else:
        e = [v for _, v in x["earnings"]]; dd = {k: v for k, v in x["deductions"]}
        f["gross_diff"] = float(abs(sum(e) - x["gross"])); f["ded_diff"] = float(abs(sum(dd.values()) - x["total_ded"]))
        f["net_diff"] = float(abs(x["gross"] - x["total_ded"] - x["net"]))
        g = max(x["gross"], 1.0); f["paye_dev"] = float(abs(dd.get("PAYE", 0) - D._paye(x["gross"])) / g)
        f["uif_dev"] = float(abs(dd.get("UIF", 0) - D._uif(x["gross"])))
        amounts = e + list(dd.values())
    f["benford_chi2"] = _benford_chi2(amounts)
    f["round_frac"] = float(np.mean([abs(a * 100 - round(a * 100)) < 1e-6 and round(a * 100) % 100 == 0 for a in amounts])) if amounts else 0.0
    f["dup_frac"] = float(1 - len(set(amounts)) / len(amounts)) if amounts else 0.0
    return f


# ----------------------------------------------------------------------------------------------------- driver
def _one(rec, data_dir):
    cl = cv2.imread(os.path.join(data_dir, rec["sig_claimed"]), cv2.IMREAD_GRAYSCALE)
    rf = cv2.imread(os.path.join(data_dir, rec["sig_reference"]), cv2.IMREAD_GRAYSCALE)
    return (rec["case_id"], image_features(os.path.join(data_dir, rec["image"])), metadata_features(rec["metadata"]),
            arithmetic_features(rec["extracted"]), S.signature_features(cl, rf))


def build_features(data_dir, n_jobs=-1):
    recs = [json.loads(l) for l in open(os.path.join(data_dir, "cases.jsonl"))]
    out = Parallel(n_jobs=n_jobs)(delayed(_one)(r, data_dir) for r in recs)
    ids = [o[0] for o in out]
    meta = pd.DataFrame([{k: r[k] for k in ("case_id", "label", "y", "manipulation", "doc_type", "identity_id",
                                           "rescanned", "laundered", "clean_font", "benign_tool")} for r in recs]).set_index("case_id")
    feats = {n: pd.DataFrame([o[i] for o in out], index=ids) for n, i in
             (("image", 1), ("metadata", 2), ("arithmetic", 3), ("signature", 4))}
    return meta, feats
