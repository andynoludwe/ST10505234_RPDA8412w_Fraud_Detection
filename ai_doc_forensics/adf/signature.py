"""Synthetic signatures (stroke-based identities) and pair-similarity features.

The signature branch here is a classical stand-in for the Siamese CNN: it compares a claimed signature with the
account's reference specimen using similarity measures. Swap in your Siamese model by writing its
forgery probability to the same predictions format (case_id, p).
"""
import numpy as np
import cv2
from skimage.feature import hog
from skimage.metrics import structural_similarity

W, H = 220, 90


def make_identity(rng):
    """An identity = a few smooth strokes defined by control points (in a 220x90 box)."""
    strokes = []
    for _ in range(int(rng.integers(2, 5))):
        n = int(rng.integers(4, 8))
        x = np.sort(rng.uniform(15, W - 15, n))
        y = rng.uniform(15, H - 15, n)
        strokes.append(np.stack([x, y], 1))
    return {"strokes": strokes, "width": float(rng.uniform(1.6, 3.2))}


def _smooth(points, k=40):
    t = np.linspace(0, 1, len(points))
    tt = np.linspace(0, 1, k * (len(points) - 1))
    return np.stack([np.interp(tt, t, points[:, 0]), np.interp(tt, t, points[:, 1])], 1)


def render_signature(identity, rng, sigma, scale_jit=0.05, rot_deg=3.0, pen_uniform=False, pen_noise=True):
    """Render one signature. `sigma` = point jitter in px (small for genuine, larger for forgeries)."""
    img = np.full((H, W), 255, np.uint8)
    ang = np.deg2rad(rng.normal(0, rot_deg)); s = 1 + rng.normal(0, scale_jit)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    c = np.array([W / 2, H / 2])
    for pts in identity["strokes"]:
        p = pts + rng.normal(0, sigma, pts.shape)
        p = (p - c) @ R.T + c
        curve = _smooth(p)
        for i in range(len(curve) - 1):
            w = identity["width"]
            if not pen_uniform:                       # human pen pressure varies along the stroke
                w = w * (0.75 + 0.5 * np.sin(i / 7.0 + rng.uniform(0, 6)) ** 2)
            cv2.line(img, tuple(np.round(curve[i]).astype(int)), tuple(np.round(curve[i + 1]).astype(int)),
                     0, max(1, int(round(w))), cv2.LINE_AA)
    if pen_noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


def _crop_norm(img):
    ink = img < 140
    if ink.sum() < 5:
        return np.full((32, 96), 255, np.uint8), ink
    ys, xs = np.where(ink)
    c = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(c, (96, 32), interpolation=cv2.INTER_AREA), ink


def _stroke_width(ink):
    if ink.sum() < 5:
        return 0.0, 0.0
    d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    sk = d[d > 0]
    return float(np.mean(sk)), float(np.std(sk))


def signature_features(claimed, reference):
    """Similarity features between the claimed and reference signature images (uint8, white background)."""
    a, ia = _crop_norm(claimed); b, ib = _crop_norm(reference)
    af, bf = a.astype(np.float32), b.astype(np.float32)
    corr = float(np.corrcoef(af.ravel(), bf.ravel())[0, 1]) if af.std() > 0 and bf.std() > 0 else 0.0
    ssim = float(structural_similarity(a, b, data_range=255))
    ha = hog(a, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    hb = hog(b, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    hog_cos = float(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9))
    sw_a, sws_a = _stroke_width(ia); sw_b, sws_b = _stroke_width(ib)

    def bbox_aspect(ink):
        ys, xs = np.where(ink)
        return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) if len(xs) else 1.0

    ca = np.array(np.where(ia)).mean(1) if ia.any() else np.zeros(2)
    cb = np.array(np.where(ib)).mean(1) if ib.any() else np.zeros(2)
    return {
        "sig_corr": corr, "sig_ssim": ssim, "sig_hog_cos": hog_cos,
        "sig_density_diff": abs(float(ia.mean() - ib.mean())),
        "sig_aspect_diff": abs(bbox_aspect(ia) - bbox_aspect(ib)),
        "sig_strokewidth_diff": abs(sw_a - sw_b),
        "sig_pen_uniformity_claimed": sws_a / (sw_a + 1e-6),   # human pens vary; synthetic pens are uniform
        "sig_centroid_shift": float(np.linalg.norm(ca - cb)),
    }
