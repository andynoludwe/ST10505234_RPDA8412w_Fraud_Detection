"""End-to-end run: generate linked cases -> features -> four branch models (out-of-fold) -> join by case_id -> fusion -> evaluation."""
import json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import GenConfig, LABELS
from .generator import generate_dataset
from .features import build_features
from . import models as M, evaluate as E, combine as C

BR = C.BRANCHES


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))


def run(cfg: GenConfig, out_dir: str, n_jobs=-1, reuse_data=False, n_boot=500, verbose=True):
    t0 = time.time(); log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    data_dir, pred_dir, res_dir = (os.path.join(out_dir, d) for d in ("data", "predictions", "results"))
    for d in (pred_dir, res_dir): os.makedirs(d, exist_ok=True)
    log(f"[{cfg.difficulty}] 1/6 generating {cfg.n_cases} linked cases")
    if not (reuse_data and os.path.exists(os.path.join(data_dir, "cases.jsonl"))):
        generate_dataset(cfg, data_dir, verbose=verbose)
    log(f"[{cfg.difficulty}] 2/6 extracting features")
    meta, feats = build_features(data_dir, n_jobs=n_jobs)
    ids = meta.index
    feats = {k: v.reindex(ids).fillna(0.0) for k, v in feats.items()}

    log(f"[{cfg.difficulty}] 3/6 leakage-free split by signature identity")
    tr_ids, te_ids = M.split_by_identity(meta, 0.3, cfg.seed)
    y_tr, y_te = meta.loc[tr_ids, "y"].values, meta.loc[te_ids, "y"].values
    g_tr = meta.loc[tr_ids, "identity_id"].values
    split_info = dict(n_train=len(tr_ids), n_test=len(te_ids), train_pos_rate=float(y_tr.mean()), test_pos_rate=float(y_te.mean()),
                      identities_train=int(len(set(g_tr))), identities_test=int(meta.loc[te_ids, "identity_id"].nunique()))

    log(f"[{cfg.difficulty}] 4/6 training the four branches (5-fold out-of-fold on train)")
    for b in BR:
        Xtr, Xte = feats[b].loc[tr_ids].values, feats[b].loc[te_ids].values
        mk = lambda b=b: M.make_branch_model(b, cfg.seed)
        oof = M.oof_predict(mk, Xtr, y_tr, g_tr, seed=cfg.seed)
        full = mk().fit(Xtr, y_tr); pte = full.predict_proba(Xte)[:, 1]
        C.write_branch_predictions(pred_dir, b, list(tr_ids) + list(te_ids), ["train"] * len(tr_ids) + ["test"] * len(te_ids),
                                   np.concatenate([oof, pte]))
    P = C.join_predictions(pred_dir)               # case_id-keyed join of the branches' real outputs
    Ptr, Pte = P.loc[tr_ids], P.loc[te_ids]
    cols = [f"p_{b}" for b in BR]
    Ztr, Zte = M.logit(Ptr[cols].values), M.logit(Pte[cols].values)

    log(f"[{cfg.difficulty}] 5/6 fusion and baselines")
    scores = {}      # name -> (oof_train, test)
    for j, b in enumerate(BR):
        scores[f"branch_{b}"] = (Ptr[cols[j]].values, Pte[cols[j]].values)
    scores["mean_of_branches"] = (Ptr[cols].values.mean(1), Pte[cols].values.mean(1))
    for kind, name in (("lr", "fusion_lr"), ("gb", "fusion_gb")):
        mk = lambda kind=kind: M.make_fusion(kind, cfg.seed)
        oo = M.oof_predict(mk, Ztr, y_tr, g_tr, seed=cfg.seed + 1)
        scores[name] = (oo, mk().fit(Ztr, y_tr).predict_proba(Zte)[:, 1])
    Xall_tr = np.hstack([feats[b].loc[tr_ids].values for b in BR]); Xall_te = np.hstack([feats[b].loc[te_ids].values for b in BR])
    mk = lambda: M.make_branch_model("image", cfg.seed)
    oo = M.oof_predict(mk, Xall_tr, y_tr, g_tr, seed=cfg.seed + 2)
    scores["early_fusion_all_features"] = (oo, mk().fit(Xall_tr, y_tr).predict_proba(Xall_te)[:, 1])

    thr = {k: M.best_f1_threshold(y_tr, v[0]) for k, v in scores.items()}      # chosen on train OOF only
    table = {k: {**E.metrics(y_te, v[1], thr[k]), "oof_auc_train": float(roc_auc_score(y_tr, v[0]))} for k, v in scores.items()}
    best_branch = max([f"branch_{b}" for b in BR], key=lambda k: table[k]["oof_auc_train"])    # picked on train, not on test
    preds = {k: (v[1] >= thr[k]).astype(int) for k, v in scores.items()}
    ci = {k: E.bootstrap_ci(y_te, scores[k][1], thr[k], n_boot, cfg.seed, other=scores[best_branch][1] if k == "fusion_lr" else None) for k in scores}
    mc = {f"fusion_lr_vs_{o}": E.mcnemar(y_te, preds["fusion_lr"], preds[o]) for o in (best_branch, "mean_of_branches", "early_fusion_all_features", "fusion_gb")}

    # ablation: drop one branch from the fusion
    abl = {}
    for j, b in enumerate(BR):
        keep = [i for i in range(len(BR)) if i != j]
        m = M.make_fusion("lr", cfg.seed).fit(Ztr[:, keep], y_tr); p = m.predict_proba(Zte[:, keep])[:, 1]
        oo = M.oof_predict(lambda: M.make_fusion("lr", cfg.seed), Ztr[:, keep], y_tr, g_tr, seed=cfg.seed + 1)
        abl[f"without_{b}"] = E.metrics(y_te, p, M.best_f1_threshold(y_tr, oo))
    # importance
    lr = M.make_fusion("lr", cfg.seed).fit(Ztr, y_tr); coef = dict(zip(BR, map(float, lr.coef_[0])))
    rng = np.random.default_rng(cfg.seed); base = roc_auc_score(y_te, lr.predict_proba(Zte)[:, 1]); perm = {}
    for j, b in enumerate(BR):
        drops = []
        for _ in range(30):
            Zp = Zte.copy(); Zp[:, j] = rng.permutation(Zp[:, j]); drops.append(base - roc_auc_score(y_te, lr.predict_proba(Zp)[:, 1]))
        perm[b] = float(np.mean(drops))
    shap_imp = None
    try:
        import shap
        ex = shap.LinearExplainer(lr, Ztr); sv = np.asarray(ex.shap_values(Zte)); shap_imp = dict(zip(BR, map(float, np.abs(sv).mean(0))))
        pd.DataFrame(sv, index=te_ids, columns=BR).to_csv(os.path.join(res_dir, "fusion_shap_values.csv"))
    except Exception as e:
        log("   (shap not available or failed; using permutation importance only):", type(e).__name__)

    # subgroups on the test set (fusion_lr flagged rate)
    mt = meta.loc[te_ids]; pf = preds["fusion_lr"]; sub = {}
    sub["by_label_flagged_rate"] = mt.assign(f=pf).groupby("label")["f"].mean().to_dict()
    sub["by_manipulation_flagged_rate"] = mt.assign(f=pf).groupby("manipulation")["f"].agg(["mean", "count"]).round(3).to_dict("index")
    sub["genuine_rescanned_vs_born_digital_flagged_rate"] = mt[mt.label == "genuine"].assign(f=pf[(mt.label == "genuine").values]).groupby("rescanned")["f"].mean().rename(index=str).to_dict()
    ed = mt.label == "ai_edited"
    sub["edited_hard_cases_flagged_rate"] = {c: float(pf[(ed & (mt[c] == v)).values].mean()) if (ed & (mt[c] == v)).sum() else None
                                             for c, v in (("clean_font", True), ("benign_tool", True), ("laundered", True))}
    sub["by_doc_type_flagged_rate"] = {f"{a}|{b}": v for (a, b), v in mt.assign(f=pf).groupby(["doc_type", "label"])["f"].mean().round(3).items()}
    # three-class type classifier on all features (secondary task)
    lab_map = {l: i for i, l in enumerate(LABELS)}; ytr3 = meta.loc[tr_ids, "label"].map(lab_map).values; yte3 = meta.loc[te_ids, "label"].map(lab_map).values
    m3 = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=cfg.seed).fit(Xall_tr, ytr3)
    p3 = m3.predict(Xall_te)
    type_task = dict(labels=LABELS, accuracy=float(accuracy_score(yte3, p3)), macro_f1=float(f1_score(yte3, p3, average="macro")),
                     confusion_matrix=confusion_matrix(yte3, p3, labels=[0, 1, 2]).tolist())

    log(f"[{cfg.difficulty}] 6/6 writing results and figures")
    results = dict(config=cfg.to_dict(), split=split_info, thresholds=thr, test_metrics=table, best_single_branch_by_train_oof=best_branch,
                   bootstrap_ci=ci, mcnemar_fusion_lr=mc, ablation_fusion_lr=abl, fusion_lr_coefficients=coef,
                   fusion_lr_permutation_auc_drop=perm, fusion_lr_mean_abs_shap=shap_imp, subgroups=sub, type_classification=type_task,
                   runtime_seconds=round(time.time() - t0, 1))
    _save_json(results, os.path.join(res_dir, "results.json"))
    pd.DataFrame(table).T.round(4).to_csv(os.path.join(res_dir, "test_metrics.csv"))
    out_pred = pd.DataFrame({k: v[1] for k, v in scores.items()}, index=te_ids).join(meta.loc[te_ids, ["label", "manipulation", "y"]])
    out_pred.to_csv(os.path.join(res_dir, "test_predictions_all_models.csv"))
    _figures(res_dir, y_te, scores, table, abl, perm, best_branch, cfg)
    log(f"[{cfg.difficulty}] done in {results['runtime_seconds']} s")
    return results


def _figures(res_dir, y_te, scores, table, abl, perm, best_branch, cfg):
    show = [best_branch, "mean_of_branches", "early_fusion_all_features", "fusion_lr", "fusion_gb"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k in show:
        fpr, tpr, _ = roc_curve(y_te, scores[k][1]); ax[0].plot(fpr, tpr, label=f"{k} (AUC {table[k]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_te, scores[k][1]); ax[1].plot(rc, pr, label=f"{k} (AP {table[k]['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test, synthetic data)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(res_dir, "roc_pr.png"), dpi=140); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    names = ["full fusion"] + [k.replace("without_", "w/o ") for k in abl]; vals = [table["fusion_lr"]["roc_auc"]] + [abl[k]["roc_auc"] for k in abl]
    ax[0].barh(names[::-1], vals[::-1], color="#4c78a8"); ax[0].set(xlim=(max(0.5, min(vals) - 0.05), 1.0), xlabel="Test ROC-AUC", title="Ablation: remove one branch")
    ax[1].barh(list(perm)[::-1], list(perm.values())[::-1], color="#e45756"); ax[1].set(xlabel="AUC drop when shuffled", title="Fusion permutation importance")
    fig.tight_layout(); fig.savefig(os.path.join(res_dir, "ablation_importance.png"), dpi=140); plt.close(fig)
