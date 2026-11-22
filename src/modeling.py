"""Ham dung chung cho Phan 5: dac ta mo hinh, chi so xep hang, nguong theo chi phi, hieu chuan."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import FEATURE_COLS, make_preprocessor

SEED = 42
BASE3 = ["eq_ratio", "npl_ratio", "roa"]
HGB_CONFIGS = {
    "A": dict(learning_rate=0.1, max_depth=3, max_iter=150, l2_regularization=1.0),
    "B": dict(learning_rate=0.1, max_depth=5, max_iter=150, l2_regularization=1.0),
    "C": dict(learning_rate=0.05, max_depth=4, max_iter=300, l2_regularization=1.0),
}


def model_specs():
    specs = {
        "baseline_rule_low_capital": {"type": "rule", "cols": ["eq_ratio"]},    # xep hang theo von thap, khong hoc
        "baseline_logit_3": {"type": "logit", "cols": BASE3},                    # hoi quy logistic 3 bien
        "logit_28": {"type": "logit", "cols": FEATURE_COLS},
    }
    for k, p in HGB_CONFIGS.items():
        specs[f"hgb_28_{k}"] = {"type": "hgb", "cols": FEATURE_COLS, "params": p}
    return specs


def fit_predict(spec, X_train, y_train, X_apply):
    """X_*: ma tran numpy gom du FEATURE_COLS. Moi tien xu ly (winsorize, impute, scale) CHI fit tren X_train."""
    idx = [FEATURE_COLS.index(c) for c in spec["cols"]]
    xtr, xap = X_train[:, idx], X_apply[:, idx]
    if spec["type"] == "rule":
        fill = np.nanmedian(xtr[:, 0])
        return -np.where(np.isnan(xap[:, 0]), fill, xap[:, 0])
    if spec["type"] == "logit":
        clf = LogisticRegression(C=1.0, max_iter=300)
    else:
        clf = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **spec["params"])
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(xtr, y_train)
    return pipe.predict_proba(xap)[:, 1]


def topk_recall(quarters, score, y, k):
    """Moi quy gan co top k% ngan hang diem cao nhat; recall gop = TP / tong so dong duong."""
    tp = 0
    for q in np.unique(quarters):
        m = quarters == q
        n = int(m.sum())
        kk = max(1, int(np.ceil(k * n)))
        order = np.argsort(-score[m], kind="stable")[:kk]
        tp += int(y[m][order].sum())
    return tp / max(int(y.sum()), 1)


def best_threshold(score, y, c_fp, c_fn_row):
    """Chon nguong giam thieu tong chi phi = c_fp * FP + c_fn_row * FN (gan co khi score >= nguong)."""
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order].astype(float)
    tp, fp = np.cumsum(yy), np.cumsum(1.0 - yy)
    p = yy.sum()
    cost = c_fp * fp + c_fn_row * (p - tp)
    cost_none = c_fn_row * p
    cost_all = c_fp * (len(s) - p)
    k = int(np.argmin(cost))
    if cost[k] >= cost_none:
        return {"threshold": float("inf"), "flagged_share": 0.0, "recall": 0.0, "precision": float("nan"),
                "cost": cost_none, "cost_flag_none": cost_none, "cost_flag_all": cost_all}
    return {"threshold": float(s[k]), "flagged_share": float((k + 1) / len(s)), "recall": float(tp[k] / p),
            "precision": float(tp[k] / (k + 1)), "cost": float(cost[k]), "cost_flag_none": float(cost_none),
            "cost_flag_all": float(cost_all)}


def calibration_table(score, y, qs=(0, .5, .8, .9, .95, .98, .99, .995, 1.0)):
    """Chia theo phan vi cua diem (nhieu phan vi o duoi cao vi du lieu rat mat can bang)."""
    edges = np.unique(np.quantile(score, qs))
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b + 1, "rows": int(m.sum()), "mean_predicted": float(score[m].mean()),
                         "observed_rate": float(y[m].mean())})
    return rows
