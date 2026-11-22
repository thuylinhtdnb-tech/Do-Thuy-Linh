"""Phan 5a - CHI DUNG DEV: so sanh mo hinh, kiem dinh thoi gian, chon sieu tham so, dat nguong theo chi phi.

Chay:  python src\08_model_dev.py        (~5-15 phut)
KHONG danh gia tren tap TEST (chi doc cac dong dev: split == "dev" va usable).
Dau ra: reports/fold_metrics_dev.csv, model_selection_dev.csv, threshold_dev.csv, calibration_dev.csv,
        frozen_config.json (mo hinh + nguong da DONG BANG), data/curated/features_dev.parquet (cho demo MLlib)

Kiem dinh: forward chaining (cua so mo rong) voi KHOANG TRONG (purge) 4 quy giua train va validation:
train gom cac dong REPDTE <= bat dau validation - 4 quy, nen cua so nhan cua train khong cham validation.
"""
import json
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from features import FEATURE_COLS
from modeling import SEED, best_threshold, calibration_table, fit_predict, model_specs, topk_recall

warnings.filterwarnings("ignore")
REP = C.REPORTS
CUR = C.DATA / "curated"

FOLDS = [  # (bat dau validation, ket thuc validation)
    (date(1989, 3, 31), date(1990, 12, 31)),
    (date(1991, 3, 31), date(1992, 12, 31)),
    (date(1993, 3, 31), date(1999, 12, 31)),
    (date(2000, 3, 31), date(2006, 12, 31)),
]
FP_FRACTION = 0.005                          # GIA DINH: chi phi bao dong gia = 0.5% chi phi trung binh cua mot vu that bai
SENSITIVITY = [0.001, 0.0025, 0.005, 0.01, 0.02]


def main():
    REP.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    dev = (pl.scan_parquet(CUR / "features.parquet")
           .filter((pl.col("split") == "dev") & pl.col("usable"))
           .select(["CERT", "REPDTE", "year"] + FEATURE_COLS + ["y_fail_4q"]).collect())
    dev.write_parquet(CUR / "features_dev.parquet", compression="zstd")        # chi chua dong dev (cho Spark MLlib)
    X = dev.select(FEATURE_COLS).to_numpy()
    y = dev["y_fail_4q"].to_numpy().astype(int)
    dates = dev["REPDTE"].to_numpy()
    print(f"Dev usable: {dev.height:,} dong, {int(y.sum()):,} dong duong ({100 * y.mean():.3f}%)")

    specs = model_specs()
    rows, preds = [], {}
    for fi, (vs, ve) in enumerate(FOLDS, 1):
        tr = dates <= np.datetime64(date(vs.year - 1, vs.month, vs.day))      # purge 4 quy
        va = (dates >= np.datetime64(vs)) & (dates <= np.datetime64(ve))
        print(f"Fold {fi}: train <= {date(vs.year - 1, vs.month, vs.day)} ({int(tr.sum()):,} dong, {int(y[tr].sum()):,} duong) | "
              f"validation {vs} .. {ve} ({int(va.sum()):,} dong, {int(y[va].sum()):,} duong)")
        for name, spec in specs.items():
            t = time.perf_counter()
            s = fit_predict(spec, X[tr], y[tr], X[va])
            preds.setdefault(name, []).append((np.where(va)[0], s))
            prob = spec["type"] != "rule"
            rows.append({"model": name, "fold": fi, "n_train": int(tr.sum()), "n_val": int(va.sum()),
                         "pos_val": int(y[va].sum()),
                         "pr_auc": average_precision_score(y[va], s), "roc_auc": roc_auc_score(y[va], s),
                         "brier": brier_score_loss(y[va], s) if prob else float("nan"),
                         "recall_top1pct": topk_recall(dates[va], s, y[va], 0.01),
                         "recall_top5pct": topk_recall(dates[va], s, y[va], 0.05),
                         "seconds": round(time.perf_counter() - t, 1)})
            print(f"   {name:28s} PR-AUC {rows[-1]['pr_auc']:.4f}  ROC-AUC {rows[-1]['roc_auc']:.4f}  "
                  f"({rows[-1]['seconds']}s)")
    fm = pl.DataFrame(rows)
    fm.write_csv(REP / "fold_metrics_dev.csv")
    sel = (fm.group_by("model").agg(pl.col("pr_auc").mean().alias("mean_pr_auc"),
                                    pl.col("roc_auc").mean().alias("mean_roc_auc"),
                                    pl.col("brier").mean().alias("mean_brier"),
                                    pl.col("recall_top1pct").mean().alias("mean_recall_top1pct"),
                                    pl.col("recall_top5pct").mean().alias("mean_recall_top5pct"))
           .sort("mean_pr_auc", descending=True))
    sel.write_csv(REP / "model_selection_dev.csv")
    print("\nTong hop forward-chaining tren DEV (trung binh 4 fold):")
    print(sel)
    chosen = sel["model"][0]

    # ---- Chi phi: that bai bo sot lay tu COST cua FDIC (chi cac vu o dev)
    bank = pl.read_parquet(CUR / "bank_dim.parquet").select(["CERT", "cost_k"])
    pos_banks = dev.filter(pl.col("y_fail_4q") == 1).select("CERT").unique()
    costs = pos_banks.join(bank, on="CERT", how="left")["cost_k"].drop_nulls()
    coverage = costs.len() / pos_banks.height
    c_fn = float(costs.mean())                      # nghin USD / vu that bai
    c_fn_row = c_fn / 4                             # moi vu co toi da 4 dong duong
    print(f"\nChi phi bo sot: trung binh {c_fn:,.0f} nghin USD / vu (COST co o {100 * coverage:.0f}% ngan hang that bai trong dev)")

    # ---- Nguong tren du doan out-of-fold cua mo hinh duoc chon
    idx = np.concatenate([p[0] for p in preds[chosen]])
    s_oof = np.concatenate([p[1] for p in preds[chosen]])
    y_oof = y[idx]
    thr_rows = []
    for f in SENSITIVITY:
        c_fp = f * c_fn
        r = best_threshold(s_oof, y_oof, c_fp, c_fn_row)
        r.update({"fp_fraction": f, "c_fp_thousand_usd": c_fp, "theoretical_p_star": c_fp / (c_fp + c_fn_row)})
        thr_rows.append(r)
    thr = pl.DataFrame(thr_rows).select(["fp_fraction", "c_fp_thousand_usd", "theoretical_p_star", "threshold",
                                         "flagged_share", "recall", "precision", "cost", "cost_flag_none",
                                         "cost_flag_all"])
    thr.write_csv(REP / "threshold_dev.csv")
    print("\nNguong theo chi phi tren du doan out-of-fold (DEV):")
    print(thr)
    base = next(r for r in thr_rows if r["fp_fraction"] == FP_FRACTION)
    pl.DataFrame(calibration_table(s_oof, y_oof)).write_csv(REP / "calibration_dev.csv")

    cfg = {"created_at": datetime.now().isoformat(timespec="seconds"), "chosen_model": chosen,
           "spec": specs[chosen], "mean_cv_pr_auc": float(sel["mean_pr_auc"][0]),
           "threshold": base["threshold"], "fp_fraction": FP_FRACTION,
           "c_fn_thousand_usd": c_fn, "c_fp_thousand_usd": FP_FRACTION * c_fn,
           "cost_coverage_dev": coverage, "dev_end": "2006-12-31", "seed": SEED,
           "folds": [[str(a), str(b)] for a, b in FOLDS]}
    (REP / "frozen_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nMo hinh duoc chon: {chosen}; nguong dong bang = {base['threshold']:.5f} "
          f"(flag {100 * base['flagged_share']:.2f}% dong, recall {100 * base['recall']:.1f}%)")
    print(f"Xong ({time.perf_counter() - t0:.0f}s). Da ghi reports/frozen_config.json. CHUA dung tap TEST.")


if __name__ == "__main__":
    main()
