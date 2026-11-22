"""Phan 4: bang dac trung khong ro ri + chia dev/gap/test + tien xu ly chi fit tren train.

Chay:  python src\07_features.py
Dau vao : data/curated/panel.parquet
Dau ra  : data/curated/features.parquet
          reports/feature_dictionary.csv, split_summary.csv, feature_missingness_dev.csv,
          feature_univariate_auc_dev.csv, preprocessor_fit_dev.csv

Quy tac nghiem ngat: moi thong ke mo ta/chon dac trung o day chi tinh tren tap DEV.
Tap TEST chi duoc transform (khong fit) de kiem tra khong co NaN; khong in thong ke nhan cua test.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from features import (FEATURE_COLS, FEATURE_META, INPUT_COLS, LABEL_COLS, add_split, build_features,
                      make_preprocessor)

PANEL = C.DATA / "curated" / "panel.parquet"
REP = C.REPORTS


def main():
    REP.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    lf = pl.scan_parquet(PANEL).select(INPUT_COLS + LABEL_COLS)
    feats = add_split(build_features(lf)).collect()
    print(f"1/5 Bang dac trung: {feats.height:,} dong x {len(FEATURE_COLS)} dac trung "
          f"({time.perf_counter() - t0:.1f}s)")
    feats.write_parquet(C.DATA / "curated" / "features.parquet", compression="zstd", row_group_size=100_000)
    pl.DataFrame(FEATURE_META, schema=["feature", "group", "formula", "rationale"], orient="row") \
        .write_csv(REP / "feature_dictionary.csv")

    # ---- Chia dev / gap / test
    rows = []
    for s in ("dev", "gap", "test"):
        d = feats.filter(pl.col("split") == s)
        u = d.filter(pl.col("usable"))
        pos = u.filter(pl.col("y_fail_4q") == 1)
        rows.append({"split": s, "first_quarter": str(d["REPDTE"].min()), "last_quarter": str(d["REPDTE"].max()),
                     "rows": d.height, "usable_rows": u.height, "positive_rows": pos.height,
                     "failed_banks": pos["CERT"].n_unique(),
                     "positive_rate_pct": round(100 * pos.height / max(u.height, 1), 4)})
    summ = pl.DataFrame(rows)
    summ.write_csv(REP / "split_summary.csv")
    print("2/5 Chia thoi gian:")
    print(summ)
    n_test_fail = summ.filter(pl.col("split") == "test")["failed_banks"][0]
    assert n_test_fail >= 50, f"test chi co {n_test_fail} ngan hang that bai (<50)"

    dev_u = feats.filter((pl.col("split") == "dev") & pl.col("usable"))
    y = dev_u["y_fail_4q"].to_numpy()

    # ---- Thieu du lieu va AUC don bien: CHI tren dev
    miss = pl.DataFrame({
        "feature": FEATURE_COLS,
        "null_rate_dev": [dev_u[f].null_count() / dev_u.height for f in FEATURE_COLS]})
    miss.write_csv(REP / "feature_missingness_dev.csv")
    auc_rows = []
    for name, group, *_ in FEATURE_META:
        x = dev_u[name].to_numpy()
        ok = ~np.isnan(x)
        auc = roc_auc_score(y[ok], x[ok])
        auc_rows.append({"feature": name, "group": group, "n_non_null": int(ok.sum()),
                         "auc_dev": round(auc, 4), "auc_folded": round(max(auc, 1 - auc), 4),
                         "direction": "higher = riskier" if auc > 0.5 else "lower = riskier"})
    auc_df = pl.DataFrame(auc_rows).sort("auc_folded", descending=True)
    auc_df.write_csv(REP / "feature_univariate_auc_dev.csv")
    print("3/5 AUC don bien tren DEV (top 10):")
    print(auc_df.head(10))

    # ---- Pipeline tien xu ly: fit CHI tren dev
    Xd = dev_u.select(FEATURE_COLS).to_numpy()
    pipe = make_preprocessor().fit(Xd)
    Zd = pipe.transform(Xd)
    te_u = feats.filter((pl.col("split") == "test") & pl.col("usable"))
    Zt = pipe.transform(te_u.select(FEATURE_COLS).to_numpy())          # chi transform, khong fit
    assert not np.isnan(Zd).any() and not np.isnan(Zt).any()
    n = len(FEATURE_COLS)
    ind = getattr(pipe.named_steps["impute"].indicator_, "features_", np.array([], dtype=int))
    fit = pl.DataFrame({
        "feature": FEATURE_COLS,
        "winsor_low": pipe.named_steps["winsor"].lo_,
        "winsor_high": pipe.named_steps["winsor"].hi_,
        "median_used_for_imputation": pipe.named_steps["impute"].statistics_,
        "scale_mean": pipe.named_steps["scale"].mean_[:n],
        "scale_std": pipe.named_steps["scale"].scale_[:n],
        "has_missing_indicator": [i in set(ind.tolist()) for i in range(n)]})
    fit.write_csv(REP / "preprocessor_fit_dev.csv")
    print(f"4/5 Tien xu ly fit tren {Xd.shape[0]:,} dong dev: {n} dac trung + {len(ind)} co bao thieu "
          f"-> ma tran {Zd.shape[1]} cot; test transform {Zt.shape[0]:,} dong, khong NaN.")
    print(f"5/5 Xong ({time.perf_counter() - t0:.1f}s). Xem reports/ va data/curated/features.parquet")


if __name__ == "__main__":
    main()
