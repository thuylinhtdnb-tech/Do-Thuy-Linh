"""Do Polars tren CUNG may voi Spark (vd Colab 2 nhan) de so sanh thoi gian cong bang.

Chay tren Colab:
  !pip -q install polars
  !python 06b_polars_same_machine.py --panel panel.parquet --outdir out
Logic dac trung GIONG HET polars_features trong 05_polars_vs_pandas.py.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import polars as pl

KEYS = ["CERT", "REPDTE", "year"]
FEATS = ["eq_ratio", "npl_ratio", "reserve_to_npl", "loan_to_dep", "bro_to_dep", "log_asset",
         "asset_growth_yoy", "loan_growth_yoy", "d_npl_4q", "roa_mean4", "roa_std4", "eq_ratio_rel"]


def polars_features(d):
    c = pl.col
    d = d.sort(["CERT", "REPDTE"]).with_columns(
        (c("EQ") / c("ASSET") * 100).alias("eq_ratio"),
        pl.when(c("LNLSGR") > 0).then(c("NCLNLS") / c("LNLSGR") * 100).alias("npl_ratio"),
        pl.when(c("NCLNLS") > 0).then(c("LNATRES") / c("NCLNLS")).alias("reserve_to_npl"),
        pl.when(c("DEP") > 0).then(c("LNLSGR") / c("DEP")).alias("loan_to_dep"),
        pl.when(c("DEP") > 0).then(c("BRO") / c("DEP")).alias("bro_to_dep"),
        c("ASSET").log().alias("log_asset"),
    )
    ok4 = (c("qidx") - c("qidx").shift(4).over("CERT")) == 4
    lag_asset = c("ASSET").shift(4).over("CERT")
    lag_loans = c("LNLSGR").shift(4).over("CERT")
    d = d.with_columns(
        pl.when(ok4 & (lag_asset > 0)).then(c("ASSET") / lag_asset - 1).alias("asset_growth_yoy"),
        pl.when(ok4 & (lag_loans > 0)).then(c("LNLSGR") / lag_loans - 1).alias("loan_growth_yoy"),
        pl.when(ok4).then(c("npl_ratio") - c("npl_ratio").shift(4).over("CERT")).alias("d_npl_4q"),
        c("ROA").rolling_mean(window_size=4).over("CERT").alias("roa_mean4"),
        c("ROA").rolling_std(window_size=4).over("CERT").alias("roa_std4"),
        (c("eq_ratio") - c("eq_ratio").median().over("REPDTE")).alias("eq_ratio_rel"),
    )
    return d.select(KEYS + FEATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel.parquet")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    times = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        res = polars_features(pl.scan_parquet(a.panel)).collect()
        times.append(time.perf_counter() - t)
    row = ["polars_lazy_same_machine", round(times[0], 3), round(statistics.median(times), 3),
           round(min(times), 3), pl.__version__, pl.thread_pool_size(), res.height]
    with open(out / "polars_same_machine.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["engine", "first_run_s", "median_s", "min_s", "polars_version", "threads", "rows"])
        wr.writerow(row)
    print("engine, first_run_s, median_s, min_s, polars_version, threads, rows")
    print(row)


if __name__ == "__main__":
    main()
