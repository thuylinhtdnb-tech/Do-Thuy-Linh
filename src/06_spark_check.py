"""Phan 3b: tai hien pipeline dac trung bang PySpark, doi chieu voi Polars, doc execution plan.

Chay (local co Java 17+ va pyspark, hoac Google Colab):
  python src/06_spark_check.py --panel data/curated/panel.parquet \
      --checksums reports/feature_checksums_polars.csv --sample reports/feature_sample_polars.csv
Can (tao boi 05_polars_vs_pandas.py): feature_checksums_polars.csv, feature_sample_polars.csv
Dau ra: spark_explain.txt, spark_vs_polars.csv, spark_timing.csv (trong --outdir)
"""
import argparse
import contextlib
import csv
import io
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

NEEDED = ["CERT", "REPDTE", "year", "qidx", "ASSET", "DEP", "EQ", "ROA", "NCLNLS", "LNATRES",
          "LNLSGR", "BRO", "y_fail_4q", "usable"]
KEYS = ["CERT", "REPDTE", "year"]
FEATS = ["eq_ratio", "npl_ratio", "reserve_to_npl", "loan_to_dep", "bro_to_dep", "log_asset",
         "asset_growth_yoy", "loan_growth_yoy", "d_npl_4q", "roa_mean4", "roa_std4", "eq_ratio_rel"]


def spark_features(df):
    """Cung logic voi polars_features / pandas_features trong 05_polars_vs_pandas.py."""
    c = F.col
    d = (df.select(*NEEDED)
         .withColumn("eq_ratio", c("EQ") / c("ASSET") * 100)
         .withColumn("npl_ratio", F.when(c("LNLSGR") > 0, c("NCLNLS") / c("LNLSGR") * 100))
         .withColumn("reserve_to_npl", F.when(c("NCLNLS") > 0, c("LNATRES") / c("NCLNLS")))
         .withColumn("loan_to_dep", F.when(c("DEP") > 0, c("LNLSGR") / c("DEP")))
         .withColumn("bro_to_dep", F.when(c("DEP") > 0, c("BRO") / c("DEP")))
         .withColumn("log_asset", F.log(c("ASSET"))))
    w = Window.partitionBy("CERT").orderBy("REPDTE")
    w4 = w.rowsBetween(-3, 0)
    ok4 = (c("qidx") - F.lag("qidx", 4).over(w)) == 4
    lag_asset = F.lag("ASSET", 4).over(w)
    lag_loans = F.lag("LNLSGR", 4).over(w)
    d = (d.withColumn("asset_growth_yoy", F.when(ok4 & (lag_asset > 0), c("ASSET") / lag_asset - 1))
         .withColumn("loan_growth_yoy", F.when(ok4 & (lag_loans > 0), c("LNLSGR") / lag_loans - 1))
         .withColumn("d_npl_4q", F.when(ok4, c("npl_ratio") - F.lag("npl_ratio", 4).over(w)))
         .withColumn("roa_mean4", F.when(F.count("ROA").over(w4) == 4, F.avg("ROA").over(w4)))
         .withColumn("roa_std4", F.when(F.count("ROA").over(w4) == 4, F.stddev_samp("ROA").over(w4)))
         .withColumn("eq_ratio_rel", c("eq_ratio") - F.median("eq_ratio").over(Window.partitionBy("REPDTE"))))
    return d.select(*KEYS, *FEATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/curated/panel.parquet")
    ap.add_argument("--checksums", default="reports/feature_checksums_polars.csv")
    ap.add_argument("--sample", default="reports/feature_sample_polars.csv")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder.master("local[*]").appName("bft65b")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.ansi.enabled", "false")
             .config("spark.driver.memory", "4g")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    def build():
        ft = spark_features(spark.read.parquet(a.panel))
        exprs = [F.count(F.lit(1)).alias("rows")]
        for f in FEATS:
            exprs += [F.count(f).alias(f"{f}__n"), F.sum(f).alias(f"{f}__sum")]
        return ft, ft.groupBy("year").agg(*exprs).orderBy("year")

    feats, agg = build()

    # ---- Execution plan
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agg.explain("formatted")
    (out / "spark_explain.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())

    # ---- Thoi gian: lan dau (gom khoi dong JVM/JIT) + cac lan sau, MOI LAN DUNG DataFrame MOI.
    # (Chay lai action tren cung mot DataFrame co the dung lai file shuffle cua lan truoc -> thoi gian sai.)
    t = time.perf_counter()
    rows = [r.asDict() for r in agg.collect()]
    cold = time.perf_counter() - t
    times = []
    for _ in range(a.repeats):
        _, fresh = build()
        t = time.perf_counter()
        rows = [r.asDict() for r in fresh.collect()]
        times.append(time.perf_counter() - t)
    with open(out / "spark_timing.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["engine", "cold_first_run_s", "median_s", "min_s", "spark_version",
                     "default_parallelism", "shuffle_partitions"])
        wr.writerow(["pyspark_local", round(cold, 3), round(statistics.median(times), 3), round(min(times), 3),
                     spark.version, spark.sparkContext.defaultParallelism, 8])

    # ---- Doi chieu tong hop theo nam voi Polars
    pol = {int(r["year"]): r for r in csv.DictReader(open(a.checksums))}
    results = []
    max_rel, n_mismatch, year_missing = 0.0, 0, 0
    for r in rows:
        p = pol.get(int(r["year"]))
        if p is None:
            year_missing += 1
            continue
        if int(p["rows"]) != r["rows"]:
            n_mismatch += 1
        for f in FEATS:
            if int(p[f"{f}__n"]) != r[f"{f}__n"]:
                n_mismatch += 1
            ps = float(p[f"{f}__sum"]) if p[f"{f}__sum"] not in ("", None) else 0.0
            ss = r[f"{f}__sum"] or 0.0
            max_rel = max(max_rel, abs(ps - ss) / max(1.0, abs(ps)))
    results.append({"check": "years in Spark / Polars", "value": f"{len(rows)} / {len(pol)}"})
    results.append({"check": "years missing in Polars", "value": year_missing})
    results.append({"check": "count mismatches (rows and non-null counts)", "value": n_mismatch})
    results.append({"check": "max relative difference of yearly sums", "value": f"{max_rel:.3e}"})

    # ---- Doi chieu tung dong tren mau ngan hang (CERT % 50 == 0)
    try:
        import pandas as pd
        sp = feats.filter(F.col("CERT") % 50 == 0).select("CERT", "REPDTE", *FEATS).toPandas()
        sp["REPDTE"] = pd.to_datetime(sp["REPDTE"])
        po = pd.read_csv(a.sample, parse_dates=["REPDTE"])
        m = sp.merge(po, on=["CERT", "REPDTE"], suffixes=("_s", "_p"), how="outer", indicator=True)
        results.append({"check": "sample rows only in Spark / only in Polars",
                        "value": f"{(m['_merge'] == 'left_only').sum()} / {(m['_merge'] == 'right_only').sum()}"})
        m = m[m["_merge"] == "both"]
        results.append({"check": "sample rows compared", "value": len(m)})
        worst, nullmm = 0.0, 0
        for f in FEATS:
            x, y = m[f + "_s"], m[f + "_p"]
            nullmm += int((x.isna() != y.isna()).sum())
            both = x.notna() & y.notna()
            if both.any():
                d = ((x - y).abs() / pd.concat([x.abs(), pd.Series(1.0, index=x.index)], axis=1).max(axis=1))[both]
                worst = max(worst, float(d.max()))
        results.append({"check": "sample null-pattern mismatches", "value": nullmm})
        results.append({"check": "sample max relative difference", "value": f"{worst:.3e}"})
    except ImportError:
        results.append({"check": "row-level sample check", "value": "skipped (pandas not installed)"})

    ok = n_mismatch == 0 and max_rel < 1e-6
    results.append({"check": "OVERALL (counts equal and yearly sums within 1e-6)", "value": "MATCH" if ok else "CHECK"})
    with open(out / "spark_vs_polars.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["check", "value"])
        wr.writeheader()
        wr.writerows(results)
    for r in results:
        print(f"{r['check']}: {r['value']}")
    print(f"Spark median time: {statistics.median(times):.3f}s over {a.repeats} runs")
    spark.stop()


if __name__ == "__main__":
    main()
