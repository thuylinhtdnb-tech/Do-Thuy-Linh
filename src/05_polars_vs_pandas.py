"""Phan 3a: Polars (eager / lazy / streaming) vs pandas tren CUNG mot pipeline dac trung.

Do: thoi gian (median 3 lan), RAM dinh (moi engine chay trong tien trinh rieng), mo rong theo so luong CPU
va theo so dong (nhan ban du lieu), doi chieu ket qua pandas vs Polars, ke hoach thuc thi cua Polars.

Chay:  python src\05_polars_vs_pandas.py                  (day du, ~5-15 phut)
       python src\05_polars_vs_pandas.py --skip-scaling   (nhanh hon, bo qua thi nghiem nhan ban du lieu)
Can:   pip install pandas psutil
Dau ra (reports/): benchmark_engines.csv, thread_scaling.csv, memory_scaling.csv, scaling_summary.md,
       polars_explain.txt, equivalence_pandas_vs_polars.csv, feature_checksums_polars.csv, feature_sample_polars.csv
"""
import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

PANEL = C.DATA / "curated" / "panel.parquet"
REP = C.REPORTS
OFFSET = 1_000_000  # dich CERT khi nhan ban du lieu de cac ban sao khong trung ngan hang

NEEDED = ["CERT", "REPDTE", "year", "qidx", "ASSET", "DEP", "EQ", "ROA", "NCLNLS", "LNATRES",
          "LNLSGR", "BRO", "y_fail_4q", "usable"]
KEYS = ["CERT", "REPDTE", "year"]
FEATS = ["eq_ratio", "npl_ratio", "reserve_to_npl", "loan_to_dep", "bro_to_dep", "log_asset",
         "asset_growth_yoy", "loan_growth_yoy", "d_npl_4q", "roa_mean4", "roa_std4", "eq_ratio_rel"]


# ---------------------------------------------------------------- Pipeline (cung logic o ca 3 cong cu)
def polars_features(d):
    """d la LazyFrame hoac DataFrame; cung mot ham cho eager va lazy."""
    c = pl.col
    d = d.sort(["CERT", "REPDTE"]).with_columns(
        (c("EQ") / c("ASSET") * 100).alias("eq_ratio"),
        pl.when(c("LNLSGR") > 0).then(c("NCLNLS") / c("LNLSGR") * 100).alias("npl_ratio"),
        pl.when(c("NCLNLS") > 0).then(c("LNATRES") / c("NCLNLS")).alias("reserve_to_npl"),
        pl.when(c("DEP") > 0).then(c("LNLSGR") / c("DEP")).alias("loan_to_dep"),
        pl.when(c("DEP") > 0).then(c("BRO") / c("DEP")).alias("bro_to_dep"),
        c("ASSET").log().alias("log_asset"),
    )
    ok4 = (c("qidx") - c("qidx").shift(4).over("CERT")) == 4          # lag 4 quy dung theo lich
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


def pandas_features(df):
    import numpy as np
    df = df.sort_values(["CERT", "REPDTE"], kind="stable").reset_index(drop=True)
    df["eq_ratio"] = df["EQ"] / df["ASSET"] * 100
    df["npl_ratio"] = (df["NCLNLS"] / df["LNLSGR"] * 100).where(df["LNLSGR"] > 0)
    df["reserve_to_npl"] = (df["LNATRES"] / df["NCLNLS"]).where(df["NCLNLS"] > 0)
    df["loan_to_dep"] = (df["LNLSGR"] / df["DEP"]).where(df["DEP"] > 0)
    df["bro_to_dep"] = (df["BRO"] / df["DEP"]).where(df["DEP"] > 0)
    df["log_asset"] = np.log(df["ASSET"])
    g = df.groupby("CERT", sort=False)
    ok4 = (df["qidx"] - g["qidx"].shift(4)) == 4
    lag_asset = g["ASSET"].shift(4)
    lag_loans = g["LNLSGR"].shift(4)
    df["asset_growth_yoy"] = (df["ASSET"] / lag_asset - 1).where(ok4 & (lag_asset > 0))
    df["loan_growth_yoy"] = (df["LNLSGR"] / lag_loans - 1).where(ok4 & (lag_loans > 0))
    df["d_npl_4q"] = (df["npl_ratio"] - g["npl_ratio"].shift(4)).where(ok4)
    roll = g["ROA"].rolling(4)
    df["roa_mean4"] = roll.mean().reset_index(level=0, drop=True)
    df["roa_std4"] = roll.std().reset_index(level=0, drop=True)
    df["eq_ratio_rel"] = df["eq_ratio"] - df.groupby("REPDTE")["eq_ratio"].transform("median")
    return df[KEYS + FEATS]


# ---------------------------------------------------------------- Cac cach thuc thi
def exec_pandas(scale):
    import pandas as pd
    df = pd.read_parquet(PANEL, columns=NEEDED)
    if scale > 1:
        df = pd.concat([df.assign(CERT=df["CERT"] + i * OFFSET) for i in range(scale)], ignore_index=True)
    return pandas_features(df)


def exec_polars_eager(scale, all_cols=False):
    df = pl.read_parquet(PANEL) if all_cols else pl.read_parquet(PANEL, columns=NEEDED)
    if scale > 1:
        df = pl.concat([df.with_columns(pl.col("CERT") + i * OFFSET) for i in range(scale)])
    return polars_features(df)


def exec_polars_lazy(scale, streaming=False):
    lf = pl.scan_parquet(PANEL)
    if scale > 1:
        lf = pl.concat([lf.with_columns(pl.col("CERT") + i * OFFSET) for i in range(scale)])
    q = polars_features(lf)
    return q.collect(engine="streaming") if streaming else q.collect()


ENGINES = {
    "pandas": lambda s: exec_pandas(s),
    "polars_eager_all_cols": lambda s: exec_polars_eager(s, all_cols=True),
    "polars_eager": lambda s: exec_polars_eager(s),
    "polars_lazy": lambda s: exec_polars_lazy(s),
    "polars_streaming": lambda s: exec_polars_lazy(s, streaming=True),
}


def peak_mb():
    import psutil
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):                 # Windows
        return mi.peak_wset / 1e6
    try:
        import resource                          # Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return mi.rss / 1e6


# ---------------------------------------------------------------- Worker (1 engine = 1 tien trinh)
def run_worker(engine, repeats, scale, out):
    base = peak_mb()
    times, res = [], None
    for r in range(repeats):
        res = None
        gc.collect()
        t = time.perf_counter()
        res = ENGINES[engine](scale)
        times.append(time.perf_counter() - t)
        rows = len(res)
        if r < repeats - 1:
            del res
    peak = peak_mb()
    if out:
        if hasattr(res, "write_parquet"):
            res.write_parquet(out)
        else:
            res.to_parquet(out)
    print("RESULT " + json.dumps({
        "engine": engine, "scale": scale, "rows": rows, "repeats": repeats,
        "median_s": round(statistics.median(times), 3), "min_s": round(min(times), 3),
        "peak_mb": round(peak, 1), "baseline_mb": round(base, 1),
        "threads": pl.thread_pool_size()}))


def spawn(engine, repeats, scale, out=None, env_extra=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", engine,
           "--repeats", str(repeats), "--scale", str(scale)]
    if out:
        cmd += ["--out", str(out)]
    env = os.environ.copy()
    env.update(env_extra or {})
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"engine": engine, "scale": scale, "error": (p.stderr or p.stdout)[-500:]}


# ---------------------------------------------------------------- Doi chieu pandas vs Polars
def compare_outputs(a_path, b_path):
    a = pl.read_parquet(a_path).with_columns(pl.col("REPDTE").cast(pl.Date))
    b = pl.read_parquet(b_path).with_columns(pl.col("REPDTE").cast(pl.Date))
    j = a.join(b, on=["CERT", "REPDTE"], how="inner", suffix="_pd")
    rows = [{"feature": "(row count polars / pandas / joined)", "n_both_non_null": a.height,
             "null_mismatch": b.height, "max_abs_diff": float(j.height), "max_rel_diff": None}]
    for f in FEATS:
        x, y = pl.col(f).fill_nan(None), pl.col(f + "_pd").fill_nan(None)
        both = x.is_not_null() & y.is_not_null()
        r = j.select(
            both.sum().alias("n"),
            (x.is_null() != y.is_null()).sum().alias("mm"),
            (x - y).abs().filter(both).max().alias("mad"),
            ((x - y).abs() / pl.max_horizontal(x.abs(), pl.lit(1.0))).filter(both).max().alias("mrd"),
        ).row(0)
        rows.append({"feature": f, "n_both_non_null": r[0], "null_mismatch": r[1],
                     "max_abs_diff": r[2], "max_rel_diff": r[3]})
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--skip-scaling", action="store_true")
    a = ap.parse_args()
    if a.worker:
        run_worker(a.worker, a.repeats, a.scale, a.out)
        return

    REP.mkdir(exist_ok=True)
    import psutil
    ram_mb = psutil.virtual_memory().total / 1e6

    # 1. Ke hoach thuc thi Polars
    print("1/6 Ke hoach thuc thi Polars ...")
    q = polars_features(pl.scan_parquet(PANEL))
    txt = "== NON-OPTIMIZED PLAN ==\n" + q.explain(optimized=False) + "\n\n== OPTIMIZED PLAN ==\n" + q.explain()
    try:
        txt += "\n\n== STREAMING ENGINE PLAN ==\n" + q.explain(engine="streaming")
    except Exception as e:  # noqa
        txt += f"\n\n(streaming plan not available: {e})"
    (REP / "polars_explain.txt").write_text(txt, encoding="utf-8")

    # 2. So sanh engine
    print("2/6 So sanh engine (moi engine 1 tien trinh rieng, median cua 3 lan) ...")
    res = []
    for eng in ENGINES:
        out = C.DATA / "curated" / f"bench_{eng}.parquet" if eng in ("pandas", "polars_lazy") else None
        r = spawn(eng, 3, 1, out)
        res.append(r)
        print("   ", r)

    # 3. Mo rong theo so luong CPU
    print("3/6 Mo rong theo so luong CPU (polars_lazy) ...")
    th = []
    for n in (1, 2, 4, 8):
        r = spawn("polars_lazy", 3, 1, env_extra={"POLARS_MAX_THREADS": str(n)})
        th.append(r)
        print("   ", r)

    # 4. Mo rong theo so dong (nhan ban du lieu)
    mem = [r for r in res if r["engine"] in ("pandas", "polars_lazy", "polars_streaming") and "error" not in r]
    if not a.skip_scaling:
        print("4/6 Mo rong theo so dong (nhan ban 2x, 4x) ...")
        for eng in ("pandas", "polars_lazy", "polars_streaming"):
            for s in (2, 4):
                r = spawn(eng, 1, s)
                print("   ", r)
                if "error" not in r:
                    mem.append(r)

    pl.DataFrame(res, infer_schema_length=None).write_csv(REP / "benchmark_engines.csv")
    pl.DataFrame(th, infer_schema_length=None).write_csv(REP / "thread_scaling.csv")
    if mem:
        pl.DataFrame(mem, infer_schema_length=None).write_csv(REP / "memory_scaling.csv")

    # 5. Doi chieu
    print("5/6 Doi chieu pandas vs Polars ...")
    pa, pp = C.DATA / "curated" / "bench_polars_lazy.parquet", C.DATA / "curated" / "bench_pandas.parquet"
    if pa.exists() and pp.exists():
        eq = compare_outputs(pa, pp)
        eq.write_csv(REP / "equivalence_pandas_vs_polars.csv")
        print(eq)
    # checksum + mau cho buoc PySpark
    if pa.exists():
        a_df = pl.read_parquet(pa)
        aggs = [pl.len().alias("rows")]
        for f in FEATS:
            aggs += [pl.col(f).count().alias(f"{f}__n"), pl.col(f).sum().alias(f"{f}__sum")]
        a_df.group_by("year").agg(aggs).sort("year").write_csv(REP / "feature_checksums_polars.csv")
        a_df.filter(pl.col("CERT") % 50 == 0).sort(["CERT", "REPDTE"]).write_csv(REP / "feature_sample_polars.csv")

    # 6. Ngoai suy
    print("6/6 Ngoai suy khi nao het RAM ...")
    import numpy as np
    lines = [f"# Scaling summary (machine RAM {ram_mb / 1000:.1f} GB)\n",
             "| engine | points (scale) | MB per million rows | seconds per million rows | rows at 80% of RAM (millions) |",
             "|---|---|---|---|---|"]
    for eng in ("pandas", "polars_lazy", "polars_streaming"):
        pts = sorted([(r["rows"], r["peak_mb"], r["median_s"]) for r in mem if r["engine"] == eng])
        if len(pts) >= 2:
            x = np.array([p[0] / 1e6 for p in pts])
            b, a0 = np.polyfit(x, [p[1] for p in pts], 1)
            t, _ = np.polyfit(x, [p[2] for p in pts], 1)
            lines.append(f"| {eng} | {len(pts)} | {b:,.0f} | {t:,.2f} | {(0.8 * ram_mb - a0) / b:,.1f} |")
    (REP / "scaling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    show = pl.DataFrame(res, infer_schema_length=None).select(
        ["engine", "rows", "median_s", "min_s", "peak_mb", "baseline_mb"]) if res else None
    print(show)
    print("Xong. Xem thu muc reports/")


if __name__ == "__main__":
    main()
