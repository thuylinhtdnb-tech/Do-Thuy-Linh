"""Phan 1: kiem tra day du + thiet ke bo cuc luu tru + benchmark so voi CSV.

Chay:  python src\02_storage.py
Dau ra: data/curated/..., reports/storage_benchmark.csv, reports/storage_sizes.csv,
        reports/quarter_counts.csv, reports/machine.json, reports/figures/banks_per_quarter.png
"""
import gc, json, os, platform, statistics, sys, time
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CUR = C.DATA / "curated"
FIG = C.REPORTS / "figures"
for p in (CUR, FIG):
    p.mkdir(parents=True, exist_ok=True)
C.REPORTS.mkdir(exist_ok=True)

QEND = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def expected_quarters():
    y, q = int(C.START_Q[:4]), int(C.START_Q[-1])
    ye, qe = int(C.END_Q[:4]), int(C.END_Q[-1])
    out = []
    while (y, q) <= (ye, qe):
        out.append(date(y, *QEND[q]))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def load_raw():
    """Doc tung file quy, ep kieu ro rang: CERT Int32, REPDTE Date, con lai Float64."""
    frames = []
    for f in sorted(C.RAW_FIN.glob("REPDTE=*.parquet")):
        d = f.stem.split("=")[1]
        df = pl.read_parquet(f)
        others = [c for c in df.columns if c not in ("CERT", "REPDTE")]
        df = df.with_columns(
            pl.col("CERT").cast(pl.Int32),
            pl.lit(d).str.to_date("%Y%m%d").alias("REPDTE"),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in others],
        ).select(["CERT", "REPDTE"] + others)
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed").sort(["REPDTE", "CERT"])


def dir_size_mb(p: Path):
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def bench(fn, n=3):
    times = []
    for _ in range(n):
        gc.collect()
        t = time.perf_counter()
        r = fn()
        if isinstance(r, pl.LazyFrame):
            r = r.collect()
        _ = r.height
        times.append(time.perf_counter() - t)
        del r
    return statistics.median(times)


def main():
    print("1/5 Doc va ep kieu du lieu tho ...")
    df = load_raw()
    print(f"    {df.height:,} dong, {df.width} cot")

    # ---- Kiem tra day du ----
    exp = expected_quarters()
    cnt = df.group_by("REPDTE").len().sort("REPDTE").rename({"len": "n_banks"})
    cnt.write_csv(C.REPORTS / "quarter_counts.csv")
    have = set(cnt["REPDTE"].to_list())
    missing = [str(d) for d in exp if d not in have]
    dups = df.height - df.select(["CERT", "REPDTE"]).unique().height
    print(f"    quy mong doi={len(exp)} | co={len(have)} | thieu={missing or 'khong'} | trung={dups}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(cnt["REPDTE"].to_list(), cnt["n_banks"].to_list())
        ax.set_title("Number of reporting banks per quarter (FDIC Call Reports)")
        ax.set_ylabel("Banks")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(FIG / "banks_per_quarter.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("    (bo qua bieu do: pip install matplotlib)")

    # ---- Ghi cac bo cuc ----
    print("2/5 Ghi cac bo cuc luu tru ...")
    csv_p = CUR / "financials.csv"
    pq_snappy = CUR / "financials_snappy.parquet"
    pq_zstd = CUR / "financials_zstd.parquet"
    part_dir = CUR / "financials_by_year"
    if part_dir.exists():
        import shutil
        shutil.rmtree(part_dir)

    df.write_csv(csv_p)
    df.write_parquet(pq_snappy, compression="snappy", row_group_size=100_000, statistics=True)
    df.write_parquet(pq_zstd, compression="zstd", compression_level=3, row_group_size=100_000, statistics=True)
    dfy = df.with_columns(pl.col("REPDTE").dt.year().alias("year"))
    for y in sorted(dfy["year"].unique().to_list()):
        d = part_dir / f"year={y}"
        d.mkdir(parents=True, exist_ok=True)
        dfy.filter(pl.col("year") == y).drop("year").write_parquet(d / "part-0.parquet", compression="zstd")

    # bang tong hop nho
    for name in ("failures", "institutions"):
        pl.read_parquet(C.RAW / f"{name}.parquet").write_parquet(CUR / f"{name}.parquet", compression="zstd")

    layouts = {
        "csv": csv_p, "parquet_snappy": pq_snappy,
        "parquet_zstd": pq_zstd, "parquet_zstd_by_year": part_dir,
    }
    raw_dir_mb = dir_size_mb(C.RAW_FIN)
    sizes = [{"layout": k, "size_mb": round(dir_size_mb(v), 1)} for k, v in layouts.items()]
    sizes.append({"layout": "raw_per_quarter_parquet(download)", "size_mb": round(raw_dir_mb, 1)})
    sz = pl.DataFrame(sizes)
    csv_mb = sz.filter(pl.col("layout") == "csv")["size_mb"][0]
    sz = sz.with_columns((csv_mb / pl.col("size_mb")).round(2).alias("compression_vs_csv"))
    sz.write_csv(C.REPORTS / "storage_sizes.csv")
    print(sz)

    # ---- Benchmark ----
    print("3/5 Benchmark (median cua 3 lan) ...")
    five = ["CERT", "REPDTE", "ASSET", "EQ", "NCLNLS"]
    glob = str(part_dir / "year=*" / "*.parquet")
    d0, d1 = date(2008, 1, 1), date(2008, 12, 31)
    rd_csv = lambda cols=None: pl.read_csv(csv_p, columns=cols, try_parse_dates=True)
    tests = {
        "full_read_all_columns": {
            "csv": lambda: rd_csv(),
            "parquet_snappy": lambda: pl.read_parquet(pq_snappy),
            "parquet_zstd": lambda: pl.read_parquet(pq_zstd),
            "parquet_zstd_by_year": lambda: pl.read_parquet(glob, hive_partitioning=True),
        },
        "read_5_columns": {
            "csv": lambda: rd_csv(five),
            "parquet_snappy": lambda: pl.read_parquet(pq_snappy, columns=five),
            "parquet_zstd": lambda: pl.read_parquet(pq_zstd, columns=five),
            "parquet_zstd_by_year": lambda: pl.scan_parquet(glob, hive_partitioning=True).select(five).collect(),
        },
        "filter_year_2008": {
            "csv": lambda: rd_csv().filter(pl.col("REPDTE").is_between(d0, d1)),
            "parquet_snappy": lambda: pl.scan_parquet(pq_snappy).filter(pl.col("REPDTE").is_between(d0, d1)),
            "parquet_zstd": lambda: pl.scan_parquet(pq_zstd).filter(pl.col("REPDTE").is_between(d0, d1)),
            "parquet_zstd_by_year": lambda: pl.scan_parquet(glob, hive_partitioning=True).filter(pl.col("year") == 2008),
        },
    }
    rows = []
    for tname, impls in tests.items():
        for lname, fn in impls.items():
            t = bench(fn)
            rows.append({"test": tname, "layout": lname, "seconds_median": round(t, 3)})
            print(f"    {tname:24s} {lname:22s} {t:7.3f}s")
    res = pl.DataFrame(rows)
    csv_t = res.filter(pl.col("layout") == "csv").select(["test", pl.col("seconds_median").alias("csv_s")])
    res = res.join(csv_t, on="test").with_columns((pl.col("csv_s") / pl.col("seconds_median")).round(2).alias("speedup_vs_csv")).drop("csv_s")
    res.write_csv(C.REPORTS / "storage_benchmark.csv")

    # ---- May ----
    print("4/5 Ghi thong tin may ...")
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": platform.python_version(), "polars": pl.__version__}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = "pip install psutil"
    (C.REPORTS / "machine.json").write_text(json.dumps(info, indent=2))
    print(info)
    print("5/5 Xong. Xem thu muc reports/")


if __name__ == "__main__":
    main()
