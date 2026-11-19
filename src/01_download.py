"""Tai du lieu FDIC (Financials theo quy, Failures, Institutions) - resumable.

Vi du:
  python src/01_download.py --start 2020Q1 --end 2020Q4      # chay thu
  python src/01_download.py                                   # toan bo 1984Q1-2026Q2
  python src/01_download.py --probe                           # kiem tra truong
"""
import argparse, json, sys, time, platform
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

PAGE = 10000
QEND = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}


def session():
    s = requests.Session()
    r = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


def quarters(start, end):
    y, q = int(start[:4]), int(start[-1])
    ye, qe = int(end[:4]), int(end[-1])
    while (y, q) <= (ye, qe):
        yield f"{y}Q{q}", f"{y}{QEND[q]}"
        q += 1
        if q == 5:
            y, q = y + 1, 1


def fetch_all(s, endpoint, filters=None, fields=None, sort_by=None):
    """Phan trang qua limit/offset. Tra ve (rows, total bao boi API)."""
    rows, offset, total = [], 0, None
    while True:
        params = {"limit": PAGE, "offset": offset, "format": "json"}
        if filters:
            params["filters"] = filters
        if fields:
            params["fields"] = ",".join(fields)
        if sort_by:
            params["sort_by"] = sort_by
            params["sort_order"] = "ASC"
        resp = s.get(f"{C.API}/{endpoint}", params=params, timeout=120)
        resp.raise_for_status()
        js = resp.json()
        total = js["meta"]["total"]
        batch = [d["data"] for d in js["data"]]
        rows.extend(batch)
        offset += PAGE
        if not batch or offset >= total:
            break
    return rows, total


def load_manifest(path):
    return json.loads(path.read_text()) if path.exists() else {}


def probe(s):
    """Kiem tra truong nao khong ton tai hoac null o ca 2 moc thoi gian."""
    for repdte in ["19901231", "20231231"]:
        rows, _ = fetch_all(s, "financials", f"REPDTE:{repdte}", C.FIN_FIELDS)
        df = pl.DataFrame(rows, infer_schema_length=None)
        missing = [f for f in C.FIN_FIELDS if f not in df.columns]
        allnull = [c for c in df.columns if df[c].null_count() == df.height]
        print(f"[{repdte}] rows={df.height} truong_thieu={missing} null_toan_bo={allnull}")


def download_financials(s, start, end):
    C.RAW_FIN.mkdir(parents=True, exist_ok=True)
    mpath = C.RAW / "manifest_financials.json"
    man = load_manifest(mpath)
    for qname, repdte in tqdm(list(quarters(start, end)), desc="financials"):
        f = C.RAW_FIN / f"REPDTE={repdte}.parquet"
        if man.get(repdte, {}).get("status") == "done" and f.exists():
            continue
        t0 = time.time()
        rows, total = fetch_all(s, "financials", f"REPDTE:{repdte}", C.FIN_FIELDS, "CERT")
        if rows:
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(f, compression="zstd")
        man[repdte] = {
            "quarter": qname, "rows": len(rows), "api_total": total,
            "status": "done" if len(rows) == total else "MISMATCH",
            "seconds": round(time.time() - t0, 1),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        mpath.write_text(json.dumps(man, indent=2))  # ghi sau moi quy -> tai tiep duoc


def download_table(s, endpoint, sort_by):
    """Failures / Institutions: lay tat ca truong."""
    rows, total = fetch_all(s, endpoint, sort_by=sort_by)
    out = C.RAW / f"{endpoint}.parquet"
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out, compression="zstd")
    print(f"{endpoint}: {len(rows)} dong (API bao {total}) -> {out}")
    return {"rows": len(rows), "api_total": total}


def completeness_check():
    files = sorted(C.RAW_FIN.glob("REPDTE=*.parquet"))
    df = pl.scan_parquet(files).select(["CERT", "REPDTE"]).collect()
    dup = df.height - df.unique().height
    nq = df["REPDTE"].n_unique()
    print(f"Tong dong financials={df.height:,} | so quy={nq} | trung (CERT,REPDTE)={dup}")
    return {"fin_rows": df.height, "n_quarters": nq, "duplicates": dup}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.START_Q)
    ap.add_argument("--end", default=C.END_Q)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-tables", action="store_true")
    a = ap.parse_args()
    C.RAW.mkdir(parents=True, exist_ok=True)
    s = session()
    if a.probe:
        probe(s); sys.exit()
    download_financials(s, a.start, a.end)
    meta = {"method": "FDIC public API (api.fdic.gov/banks), limit/offset paging",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "range": [a.start, a.end]}
    if not a.skip_tables:
        meta["failures"] = download_table(s, "failures", "FAILDATE")
        meta["institutions"] = download_table(s, "institutions", "CERT")
    meta["completeness"] = completeness_check()
    (C.RAW / "download_log.json").write_text(json.dumps(meta, indent=2))
    bad = [k for k, v in load_manifest(C.RAW / "manifest_financials.json").items() if v["status"] != "done"]
    print("Quy loi (MISMATCH):", bad or "khong co")
