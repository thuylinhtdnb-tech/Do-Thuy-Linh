"""Cau hinh trung tam cho toan bo pipeline."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_FIN = RAW / "financials"
REPORTS = ROOT / "reports"

API = "https://api.fdic.gov/banks"
START_Q = "1984Q1"
END_Q = "2026Q2"

# Truong tai chinh (theo khung CAMELS). BAT BUOC doi chieu voi tu dien truong:
# https://api.fdic.gov/banks/docs/  -- script se canh bao truong nao khong ton tai / null toan bo.
FIN_FIELDS = [
    "CERT", "REPDTE", "ASSET", "DEP", "EQ", "NETINC", "ROA", "ROE",
    "NCLNLS", "LNATRES", "LNLSNET", "LNLSGR", "RBC1AAJ", "RBCRWAJ",
    "NIMY", "INTINC", "EINTEXP", "NONII", "NONIX", "BRO", "CHBAL",
    "SC", "LIAB", "NUMEMP",
]
