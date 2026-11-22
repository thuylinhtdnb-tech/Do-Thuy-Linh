"""Thu vien dac trung cho he thong canh bao som (Phan 4).

  * build_features : tinh 28 dac trung tu bang panel; CHI dung thong tin tai REPDTE q hoac truoc do
                     (lag/rolling theo tung ngan hang; trung vi cung quy chi dung du lieu cung quy).
  * add_split      : chia dev / gap / test theo thoi gian.
  * make_preprocessor : Pipeline sklearn (winsorize -> impute trung vi + co bao thieu -> chuan hoa);
                     moi tham so CHI hoc tu tap train khi goi .fit().
"""
from datetime import date

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- Chia thoi gian: dev <= 2006Q4 | gap = 4 quy REPDTE cua nam 2007 | test >= 2008Q1.
# Cua so nhan cua dong cuoi dev (REPDTE 2006-12-31) ket thuc 2007-12-31 < dong dau test (2008-03-31).
DEV_END = date(2006, 12, 31)
GAP_END = date(2007, 12, 31)

KEYS = ["CERT", "REPDTE", "year"]
INPUT_COLS = ["CERT", "REPDTE", "year", "qidx", "ASSET", "DEP", "EQ", "ROA", "NCLNLS", "LNATRES", "LNLSGR",
              "RBC1AAJ", "RBCRWAJ", "NIMY", "INTINC", "EINTEXP", "NONII", "NONIX", "BRO", "CHBAL", "SC", "LIAB"]
LABEL_COLS = ["y_fail_4q", "y_fail_4q_strict", "usable", "label_complete"]

# (ten, nhom CAMELS, cong thuc, ly do kinh te)
FEATURE_META = [
    ("eq_ratio", "Capital", "EQ / ASSET x 100", "Loss-absorbing capital; thin capital is the most direct failure signal."),
    ("tier1_rbc", "Capital", "RBC1AAJ (Tier 1 risk-based capital ratio, %)", "Regulatory capital relative to risk-weighted assets."),
    ("total_rbc", "Capital", "RBCRWAJ (total risk-based capital ratio, %)", "Broader regulatory capital cushion."),
    ("d_eq_ratio_4q", "Capital", "eq_ratio - eq_ratio 4 quarters earlier", "Capital erosion over the last year."),
    ("eq_ratio_rel", "Capital", "eq_ratio - median eq_ratio of all banks in the same quarter", "Peer-relative capital; removes macro and time effects."),
    ("npl_ratio", "Asset quality", "NCLNLS / LNLSGR x 100", "Share of loans that are non-performing; the classic leading indicator."),
    ("reserve_ratio", "Asset quality", "LNATRES / LNLSGR x 100", "Provisioning relative to the loan book."),
    ("reserve_to_npl", "Asset quality", "LNATRES / NCLNLS", "Coverage of bad loans by reserves."),
    ("texas_ratio", "Asset quality", "NCLNLS / (EQ + LNATRES)", "Share of loss-absorbing resources already tied to bad loans (equity used instead of tangible equity)."),
    ("d_npl_4q", "Asset quality", "npl_ratio - npl_ratio 4 quarters earlier", "Deterioration speed of loan quality."),
    ("npl_ratio_rel", "Asset quality", "npl_ratio - median npl_ratio in the same quarter", "Peer-relative loan quality."),
    ("roa", "Earnings", "ROA (%)", "Profitability; losses deplete capital."),
    ("nim", "Earnings", "NIMY (net interest margin, %)", "Core earnings power."),
    ("roa_mean4", "Earnings", "mean ROA over last 4 reported quarters", "Smoothed profitability, less noisy than one quarter."),
    ("roa_std4", "Earnings", "std ROA over last 4 reported quarters", "Earnings volatility."),
    ("roa_rel", "Earnings", "roa - median roa in the same quarter", "Peer-relative profitability."),
    ("efficiency_ratio", "Earnings", "NONIX / (INTINC - EINTEXP + NONII)", "Cost to income; high values mean little margin for losses."),
    ("int_exp_ratio", "Earnings", "EINTEXP x (4 / quarter number) / LIAB x 100", "Annualised funding cost; high cost signals risky funding."),
    ("loan_to_dep", "Liquidity", "LNLSGR / DEP", "Loans funded by non-deposit sources are riskier."),
    ("bro_to_dep", "Liquidity", "BRO / DEP", "Brokered deposits are runnable, rate-sensitive funding."),
    ("cash_ratio", "Liquidity", "CHBAL / ASSET x 100", "Immediate liquidity buffer."),
    ("securities_ratio", "Liquidity", "SC / ASSET x 100", "Liquid asset buffer."),
    ("dep_ratio", "Liquidity", "DEP / ASSET x 100", "Reliance on deposit funding."),
    ("asset_growth_yoy", "Growth", "ASSET / ASSET 4 quarters earlier - 1", "Rapid growth often precedes credit problems."),
    ("loan_growth_yoy", "Growth", "LNLSGR / LNLSGR 4 quarters earlier - 1", "Aggressive lending is a known risk factor."),
    ("dep_growth_yoy", "Growth", "DEP / DEP 4 quarters earlier - 1", "Deposit outflows signal loss of confidence."),
    ("log_asset", "Size", "ln(ASSET)", "Size affects diversification and resolution choices."),
    ("size_pct", "Size", "rank of ASSET among all banks in the same quarter / number of banks", "Peer-relative size."),
]
FEATURE_COLS = [m[0] for m in FEATURE_META]


def build_features(d):
    """d: LazyFrame hoac DataFrame cua panel (cot INPUT_COLS + LABEL_COLS). Chi dung qua khu va cung quy."""
    c = pl.col
    ann = 4.0 / (c("REPDTE").dt.month().cast(pl.Float64) / 3.0)   # luong tich luy trong nam -> quy ra nam

    d = d.sort(["CERT", "REPDTE"]).with_columns(
        (c("EQ") / c("ASSET") * 100).alias("eq_ratio"),
        c("RBC1AAJ").alias("tier1_rbc"),
        c("RBCRWAJ").alias("total_rbc"),
        pl.when(c("LNLSGR") > 0).then(c("NCLNLS") / c("LNLSGR") * 100).alias("npl_ratio"),
        pl.when(c("LNLSGR") > 0).then(c("LNATRES") / c("LNLSGR") * 100).alias("reserve_ratio"),
        pl.when(c("NCLNLS") > 0).then(c("LNATRES") / c("NCLNLS")).alias("reserve_to_npl"),
        pl.when((c("EQ") + c("LNATRES")) > 0).then(c("NCLNLS") / (c("EQ") + c("LNATRES"))).alias("texas_ratio"),
        c("ROA").alias("roa"),
        c("NIMY").alias("nim"),
        pl.when((c("INTINC") - c("EINTEXP") + c("NONII")) > 0)
        .then(c("NONIX") / (c("INTINC") - c("EINTEXP") + c("NONII"))).alias("efficiency_ratio"),
        pl.when(c("LIAB") > 0).then(c("EINTEXP") * ann / c("LIAB") * 100).alias("int_exp_ratio"),
        pl.when(c("DEP") > 0).then(c("LNLSGR") / c("DEP")).alias("loan_to_dep"),
        pl.when(c("DEP") > 0).then(c("BRO") / c("DEP")).alias("bro_to_dep"),
        (c("CHBAL") / c("ASSET") * 100).alias("cash_ratio"),
        (c("SC") / c("ASSET") * 100).alias("securities_ratio"),
        (c("DEP") / c("ASSET") * 100).alias("dep_ratio"),
        c("ASSET").log().alias("log_asset"),
    )

    ok4 = (c("qidx") - c("qidx").shift(4).over("CERT")) == 4      # lag dung 4 quy theo lich, khong theo vi tri dong

    def lag4(col):
        return c(col).shift(4).over("CERT")

    d = d.with_columns(
        pl.when(ok4 & (lag4("ASSET") > 0)).then(c("ASSET") / lag4("ASSET") - 1).alias("asset_growth_yoy"),
        pl.when(ok4 & (lag4("LNLSGR") > 0)).then(c("LNLSGR") / lag4("LNLSGR") - 1).alias("loan_growth_yoy"),
        pl.when(ok4 & (lag4("DEP") > 0)).then(c("DEP") / lag4("DEP") - 1).alias("dep_growth_yoy"),
        pl.when(ok4).then(c("eq_ratio") - lag4("eq_ratio")).alias("d_eq_ratio_4q"),
        pl.when(ok4).then(c("npl_ratio") - lag4("npl_ratio")).alias("d_npl_4q"),
        c("ROA").rolling_mean(window_size=4).over("CERT").alias("roa_mean4"),
        c("ROA").rolling_std(window_size=4).over("CERT").alias("roa_std4"),
        (c("eq_ratio") - c("eq_ratio").median().over("REPDTE")).alias("eq_ratio_rel"),
        (c("npl_ratio") - c("npl_ratio").median().over("REPDTE")).alias("npl_ratio_rel"),
        (c("roa") - c("roa").median().over("REPDTE")).alias("roa_rel"),
        (c("ASSET").rank().over("REPDTE") / c("ASSET").count().over("REPDTE")).alias("size_pct"),
    )
    d = d.with_columns([
        pl.when(c(f).is_infinite() | c(f).is_nan()).then(None).otherwise(c(f)).alias(f) for f in FEATURE_COLS])
    return d.select(KEYS + FEATURE_COLS + LABEL_COLS)


def add_split(d):
    c = pl.col
    return d.with_columns(
        pl.when(c("REPDTE") <= DEV_END).then(pl.lit("dev"))
        .when(c("REPDTE") <= GAP_END).then(pl.lit("gap"))
        .otherwise(pl.lit("test")).alias("split"))


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cat ngoai lai theo phan vi; cac nguong CHI hoc tu du lieu train trong fit()."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)   # NaN giu nguyen


def make_preprocessor(lower=0.005, upper=0.995):
    return Pipeline([
        ("winsor", Winsorizer(lower, upper)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
