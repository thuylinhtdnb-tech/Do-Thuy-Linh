"""Kiem thu chong ro ri du lieu (Phan 4). Chay: python -m pytest tests -q   (can pip install pytest)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from features import (DEV_END, FEATURE_COLS, GAP_END, INPUT_COLS, KEYS, LABEL_COLS,  # noqa: E402
                      add_split, build_features, make_preprocessor)

PANEL = C.DATA / "curated" / "panel.parquet"
FORBIDDEN = {"fail_date", "fail_restype", "days_to_fail", "y_fail_4q", "y_fail_4q_strict", "fail_lt_avail",
             "ambig_4q", "label_complete", "usable", "ACTIVE", "CHANGEC1", "split"}


@pytest.fixture(scope="module")
def sample():
    return (pl.scan_parquet(PANEL).select(INPUT_COLS + LABEL_COLS)
            .filter(pl.col("CERT") % 25 == 0).collect())


@pytest.mark.parametrize("cut", [date(1995, 12, 31), date(2005, 12, 31), date(2015, 12, 31)])
def test_features_at_q_do_not_depend_on_the_future(sample, cut):
    """Tinh dac trung tren du lieu bi cat tai 'cut' phai cho CUNG ket qua nhu tren du lieu day du (<= cut)."""
    full = build_features(sample).filter(pl.col("REPDTE") <= cut).sort(["CERT", "REPDTE"])
    past = build_features(sample.filter(pl.col("REPDTE") <= cut)).sort(["CERT", "REPDTE"])
    assert full.height == past.height
    for f in FEATURE_COLS:
        np.testing.assert_allclose(full[f].to_numpy(), past[f].to_numpy(), rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=f)


def test_no_label_or_future_columns_are_used_as_inputs():
    assert not FORBIDDEN & set(FEATURE_COLS)
    assert not FORBIDDEN & set(INPUT_COLS)
    assert not set(KEYS) & set(FEATURE_COLS)


def test_preprocessor_parameters_come_from_train_only(sample):
    feats = add_split(build_features(sample)).filter(pl.col("usable"))
    xd = feats.filter(pl.col("split") == "dev").select(FEATURE_COLS).to_numpy()
    xt = feats.filter(pl.col("split") == "test").select(FEATURE_COLS).to_numpy()
    pipe = make_preprocessor().fit(xd)
    lo, hi = pipe.named_steps["winsor"].lo_.copy(), pipe.named_steps["winsor"].hi_.copy()
    med = pipe.named_steps["impute"].statistics_.copy()
    np.testing.assert_allclose(lo, np.nanquantile(xd, 0.005, axis=0))     # = thong ke cua train
    pipe.transform(xt)                                                    # transform test khong doi tham so
    np.testing.assert_array_equal(lo, pipe.named_steps["winsor"].lo_)
    np.testing.assert_array_equal(hi, pipe.named_steps["winsor"].hi_)
    np.testing.assert_array_equal(med, pipe.named_steps["impute"].statistics_)
    both = np.vstack([xd, xt])                                            # neu fit ca test thi nguong se khac
    assert not np.allclose(lo, np.nanquantile(both, 0.005, axis=0), equal_nan=True)


def test_label_matches_the_four_quarter_window():
    df = pl.scan_parquet(PANEL).select(["REPDTE", "fail_date", "y_fail_4q"]).collect()
    pos = df.filter(pl.col("y_fail_4q") == 1)
    assert pos.height > 0
    assert (pos["fail_date"] > pos["REPDTE"]).all()
    assert (pos["fail_date"] <= pos["REPDTE"].dt.offset_by("1y")).all()
    neg = df.filter((pl.col("y_fail_4q") == 0) & pl.col("fail_date").is_not_null())
    assert (neg["fail_date"] > neg["REPDTE"].dt.offset_by("1y")).all()    # ke ca dong sau that bai da bi bo


def test_dev_label_windows_do_not_reach_the_test_period():
    q = pl.scan_parquet(PANEL).select("REPDTE").collect()["REPDTE"]
    dev_max = q.filter(q <= DEV_END).max()
    test_min = q.filter(q > GAP_END).min()
    assert dev_max.replace(year=dev_max.year + 1) <= test_min              # cua so nhan cua dev ket thuc truoc test
