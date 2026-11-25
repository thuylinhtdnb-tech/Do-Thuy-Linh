# BFT65B – Big-Data Pipeline for Bank Failure Early Warning (FDIC data)

**Author:** Do Thuy Linh – student ID 11230984 · Course BFT65B, National Economics University (NEU)
**Repository:** https://github.com/thuylinhtdnb-tech/Do-Thuy-Linh

A reproducible data pipeline on public FDIC data (Call Reports 1984Q1–2026Q2, failures, institutions): acquisition,
storage, data quality, a point-in-time bank-quarter panel, large-scale processing (Polars, pandas, PySpark),
leak-free features and a time-ordered validation of an early-warning model. The full discussion is in the report.

## Tested environment
Windows 10, Python 3.15.2, 8 logical CPUs, 17 GB RAM. About 3 GB of free disk and an internet connection for the
download (about 13 minutes). Java 17+ and PySpark are needed only for the three Spark scripts, which were run on
Google Colab (see below).

## Setup
Windows (cmd):
```
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
macOS / Linux: `python -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`
(the `.bat` file and the backslash paths in the commands below are Windows syntax; use `/` and the `python` commands directly).

## Run end to end
`run_all.bat` runs steps 1–5 and 7 below and the tests. Each step can also be run on its own.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python src\01_download.py` | Resumable download of Financials (one Parquet per quarter), Failures and Institutions; completeness check | ~13 min |
| 2 | `python src\02_storage.py` | Typed, sorted Parquet layouts and benchmark against CSV | ~3 min |
| 3 | `python src\03_quality.py` | Automated data-quality report (`reports/dq_report.md`) | ~1 min |
| 4 | `python src\04_build_panel.py` | Clean bank-quarter panel, failure events, labels, exit classification | ~2 min |
| 5 | `python src\05_polars_vs_pandas.py` | Polars (eager/lazy/streaming) against pandas: time, memory, scaling, equivalence | ~10 min (`--skip-scaling`: ~3 min) |
| 6 | Colab: `src/06_spark_check.py`, `src/06b_polars_same_machine.py` | PySpark reproduction, equivalence check, execution plan, same-machine timing | ~10 min |
| 7 | `python src\07_features.py` | 28 leak-free features, time split, train-only preprocessing, univariate checks | <1 min |
| – | `python -m pytest tests -q` | 7 leakage and label tests | ~10 s |
| 8 | `python src\08_model_dev.py` | Forward-chaining validation on development rows, model choice, cost-based threshold; writes `reports/frozen_config.json` | ~6 min |
| 9 | `python src\09_final_test.py` | Single evaluation on the test period (see the note below) | ~3 min |
| 10 | Colab: `src/10_mllib_demo.py` | Spark MLlib demonstration on development rows only | ~5 min |

**Colab steps (6, 10).** Open a new notebook, run
`!apt-get -qq update`, `!apt-get -qq install -y openjdk-17-jdk-headless`, `!pip -q install pyspark pandas pyarrow polars`,
upload the script and its inputs (`data/curated/panel.parquet`, `reports/feature_checksums_polars.csv`,
`reports/feature_sample_polars.csv` for step 6; `data/curated/features_dev.parquet` for step 10), run the script with
`--help` to see its arguments, and download the CSV and text files it writes to `reports/`.

**About step 9 (test period used once).** The model and threshold were frozen by step 8 (`reports/frozen_config.json`,
22 Nov 2026) before the test period was evaluated (`reports/TEST_EVALUATED.json`, 23 Nov 2026). `09_final_test.py` refuses
to run when that lock file exists. To reproduce the numbers in a fresh clone, run
`python src\09_final_test.py --accept-rerun`; the configuration is frozen and seeded, so the results are identical.
Do not commit the regenerated lock file.

## Reproducibility notes
- Fixed seed 42 (`config.py`); versions pinned in `requirements.txt`.
- Data are not in Git (`.gitignore`); `01_download.py` re-creates them. The reported results use the download of
  20 Nov 2026 (1,682,738 financial rows, 4,117 failures, 27,835 institutions). FDIC updates its data weekly, so a later
  download can differ slightly.
- Timings depend on the machine; the benchmark files in `reports/` come from the machine described above.

## Leakage controls
Features use only information available at each quarter-end (calendar lags, same-quarter medians); all preprocessing
(winsorising, imputation, scaling) is fitted on training rows only inside a scikit-learn `Pipeline`; validation is
forward-chaining with a four-quarter purge; the development / test split has a four-quarter gap; feature screening uses
development rows only. `tests/test_no_leakage.py` checks these properties.

## Repository layout
```
config.py                  paths, seed, field list
run_all.bat                runs steps 1-5, 7 and the tests (Windows)
requirements.txt           pinned dependencies
src/                       01_download ... 10_mllib_demo, features.py, modeling.py
tests/test_no_leakage.py   leakage and label tests
reports/                   all numbers, tables and figures quoted in the report
data/                      created at run time (not in Git)
```

## Key results (details in the report)
- Storage: sorted zstd Parquet is 2.7x smaller than CSV and 3-61x faster to read.
- Processing: Polars lazy runs the 12-feature pipeline 5.8x faster than pandas with identical results; Spark reproduces
  the results (counts equal, sums within 2e-14) but is about 12x slower on the same 2-core machine, so no cluster is
  needed for this data.
- Model (test period 2008Q1-2025Q2, 546 failed banks): gradient boosting PR-AUC 0.568, ROC-AUC 0.984; the cost-based
  threshold fixed on development data flags 2% of bank-quarters and catches 96.5% of failed banks, a median 297 days
  before failure.

## Citations and AI-tool disclosure
- Data: FDIC BankFind Suite API and field dictionary (https://api.fdic.gov/banks, https://api.fdic.gov/banks/docs/).
- Course material: `fdic_download_starter.py` and the exam brief for BFT65B (NEU); the download script was written
  independently and the starter was used only as a reference for the API workflow.
- Background sources: FDIC "Failures and Assistance Transactions" notes via FRED (https://fred.stlouisfed.org/graph/?g=11n96);
  a third-party reading of the FDIC history change codes (https://banksparency.com/banks/statistics/charters-and-mergers).
- Software: Polars, PyArrow, pandas, scikit-learn, PySpark (including MLlib), matplotlib, psutil, requests, tqdm, pytest.
- AI tools: Claude (Anthropic) was used to plan the project, draft the scripts (`src/01_download.py` to
  `src/10_mllib_demo.py`, `features.py`, `modeling.py`), the tests and the report text. The author ran all code (locally,
  and the PySpark parts on Google Colab) and reviewed the output before using it.


