"""Phan 5c: demo Spark MLlib (chay tren Colab hoac may co Java + pyspark). CHI dung dong DEV.

Chay:
  python 10_mllib_demo.py --data features_dev.parquet --outdir out
Huan luyen tren cac dong REPDTE <= 1999-03-31 va kiem dinh tren 2000-03-31 .. 2006-12-31
(CUNG fold 4 cua 08_model_dev.py, co khoang trong 4 quy). Khong dung tap test: file dau vao khong chua dong test.
Tien xu ly trong Spark (Imputer trung vi + VectorAssembler + StandardScaler) chi fit tren phan train;
khac sklearn o cho khong co winsorize.
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, functions as F

FEATURES = [
    "eq_ratio", "tier1_rbc", "total_rbc", "d_eq_ratio_4q", "eq_ratio_rel", "npl_ratio", "reserve_ratio",
    "reserve_to_npl", "texas_ratio", "d_npl_4q", "npl_ratio_rel", "roa", "nim", "roa_mean4", "roa_std4", "roa_rel",
    "efficiency_ratio", "int_exp_ratio", "loan_to_dep", "bro_to_dep", "cash_ratio", "securities_ratio", "dep_ratio",
    "asset_growth_yoy", "loan_growth_yoy", "dep_growth_yoy", "log_asset", "size_pct"]
TRAIN_END = date(1999, 3, 31)           # = bat dau validation (2000-03-31) - 4 quy
VAL_START, VAL_END = date(2000, 3, 31), date(2006, 12, 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="features_dev.parquet")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder.master("local[*]").appName("bft65b-mllib")
             .config("spark.sql.shuffle.partitions", "8").config("spark.driver.memory", "4g")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = (spark.read.parquet(a.data).select("REPDTE", F.col("y_fail_4q").cast("double").alias("label"), *FEATURES))
    assert df.agg(F.max("REPDTE")).first()[0] <= VAL_END, "file chua dong sau 2006: khong duoc dung o day"
    train = df.filter(F.col("REPDTE") <= F.lit(TRAIN_END)).cache()
    val = df.filter((F.col("REPDTE") >= F.lit(VAL_START)) & (F.col("REPDTE") <= F.lit(VAL_END))).cache()
    n_tr, n_va = train.count(), val.count()
    pos_tr, pos_va = train.filter("label = 1").count(), val.filter("label = 1").count()
    print(f"train {n_tr:,} dong ({pos_tr:,} duong) | validation {n_va:,} dong ({pos_va:,} duong)")

    prep = [
        Imputer(strategy="median", inputCols=FEATURES, outputCols=[f + "_imp" for f in FEATURES]),
        VectorAssembler(inputCols=[f + "_imp" for f in FEATURES], outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
    ]
    models = {
        "mllib_logistic_regression": LogisticRegression(featuresCol="features", labelCol="label",
                                                        maxIter=100, regParam=1e-4, elasticNetParam=0.0),
        "mllib_gbt": GBTClassifier(featuresCol="features", labelCol="label", maxIter=30, maxDepth=4,
                                   stepSize=0.1, maxBins=32, seed=42),
    }
    rows = []
    for name, clf in models.items():
        t = time.perf_counter()
        fitted = Pipeline(stages=prep + [clf]).fit(train)          # moi tham so (trung vi, trung binh, he so) chi tu train
        pred = fitted.transform(val)
        pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                           metricName="areaUnderPR").evaluate(pred)
        roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderROC").evaluate(pred)
        sec = time.perf_counter() - t
        rows.append([name, round(pr, 4), round(roc, 4), n_tr, n_va, pos_va, round(sec, 1), spark.version,
                     spark.sparkContext.defaultParallelism])
        print(f"{name}: PR-AUC {pr:.4f}  ROC-AUC {roc:.4f}  ({sec:.0f}s)")

    with open(out / "mllib_demo.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "pr_auc", "roc_auc", "n_train", "n_val", "positive_val_rows", "seconds",
                     "spark_version", "default_parallelism"])
        wr.writerows(rows)
    spark.stop()


if __name__ == "__main__":
    main()
