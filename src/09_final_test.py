"""Phan 5b - DANH GIA DUNG MOT LAN tren TEST.

Chay:  python src\09_final_test.py       (CHI chay sau khi 08_model_dev.py xong va da doc ket qua dev)
Doc reports/frozen_config.json (mo hinh + nguong chon tren DEV). Fit lai tren toan bo DEV, du doan TEST,
ap nguong da dong bang, KHONG chinh lai gi. Ghi reports/TEST_EVALUATED.json de chan chay lai:
neu xoa file de chay lai thi la vi pham quy tac "danh gia mot lan" va phai khai bao trong bao cao.
Cac mo hinh co so (luat von thap, logistic 3 bien, logistic 28 bien) cung duoc danh gia trong CUNG mot lan nay.
"""
import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from features import FEATURE_COLS
from modeling import calibration_table, fit_predict, model_specs, topk_recall

warnings.filterwarnings("ignore")
REP = C.REPORTS
CUR = C.DATA / "curated"
LOCK = REP / "TEST_EVALUATED.json"
CRISIS_END = date(2013, 12, 31)


def rank_metrics(y, s, quarters, prob=True):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "brier": brier_score_loss(y, s) if prob else float("nan"),
            "recall_top1pct_per_quarter": topk_recall(quarters, s, y, 0.01),
            "recall_top5pct_per_quarter": topk_recall(quarters, s, y, 0.05)}


def main():
    if LOCK.exists() and "--accept-rerun" not in sys.argv:
        print(f"DUNG LAI: {LOCK.name} da ton tai. Tap test da duoc danh gia mot lan "
              f"({json.loads(LOCK.read_text())['evaluated_at']}). Khong chay lai.")
        sys.exit(1)
    cfg = json.loads((REP / "frozen_config.json").read_text(encoding="utf-8"))
    chosen, thr = cfg["chosen_model"], cfg["threshold"]
    c_fn, c_fp = cfg["c_fn_thousand_usd"], cfg["c_fp_thousand_usd"]
    print(f"Mo hinh dong bang: {chosen}; nguong {thr:.5f}; c_FN {c_fn:,.0f}; c_FP {c_fp:,.1f} (nghin USD)")

    feats = pl.read_parquet(CUR / "features.parquet").filter(pl.col("usable"))
    dev = feats.filter(pl.col("split") == "dev")
    test = feats.filter(pl.col("split") == "test")
    panel = pl.read_parquet(CUR / "panel.parquet", columns=["CERT", "REPDTE", "days_to_fail"])
    bank = pl.read_parquet(CUR / "bank_dim.parquet").select(["CERT", "cost_k"])
    test = (test.join(panel, on=["CERT", "REPDTE"], how="left").join(bank, on="CERT", how="left"))
    xd, yd = dev.select(FEATURE_COLS).to_numpy(), dev["y_fail_4q"].to_numpy().astype(int)
    xt, yt = test.select(FEATURE_COLS).to_numpy(), test["y_fail_4q"].to_numpy().astype(int)
    yst = test["y_fail_4q_strict"].to_numpy().astype(int)
    q = test["REPDTE"].to_numpy()
    print(f"Dev {dev.height:,} dong | Test {test.height:,} dong, {int(yt.sum()):,} dong duong, "
          f"{test.filter(pl.col('y_fail_4q') == 1)['CERT'].n_unique()} ngan hang that bai")

    # ---- Fit tren DEV, du doan TEST (moi mo hinh mot lan)
    specs = model_specs()
    specs[chosen] = cfg["spec"]
    names = list(dict.fromkeys(["baseline_rule_low_capital", "baseline_logit_3", "logit_28", chosen]))
    scores = {n: fit_predict(specs[n], xd, yd, xt) for n in names}
    comp = []
    for n in names:
        r = {"model": n + (" (chosen)" if n == chosen else "")}
        r.update(rank_metrics(yt, scores[n], q, prob=specs[n]["type"] != "rule"))
        r["pr_auc_strict_label"] = average_precision_score(yst, scores[n])
        comp.append(r)
    comp_df = pl.DataFrame(comp)
    comp_df.write_csv(REP / "test_model_comparison.csv")
    print("\nSo sanh mo hinh tren TEST:")
    print(comp_df)

    # ---- Ket qua tai nguong dong bang
    s = scores[chosen]
    flag = s >= thr
    tp, fp = int((flag & (yt == 1)).sum()), int((flag & (yt == 0)).sum())
    fn = int(((~flag) & (yt == 1)).sum())
    outcome = {"threshold": thr, "flagged_rows": int(flag.sum()), "flagged_share_pct": 100 * flag.mean(),
               "true_positive_rows": tp, "false_positive_rows": fp, "false_negative_rows": fn,
               "precision": tp / max(tp + fp, 1), "recall_rows": tp / max(tp + fn, 1)}
    td = test.with_columns(pl.Series("flag", flag), pl.Series("score", s))
    pos = td.filter(pl.col("y_fail_4q") == 1)
    banks = pos.group_by("CERT").agg(pl.col("flag").any().alias("caught"),
                                     pl.col("days_to_fail").filter(pl.col("flag")).max().alias("lead_days"))
    outcome["failed_banks"] = banks.height
    outcome["failed_banks_flagged_at_least_once"] = int(banks["caught"].sum())
    outcome["share_failed_banks_flagged_pct"] = 100 * banks["caught"].mean()
    outcome["median_lead_days_of_first_flag"] = banks["lead_days"].median()
    pl.DataFrame([outcome]).write_csv(REP / "test_threshold_outcome.csv")
    print("\nTai nguong dong bang:")
    print(pl.DataFrame([outcome]).transpose(include_header=True))

    # ---- Cac lat cat bao cao (cung mot bo du doan)
    slices = []
    for label, m in (("all test", np.ones(len(yt), bool)), ("2008-2013 (crisis)", q <= np.datetime64(CRISIS_END)),
                     ("2014-2025 (after crisis)", q > np.datetime64(CRISIS_END))):
        ym = yt[m]
        row = {"slice": label, "rows": int(m.sum()), "positive_rows": int(ym.sum()),
               "flagged_share_pct": 100 * flag[m].mean()}
        if 0 < ym.sum() < len(ym):
            row.update({"pr_auc": average_precision_score(ym, s[m]), "roc_auc": roc_auc_score(ym, s[m]),
                        "recall_rows": flag[m][ym == 1].mean(), "false_alarm_rate_pct": 100 * flag[m][ym == 0].mean()})
        slices.append(row)
    pl.DataFrame(slices, infer_schema_length=None).write_csv(REP / "test_slices.csv")
    print(pl.DataFrame(slices, infer_schema_length=None))

    # ---- Phat hien theo thoi gian con lai den luc that bai
    days = pos["days_to_fail"].to_numpy()
    bucket = np.clip(np.ceil(days / 91.3125), 1, 4).astype(int)
    lead = [{"quarters_before_failure": b, "positive_rows": int((bucket == b).sum()),
             "flagged_pct": 100 * float(pos["flag"].to_numpy()[bucket == b].mean())} for b in (1, 2, 3, 4)]
    pl.DataFrame(lead).write_csv(REP / "test_lead_time.csv")
    print(pl.DataFrame(lead))

    # ---- Chi phi thuc te (nghin USD) so voi khong lam gi / bao dong tat ca
    cost_row = np.where(np.isnan(td["cost_k"].to_numpy()), c_fn, td["cost_k"].to_numpy()) / 4
    fn_cost = float(cost_row[(~flag) & (yt == 1)].sum())
    cost_model = c_fp * fp + fn_cost
    cost_none = float(cost_row[yt == 1].sum())
    cost_all = c_fp * float((yt == 0).sum())
    cost_df = pl.DataFrame({"policy": ["model at frozen threshold", "flag nobody", "flag every bank-quarter"],
                            "cost_million_usd": [cost_model / 1000, cost_none / 1000, cost_all / 1000]})
    cost_df.write_csv(REP / "test_cost.csv")
    print(cost_df)

    pl.DataFrame(calibration_table(s, yt)).write_csv(REP / "test_calibration.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for n in names:
            p, r, _ = precision_recall_curve(yt, scores[n])
            ax.plot(r, p, label=n)
        ax.axhline(yt.mean(), color="grey", ls=":", label="no skill")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall on the test period")
        ax.legend(fontsize=7)
        fig.tight_layout()
        (REP / "figures").mkdir(exist_ok=True)
        fig.savefig(REP / "figures" / "test_pr_curve.png", dpi=200)
    except ImportError:
        pass

    LOCK.write_text(json.dumps({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                                "chosen_model": chosen, "threshold": thr,
                                "pr_auc_chosen": float(average_precision_score(yt, s))}, indent=2), encoding="utf-8")
    print("\nDanh gia test xong. Da ghi reports/TEST_EVALUATED.json (khong chay lai lenh nay).")


if __name__ == "__main__":
    main()
