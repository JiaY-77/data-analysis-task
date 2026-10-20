#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""risk_flag.py —— 派工前缺料风险标记（可直接接进调度流程）

规则（基于 dispatch_orders.csv 的回顾性验证）：
    风险 = (库存同步滞后 > 0.5 小时) 或 (网点排队 > 70 单)

输出：out/high_risk_orders.csv —— 需要人工二次确认的派工单清单
"""
import argparse
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LAG_H = 0.5   # 小时
QUEUE_N = 70  # 单
FLAG_COST_H = 20.0
FAIL_COST_H = 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/dispatch_orders.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--lag", type=float, default=LAG_H)
    ap.add_argument("--queue", type=float, default=QUEUE_N)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.path, encoding="utf-8-sig")
    df["dispatch_time_utc"] = pd.to_datetime(df["dispatch_time"], utc=True)
    df["stock_synced_at_utc"] = pd.to_datetime(df["stock_synced_at"], utc=True)
    df["lag_hours"] = (df["dispatch_time_utc"] - df["stock_synced_at_utc"]).dt.total_seconds() / 3600
    df["queue_length"] = pd.to_numeric(df["queue_length"], errors="coerce")
    df["available_stock"] = pd.to_numeric(df["available_stock"], errors="coerce")
    df["job_completed"] = pd.to_numeric(df["job_completed"], errors="coerce")

    df["lag_risk"] = df["lag_hours"] > a.lag
    df["queue_risk"] = df["queue_length"] > a.queue
    df["risk_flag"] = (df["lag_risk"] | df["queue_risk"]).map({True: "HIGH", False: ""})
    df["risk_reason"] = df.apply(
        lambda r: "；".join(
            [x for x in [
                f"库存同步滞后 {r['lag_hours']:.2f}h > {a.lag}h" if r["lag_risk"] else "",
                f"排队 {int(r['queue_length'])} > {a.queue}" if r["queue_risk"] else "",
            ] if x]
        ), axis=1)

    flagged = df[df["risk_flag"] == "HIGH"].copy()
    cols = ["dispatch_id", "dispatch_time", "service_hub", "part_category",
            "available_stock", "queue_length", "lag_hours", "risk_reason", "job_completed"]
    out = os.path.join(a.outdir, "high_risk_orders.csv")
    flagged[cols].sort_values(["lag_hours", "queue_length"], ascending=False).to_csv(
        out, index=False, encoding="utf-8-sig")

    valid = df[df["job_completed"].notna()]
    n_all, n_flag = len(valid), int((valid["risk_flag"] == "HIGH").sum())
    n_fail = int((valid["job_completed"] == 0).sum())
    caught = int(((valid["risk_flag"] == "HIGH") & (valid["job_completed"] == 0)).sum())
    cost = n_flag * FLAG_COST_H
    saved = caught * FAIL_COST_H

    print("=" * 64)
    print("派工前风险标记（回顾性验证）")
    print("=" * 64)
    print(f"  规则：库存同步滞后 > {a.lag}h 或 排队 > {a.queue} 单")
    print(f"  已回传工单 {n_all} · 标记 {n_flag} 单（{n_flag/n_all:.1%}）")
    print(f"  其中真缺料 {caught} 单 → 精确率 {caught/n_flag:.1%}，覆盖全部缺料 {caught/n_fail:.1%}")
    print(f"  标记成本 {cost:,.0f} h ｜ 挽回 {saved:,.0f} h ｜ 净节省 {saved-cost:+,.0f} h/周")
    print(f"  → 清单已写入 {out}（{len(flagged)} 行）")
    print(f"\n  业务动作：清单上的派工单，发车前要求库管/现场二次确认实物；确认不了就改派或顺延。")


if __name__ == "__main__":
    main()
