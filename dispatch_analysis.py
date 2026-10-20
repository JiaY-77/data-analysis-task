#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dispatch_analysis.py —— 维保缺料根因 + 风险标记 ROI

业务目标：减少工程师因配件缺料导致的现场空跑，提升一次派工修复率。
口径（题目给）：每次到位缺料损失 120 小时；每标记一次缺料风险花费 20 小时。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAIL_H = 120.0
FLAG_H = 20.0


def sec(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.path, encoding="utf-8-sig", low_memory=False)
    df["dt"] = pd.to_datetime(df["dispatch_time"], utc=True, errors="coerce")
    df["sync"] = pd.to_datetime(df["stock_synced_at"], utc=True, errors="coerce")
    df["lag_h"] = (df["dt"] - df["sync"]).dt.total_seconds() / 3600
    df["stock"] = pd.to_numeric(df["available_stock"], errors="coerce")
    df["queue"] = pd.to_numeric(df["queue_length"], errors="coerce")
    df["done"] = pd.to_numeric(df["job_completed"], errors="coerce")

    valid = df[df["done"].notna()].copy()
    n_all, n_valid = len(df), len(valid)
    n_missing = n_all - n_valid
    n_fail = int((valid["done"] == 0).sum())
    n_ok = int((valid["done"] == 1).sum())
    base_loss = n_fail * FAIL_H

    sec("一、总览（分母 = job_completed 已回传）")
    print(f"  派工单 {n_all:,} · 已回传 {n_valid:,} · 未回传 {n_missing:,}（{n_missing/n_all:.1%}，不计入分母）")
    print(f"  履约成功 {n_ok:,} · 现场缺料 {n_fail:,}")
    print(f"  ▶ 一次派工修复率 {n_ok/n_valid:.2%} ｜ 缺料率 {n_fail/n_valid:.2%}")
    print(f"  ▶ 缺料工时损失 {base_loss:,.0f} 小时（{n_fail} × {FAIL_H:.0f}h）")
    print(f"  时间跨度 {valid['dt'].min():%Y-%m-%d} ~ {valid['dt'].max():%Y-%m-%d}")

    sec("二、根因：库存同步滞后 vs 缺料率")
    q = valid["lag_h"].describe(percentiles=[.5, .75, .9, .95, .99])
    print(f"  滞后(小时) 中位 {q['50%']:.2f} · P75 {q['75%']:.2f} · P90 {q['90%']:.2f} · P95 {q['95%']:.2f} · 最大 {q['max']:.2f}")
    bins = [-1, 0.5, 1, 2, 4, 8, 24, 168, 1e9]
    labels = ["≤0.5h", "0.5-1h", "1-2h", "2-4h", "4-8h", "8-24h", "1-7d", ">7d"]
    valid["lag_bin"] = pd.cut(valid["lag_h"], bins=bins, labels=labels)
    g = valid.groupby("lag_bin", observed=False).agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
    g["缺料率"] = (g["缺料"] / g["单量"]).round(4)
    g["损失工时"] = (g["缺料"] * FAIL_H).astype(int)
    print(g.to_string())
    g.to_csv(os.path.join(a.outdir, "by_sync_lag.csv"), encoding="utf-8-sig")

    sec("三、账面库存 vs 缺料率")
    sb = [-0.1, 0, 2, 5, 10, 20, 31]
    sl = ["0", "1-2", "3-5", "6-10", "11-20", "21-30"]
    valid["stock_bin"] = pd.cut(valid["stock"], bins=sb, labels=sl)
    g2 = valid.groupby("stock_bin", observed=False).agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
    g2["缺料率"] = (g2["缺料"] / g2["单量"]).round(4)
    g2["损失工时"] = (g2["缺料"] * FAIL_H).astype(int)
    print(g2.to_string())
    g2.to_csv(os.path.join(a.outdir, "by_stock.csv"), encoding="utf-8-sig")

    sec("四、网点排队长度 vs 缺料率")
    qb = [4, 20, 40, 60, 80, 100]
    ql = ["5-20", "21-40", "41-60", "61-80", "81-98"]
    valid["q_bin"] = pd.cut(valid["queue"], bins=qb, labels=ql)
    g3 = valid.groupby("q_bin", observed=False).agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
    g3["缺料率"] = (g3["缺料"] / g3["单量"]).round(4)
    print(g3.to_string())
    g3.to_csv(os.path.join(a.outdir, "by_queue.csv"), encoding="utf-8-sig")

    sec("五、网点 × 配件：缺料损失排行（Top 8）")
    g4 = valid.groupby(["service_hub", "part_category"]).agg(
        单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
    g4["缺料率"] = (g4["缺料"] / g4["单量"]).round(3)
    g4["损失工时"] = (g4["缺料"] * FAIL_H).astype(int)
    g4 = g4.sort_values("损失工时", ascending=False)
    print(g4.head(8).to_string())
    g4.to_csv(os.path.join(a.outdir, "by_hub_part.csv"), encoding="utf-8-sig")

    sec("六、按网点汇总")
    g5 = valid.groupby("service_hub").agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())),
                                          平均滞后h=("lag_h", "mean"), 平均库存=("stock", "mean"), 平均排队=("queue", "mean"))
    g5["缺料率"] = (g5["缺料"] / g5["单量"]).round(3)
    g5["损失工时"] = (g5["缺料"] * FAIL_H).astype(int)
    print(g5.round(2).sort_values("损失工时", ascending=False).to_string())

    sec("七、风险标记规则 ROI 扫描（净节省 = 拦住缺料×120 − 标记单量×20）")
    rows = []
    for s in range(0, 31, 2):
        for h in [0, 1, 2, 4, 6, 8, 12, 18, 24, 36, 48, 72]:
            m = valid["stock"].le(s) | valid["lag_h"].gt(h)
            n = int(m.sum())
            if n == 0 or n >= n_valid:
                continue
            d = int((m & (valid["done"] == 0)).sum())
            rows.append(dict(
                库存阈值=s, 滞后阈值h=h, 标记单量=n, 精确率=round(d / n, 3),
                覆盖缺料=round(d / n_fail, 3), 标记成本h=int(n * FLAG_H),
                挽回h=int(d * FAIL_H), 净节省h=int(d * FAIL_H - n * FLAG_H)))
    roi = pd.DataFrame(rows).sort_values("净节省h", ascending=False)
    print(roi.head(10).to_string(index=False))
    roi.to_csv(os.path.join(a.outdir, "roi_scan.csv"), index=False, encoding="utf-8-sig")

    if len(roi):
        best = roi.iloc[0]
        # 可执行阈值（整数、好记）
        m = valid["stock"].le(2) | valid["lag_h"].gt(8)
        n = int(m.sum())
        d = int((m & (valid["done"] == 0)).sum())
        print(f"\n  ▶ 推荐落地规则：库存 ≤2 件 或 同步滞后 >8 小时 → 标记风险")
        print(f"    命中 {n} 单（{n/n_valid:.1%}），其中真缺料 {d} 单（精确率 {d/n:.1%}，覆盖 {d/n_fail:.1%} 缺料）")
        print(f"    标记成本 {n*FLAG_H:,.0f}h ｜ 挽回 {d*FAIL_H:,.0f}h ｜ 净节省 {d*FAIL_H-n*FLAG_H:,.0f}h")

    sec("八、按天趋势")
    valid["day"] = valid["dt"].dt.date
    g6 = valid.groupby("day").agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
    g6["缺料率"] = (g6["缺料"] / g6["单量"]).round(3)
    print(g6.to_string())

    # 落盘明细，方便 README 引用
    keep = ["dispatch_id", "dispatch_time", "stock_synced_at", "service_hub", "part_category",
            "available_stock", "queue_length", "job_completed", "lag_h"]
    valid[keep].to_csv(os.path.join(a.outdir, "cleaned_with_lag.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] 结果已写入 {a.outdir}/")


if __name__ == "__main__":
    main()
