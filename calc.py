#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""calc.py —— 关键数字快算（对应 README 里引用的每一个数）"""
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

df = pd.read_csv("data/dispatch_orders.csv", encoding="utf-8-sig")
df["dt"] = pd.to_datetime(df["dispatch_time"], utc=True)
df["sync"] = pd.to_datetime(df["stock_synced_at"], utc=True)
df["lag"] = (df["dt"] - df["sync"]).dt.total_seconds() / 3600
df["done"] = pd.to_numeric(df["job_completed"], errors="coerce")
df["q"] = pd.to_numeric(df["queue_length"], errors="coerce")
df["stk"] = pd.to_numeric(df["available_stock"], errors="coerce")
v = df[df["done"].notna()]
n, f = len(v), int((v["done"] == 0).sum())
base = f / n

print(f"1) 总体缺料率 = {f}/{n} = {base:.2%}")
print(f"2) 滞后 均值 {v['lag'].mean():.4f} h · 中位 {v['lag'].median():.4f} h · P90 {v['lag'].quantile(.9):.4f} h · 最大 {v['lag'].max():.4f} h")

for th in (50, 70):
    m = v["q"] > th
    print(f"3) queue>{th}: {int(m.sum())} 单 · 缺料 {int((m & (v['done']==0)).sum())} · 缺料率 {(m & (v['done']==0)).sum()/m.sum():.2%}")

m0 = v["stk"] == 0
print(f"4) stock=0: {int(m0.sum())} 单 · 缺料 {int((m0 & (v['done']==0)).sum())} · 缺料率 {(m0 & (v['done']==0)).sum()/m0.sum():.2%}")

hub = v.groupby("service_hub").agg(n=("done", "size"), f=("done", lambda s: int((s == 0).sum())))
hub["rate"] = hub["f"] / hub["n"]
print(f"5) 网点（均值 {base:.2%}，两倍 {2*base:.2%}）:")
print(hub.sort_values("rate", ascending=False).round(4).to_string())
print(f"   存在超过两倍的网点: {bool((hub['rate'] > 2*base).any())}")

print("\n-- 规则（且）：滞后 > L 且 排队 > Q --")
for L, Q in [(0.5, 70), (0.5, 50), (0.25, 70)]:
    m = (v["lag"] > L) & (v["q"] > Q)
    nn, dd = int(m.sum()), int((m & (v["done"] == 0)).sum())
    if nn:
        print(f"L>{L}h & Q>{Q}: {nn} 单 · 缺料 {dd} · 精确率 {dd/nn:.1%} · 覆盖 {dd/f:.1%} · 净 {dd*120-nn*20:+.0f}h")

print("\n-- 规则（或）：滞后 > L 或 排队 > Q --")
for L, Q in [(0.5, 70), (0.5, 80), (0.25, 70)]:
    m = (v["lag"] > L) | (v["q"] > Q)
    nn, dd = int(m.sum()), int((m & (v["done"] == 0)).sum())
    print(f"L>{L}h | Q>{Q}: {nn} 单 · 缺料 {dd} · 精确率 {dd/nn:.1%} · 覆盖 {dd/f:.1%} · 净 {dd*120-nn*20:+.0f}h")
