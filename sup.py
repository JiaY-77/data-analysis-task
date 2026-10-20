#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补充：配件维度 + 变量交叉验证 + 规则组合扫描"""
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAIL_H, FLAG_H = 120.0, 20.0
df = pd.read_csv("data/dispatch_orders.csv", encoding="utf-8-sig")
df["dt"] = pd.to_datetime(df["dispatch_time"], utc=True)
df["sync"] = pd.to_datetime(df["stock_synced_at"], utc=True)
df["lag"] = (df["dt"] - df["sync"]).dt.total_seconds() / 3600
df["done"] = pd.to_numeric(df["job_completed"], errors="coerce")
df["queue"] = pd.to_numeric(df["queue_length"], errors="coerce")
v = df[df["done"].notna()].copy()
n_fail = int((v["done"] == 0).sum())


def rate(g):
    g["缺料率"] = (g["缺料"] / g["单量"]).round(3)
    g["损失h"] = (g["缺料"] * FAIL_H).astype(int)
    return g.sort_values("缺料率", ascending=False)


print("=" * 70)
print("A. 配件大类：缺料率排行")
print("=" * 70)
g = v.groupby("part_category").agg(单量=("done", "size"), 缺料=("done", lambda s: int((s == 0).sum())))
print(rate(g).to_string())

print("\n" + "=" * 70)
print("B. 同步滞后 × 排队长度（看滞后是不是只是'忙'的代理）")
print("=" * 70)
v["lag2"] = (v["lag"] > 0.5).map({True: "滞后>0.5h", False: "滞后≤0.5h"})
v["qb"] = pd.cut(v["queue"], [4, 40, 60, 80, 100], labels=["≤40", "41-60", "61-80", "81-98"])
t = v.pivot_table(index="lag2", columns="qb", values="done", aggfunc=["size", lambda s: (s == 0).sum()], observed=False)
print(t.to_string())

print("\n" + "=" * 70)
print("C. 账面库存 × 排队长度")
print("=" * 70)
v["sb"] = pd.cut(v["available_stock"], [-1, 2, 10, 30], labels=["≤2", "3-10", "11-30"])
t2 = v.pivot_table(index="sb", columns="qb", values="done", aggfunc=[lambda s: len(s), lambda s: (s == 0).sum()], observed=False)
print(t2.to_string())

print("\n" + "=" * 70)
print("D. 单变量规则对比（净节省 = 拦住×120 − 标记×20）")
print("=" * 70)
rules = {
    "排队 >80": v["queue"] > 80,
    "排队 >60": v["queue"] > 60,
    "滞后 >0.5h": v["lag"] > 0.5,
    "库存 ≤2": v["available_stock"] <= 2,
    "排队>80 或 滞后>0.5h": (v["queue"] > 80) | (v["lag"] > 0.5),
    "排队>60 或 滞后>0.5h": (v["queue"] > 60) | (v["lag"] > 0.5),
    "排队>80 或 库存≤2": (v["queue"] > 80) | (v["available_stock"] <= 2),
    "排队>60 或 库存≤25": (v["queue"] > 60) | (v["available_stock"] <= 25),
}
rows = []
for name, m in rules.items():
    n = int(m.sum())
    d = int((m & (v["done"] == 0)).sum())
    rows.append(dict(规则=name, 标记单量=n, 精确率=round(d / n, 3) if n else 0,
                     覆盖缺料=round(d / n_fail, 3), 成本h=int(n * FLAG_H),
                     挽回h=int(d * FAIL_H), 净节省h=int(d * FAIL_H - n * FLAG_H)))
print(pd.DataFrame(rows).sort_values("净节省h", ascending=False).to_string(index=False))

print("\n" + "=" * 70)
print("E. 组合规则细扫（排队阈值 × 滞后阈值）")
print("=" * 70)
rows = []
for q in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
    for lag in [0.25, 0.5, 0.75]:
        m = (v["queue"] >= q) | (v["lag"] > lag)
        n = int(m.sum())
        if n == 0:
            continue
        d = int((m & (v["done"] == 0)).sum())
        rows.append({"排队>=": q, "滞后>h": lag, "标记": n, "精确率": round(d / n, 3),
                     "覆盖": round(d / n_fail, 3), "净h": int(d * FAIL_H - n * FLAG_H)})
print(pd.DataFrame(rows).sort_values("净h", ascending=False).head(8).to_string(index=False))
