#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""robust_check.py —— 稳健性检验（补上「相关 vs 因果」这一关）

三件事：
1. 多变量 logistic：控制库存/排队/网点/品类后，滞后效应还在不在（看 OR 变化）
2. 分层：每个网点、每个品类内部，滞后效应是否都成立
3. 时间切分：前 5 天定阈值 → 后 2 天验证（回应「同数据选阈值」的过拟合质疑）
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

df = pd.read_csv("data/dispatch_orders.csv", encoding="utf-8-sig")
df["dt"] = pd.to_datetime(df["dispatch_time"], utc=True)
df["sync"] = pd.to_datetime(df["stock_synced_at"], utc=True)
df["lag"] = (df["dt"] - df["sync"]).dt.total_seconds() / 3600
df["done"] = pd.to_numeric(df["job_completed"], errors="coerce")
df["stock"] = pd.to_numeric(df["available_stock"], errors="coerce")
df["queue"] = pd.to_numeric(df["queue_length"], errors="coerce")
v = df[df["done"].notna()].copy()
v["y"] = (v["done"] == 0).astype(int)
v["lag_high"] = (v["lag"] > 0.5).astype(int)
v["day"] = v["dt"].dt.date
v["hour"] = v["dt"].dt.hour
v["part_category"] = v["part_category"].astype(str)
v["service_hub"] = v["service_hub"].astype(str)

print("=" * 72)
print("一、多变量 logistic：滞后效应的 OR 会不会被混杂吃掉")
print("=" * 72)
models = {
    "M1 只有滞后": "y ~ lag_high",
    "M2 +库存+排队": "y ~ lag_high + stock + queue",
    "M3 +网点": "y ~ lag_high + stock + queue + C(service_hub)",
    "M4 +网点+品类": "y ~ lag_high + stock + queue + C(service_hub) + C(part_category)",
}
rows = []
for name, f in models.items():
    try:
        m = smf.logit(f, data=v).fit(disp=False)
        ci = m.conf_int().loc["lag_high"]
        rows.append({"模型": name, "OR": round(float(np.exp(m.params["lag_high"])), 2),
                     "95%CI 下限": round(float(np.exp(ci[0])), 2),
                     "95%CI 上限": round(float(np.exp(ci[1])), 2),
                     "p 值": round(float(m.pvalues["lag_high"]), 4), "样本": int(m.nobs)})
    except Exception as e:
        rows.append({"模型": name, "OR": f"失败:{type(e).__name__}"})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 72)
print("二、分层：每个网点 / 每个品类内部，滞后>0.5h 是否都更差")
print("=" * 72)


def strat(col):
    out = []
    for k, sub in v.groupby(col):
        hi, lo = sub[sub.lag_high == 1], sub[sub.lag_high == 0]
        if len(hi) < 5 or len(lo) < 5:
            out.append({col: k, "高滞后率": f"{hi['y'].mean():.1%}" if len(hi) else "-",
                        "低滞后率": f"{lo['y'].mean():.1%}" if len(lo) else "-",
                        "样本(高/低)": f"{len(hi)}/{len(lo)}", "判定": "样本不足"})
            continue
        rh, rl = hi["y"].mean(), lo["y"].mean()
        out.append({col: k, "高滞后率": f"{rh:.1%}", "低滞后率": f"{rl:.1%}",
                    "样本(高/低)": f"{len(hi)}/{len(lo)}",
                    "判定": "同向" if rh > rl else "反例"})
    print(pd.DataFrame(out).to_string(index=False))


print("\n-- 按网点 --")
strat("service_hub")
print("\n-- 按品类 --")
strat("part_category")

print("\n" + "=" * 72)
print("三、时间切分：前 5 天定阈值 → 后 2 天验证")
print("=" * 72)
days = sorted(v["day"].unique())
train = v[v["day"].isin(days[:5])]
test = v[v["day"].isin(days[5:])]
print(f"  训练 {days[0]} ~ {days[4]}（{len(train)} 单）｜验证 {days[5]} ~ {days[-1]}（{len(test)} 单）")
for name, sub in [("训练集", train), ("验证集", test)]:
    hi, lo = sub[sub.lag_high == 1], sub[sub.lag_high == 0]
    if len(hi) and len(lo):
        print(f"  {name}: 滞后>0.5h 缺料率 {hi['y'].mean():.1%}（{int(hi['y'].sum())}/{len(hi)}）"
              f" vs ≤0.5h {lo['y'].mean():.1%}（{int(lo['y'].sum())}/{len(lo)}）"
              f" → 倍数 {(hi['y'].mean() / lo['y'].mean() if lo['y'].mean() else float('nan')):.1f}")

print("\n" + "=" * 72)
print("四、剂量反应：滞后分更细，看是不是单调（平台=可能混杂）")
print("=" * 72)
v["lag_bin2"] = pd.cut(v["lag"], [-1, 0.15, 0.3, 0.5, 0.7, 1.2],
                       labels=["≤9分", "9-18分", "18-30分", "30-42分", ">42分"])
g = v.groupby("lag_bin2", observed=False).agg(单量=("y", "size"), 缺料=("y", "sum"))
g["缺料率"] = (g["缺料"] / g["单量"]).round(3)
print(g.to_string())
