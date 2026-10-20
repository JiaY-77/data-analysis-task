#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""出两张核心图：滞后断崖 + 滞后×排队热力图"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/dispatch_orders.csv", encoding="utf-8-sig")
df["dt"] = pd.to_datetime(df["dispatch_time"], utc=True)
df["sync"] = pd.to_datetime(df["stock_synced_at"], utc=True)
df["lag"] = (df["dt"] - df["sync"]).dt.total_seconds() / 3600
df["done"] = pd.to_numeric(df["job_completed"], errors="coerce")
df["queue"] = pd.to_numeric(df["queue_length"], errors="coerce")
v = df[df["done"].notna()].copy()
base = (v["done"] == 0).mean()

# 图 1：滞后断崖
v["lag_bin"] = pd.cut(v["lag"], [-1, 0.5, 1, 2], labels=["≤0.5 小时", "0.5–1 小时", "1–2 小时"])
g = v.groupby("lag_bin", observed=False).agg(n=("done", "size"), f=("done", lambda s: int((s == 0).sum())))
g["rate"] = g["f"] / g["n"] * 100

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
bars = ax.bar(g.index.astype(str), g["rate"], color=["#2E5EAA", "#E4572E", "#E4572E"], width=0.55)
ax.axhline(base * 100, ls="--", lw=1.2, color="#888", label=f"整体缺料率 {base*100:.1f}%")
for b, (n, f, r) in zip(bars, g[["n", "f", "rate"]].itertuples(index=False)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
            f"{r:.1f}%\n({f}/{n} 单)", ha="center", fontsize=10)
ax.set_ylabel("缺料率（%）")
ax.set_xlabel("派工时库存数据的同步滞后")
ax.set_title("库存同步滞后 >0.5 小时，缺料率从 6.1% 跳到 31.6%", fontsize=12, pad=12)
ax.set_ylim(0, max(g["rate"]) * 1.35)
ax.legend(loc="upper left", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("out/fig1_lag_gap.png")
print("图1 已出：out/fig1_lag_gap.png")

# 图 2：滞后 × 排队 热力图
v["lag2"] = (v["lag"] > 0.5).map({False: "滞后 ≤0.5h", True: "滞后 >0.5h"})
v["qb"] = pd.cut(v["queue"], [4, 40, 60, 80, 100], labels=["≤40", "41–60", "61–80", "81–98"])
piv = v.pivot_table(index="lag2", columns="qb", values="done",
                    aggfunc=lambda s: (s == 0).sum() / len(s) * 100, observed=False)
cnt = v.pivot_table(index="lag2", columns="qb", values="done", aggfunc="size", observed=False)

fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=150)
im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=piv.values.max() * 1.05)
ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
ax.set_xlabel("派工时网点排队长度（单）")
ax.set_ylabel("库存同步滞后")
ax.set_title("每一档排队下，滞后 >0.5h 的缺料率都显著更高", fontsize=12, pad=12)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        val = piv.values[i, j]
        n = cnt.values[i, j]
        if pd.isna(val):
            continue
        ax.text(j, i, f"{val:.1f}%\n({int(n)} 单)", ha="center", va="center",
                fontsize=9.5, color="black" if val < piv.values.max() * 0.65 else "white")
fig.colorbar(im, ax=ax, label="缺料率（%）")
fig.tight_layout()
fig.savefig("out/fig2_matrix.png")
print("图2 已出：out/fig2_matrix.png")
print(f"\n基线缺料率 {base*100:.2f}%")
