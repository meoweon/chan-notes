"""个股级市场周期 × 确认买点收益 条件期望表（第 2 步：先看路由有没有东西可路由）。

A 套（缠论原生）：按信号时个股所处线段的结构分
  趋势上涨 = 线段内含≥2中枢 且 线段向上
  趋势下跌 = 线段内含≥2中枢 且 线段向下
  盘整     = 线段内含 1 中枢
  无结构   = 线段内尚无中枢
B 套（ER 量表）：bg_cycle_stage（±2 强趋势/±1 弱趋势/0 区间）

输出中文表：笔数 / 平均每笔盈亏(bp) / 胜率 / 盈亏比 / 样本内(2020-2023) / 样本外(2024-2026)
另附：门控内（frenzy 热期）样本外口径。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

df = pd.read_parquet(ROOT / "data" / "events" / "events_v8.parquet")
df = df[df["pol144_ret"].notna()].copy()  # pol144_ret 只对 conf_buy 打点（8,646 笔）
df["event_time"] = pd.to_datetime(df["event_time"])
df["ret"] = df["pol144_ret"] * 1e4  # 存的是小数收益，转 bp
SPLIT = pd.Timestamp("2024-01-01")

# frenzy 门控（PIT：T-1 日）
fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"].shift(1)
thr = fz[fz.index < SPLIT].median()
df["d"] = df["event_time"].dt.normalize()
df["hot"] = df["d"].map(fz >= thr).fillna(False).astype(bool)

def state_a(r):
    if r["ch_seg_n_zs"] >= 2:
        return "趋势上涨" if r["ch_seg_dir"] > 0 else "趋势下跌"
    if r["ch_seg_n_zs"] == 1:
        return "盘整（1中枢）"
    return "无结构"

def state_b(r):
    s = r["bg_cycle_stage"]
    return {2: "强趋势上涨", 1: "弱趋势上涨", 0: "区间", -1: "弱趋势下跌", -2: "强趋势下跌"}.get(int(s), "未知")

df["状态A"] = df.apply(state_a, axis=1)
df["状态B"] = df.apply(state_b, axis=1)

def table(g, name):
    out = []
    for st, sub in g.groupby(name):
        if len(sub) < 30:
            continue
        ins = sub[sub.event_time < SPLIT]["ret"]
        oos = sub[sub.event_time >= SPLIT]["ret"]
        oos_hot = sub[(sub.event_time >= SPLIT) & sub.hot]["ret"]
        w = sub[sub.ret > 0]["ret"]
        l = sub[sub.ret <= 0]["ret"]
        rr = (w.mean() / abs(l.mean())) if len(l) and l.mean() != 0 else float("nan")
        out.append({
            "状态": st, "笔数": len(sub),
            "平均每笔盈亏(bp)": round(sub["ret"].mean(), 1),
            "胜率": f"{(sub.ret > 0).mean()*100:.1f}%",
            "盈亏比": round(rr, 2),
            "样本内20-23(bp)": round(ins.mean(), 1) if len(ins) else None,
            "样本外24-26(bp)": round(oos.mean(), 1) if len(oos) else None,
            "样本外且热期(bp)": round(oos_hot.mean(), 1) if len(oos_hot) else None,
            "热期笔数": len(oos_hot),
        })
    t = pd.DataFrame(out).sort_values("平均每笔盈亏(bp)", ascending=False)
    return t

for name in ["状态A", "状态B"]:
    t = table(df, name)
    print(f"\n===== {name} =====")
    print(t.to_string(index=False))
    t.to_csv(ROOT / "reports" / f"stock_state_{name[-1]}.csv", index=False, encoding="utf-8-sig")

# ER60 连续量五分位对照
df["er档"] = pd.qcut(df["bg_er60"], 5, labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"])
t = table(df, "er档")
print("\n===== bg_er60 五分位 =====")
print(t.to_string(index=False))
print(f"\n总笔数 {len(df)}，全体平均 {df.ret.mean():.1f}bp")
