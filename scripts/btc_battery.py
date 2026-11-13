"""BTC 跨市场验证电池（一次性分析脚本）。

对应 A 股实验：#1 裸BSP / #5 退出网格 / #8 去尾 / 存活率 / 卖侧（可做空!）。
注意：加密 30m 每天 48 根（A股 8 根），144bar≈3天；补 864bar≈18天 对齐 A 股持有期。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.models import train

COST = 0.0012  # taker 往返 10bp + 滑点 2bp

df = train.load("events_b1")
df = df[df.code == "BTCUSDT"].copy()
arrays = {c: for_policy(a) for c, a in load_many(["BTCUSDT"], "30m").items()}
b = arrays["BTCUSDT"]

print(f"events={len(df)}  span={df.event_time.min()} ~ {df.event_time.max()}")
print(f"bsp_type 分布: {df[df.is_buy==1].bsp_type.value_counts().to_dict()}")

# ---- 1. 裸 BSP（信号次 bar 开盘进，24bar/2ATR）----
def eval_subset(sub, entry_col, hold, stop):
    rets = []
    for r in sub.itertuples():
        i0 = int(getattr(r, entry_col))
        res = simulate_exit(b, i0, int(r.is_buy), float(r.atr), hold=hold, stop_atr=stop, cost=COST)
        if res is not None:
            rets.append(res.ret)
    rets = np.array(rets)
    if len(rets) == 0:
        return None
    return dict(n=len(rets), E=rets.mean() * 1e4, win=(rets > 0).mean() * 100)

buy_all = df[df.is_buy == 1]
sell_all = df[df.is_buy == 0]
r = eval_subset(buy_all, "bar_idx", 24, 2.0)
print(f"\n[1] 裸买点 24/2ATR: n={r['n']} E={r['E']:+.1f}bp win={r['win']:.1f}%")
r = eval_subset(sell_all, "bar_idx", 24, 2.0)
print(f"    裸卖点 24/2ATR: n={r['n']} E={r['E']:+.1f}bp win={r['win']:.1f}%  (做空口径)")

# ---- 2. 确认买点退出网格 ----
cb = df[(df.is_buy == 1) & (df.conf_ok == 1)]
cs = df[(df.is_buy == 0) & (df.conf_ok == 1)]
print(f"\n[2] 确认买点 n={len(cb)} / 确认卖点 n={len(cs)}  退出网格 (E bp):")
print(f"{'hold\\stop':>10s}" + "".join(f"{s:>9.1f}" for s in (2.0, 3.0, 4.0)))
for hold in (48, 96, 144, 288, 864):
    row = f"{hold:>10d}"
    for stop in (2.0, 3.0, 4.0):
        r = eval_subset(cb, "conf_entry_idx", hold, stop)
        row += f"{r['E']:>+9.1f}" if r else f"{'--':>9s}"
    print(row)
print("  (卖侧做空同样网格):")
for hold in (144, 288, 864):
    row = f"{hold:>10d}"
    for stop in (2.0, 3.0, 4.0):
        r = eval_subset(cs, "conf_entry_idx", hold, stop)
        row += f"{r['E']:>+9.1f}" if r else f"{'--':>9s}"
    print(row)

# ---- 3. 去尾测试（144/3ATR 确认买点）----
rets = []
for r_ in cb.itertuples():
    res = simulate_exit(b, int(r_.conf_entry_idx), 1, float(r_.atr), hold=144, stop_atr=3.0, cost=COST)
    if res is not None:
        rets.append(res.ret)
rets = np.sort(np.array(rets))
n = len(rets)
print(f"\n[3] 去尾测试 (144/3ATR 确认买点, n={n}):")
print(f"    全体 E={rets.mean()*1e4:+.1f}bp  中位={np.median(rets)*1e4:+.1f}bp")
for q in (0.01, 0.05, 0.10):
    cut = rets[: int(n * (1 - q))]
    print(f"    去 top{int(q*100)}%: E={cut.mean()*1e4:+.1f}bp")
top5 = rets[int(n * 0.95):]
print(f"    top5% 均值={top5.mean()*1e4:+.0f}bp, 贡献净利润 {top5.sum()/max(rets.sum(),1e-9)*100:.0f}%")

# ---- 4. 存活 ----
surv = df.clear_bar == -1
print(f"\n[4] BSP 存活: 回放结束仍存活 {surv.mean()*100:.1f}%（A股 30m 是 42%）")
if "survive_144" in df.columns:
    s = df.dropna(subset=["survive_144"])
    print(f"    survive_144 基率: {s.survive_144.mean()*100:.1f}%")

# ---- 5. 分年度（144/3ATR）----
cb2 = cb.copy()
cb2["ret144"] = [
    (simulate_exit(b, int(r_.conf_entry_idx), 1, float(r_.atr), hold=144, stop_atr=3.0, cost=COST).ret
     if simulate_exit(b, int(r_.conf_entry_idx), 1, float(r_.atr), hold=144, stop_atr=3.0, cost=COST) else np.nan)
    for r_ in cb2.itertuples()]
cb2["year"] = pd.to_datetime(cb2.event_time).dt.year
g = cb2.groupby("year").ret144.agg(["count", "mean"])
g["mean"] = g["mean"] * 1e4
print("\n[5] 分年度 E(bp):")
print(g.round(1).to_string())
