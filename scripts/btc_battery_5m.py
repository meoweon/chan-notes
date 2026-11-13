"""BTC 5m 验证电池（events_b5）——先机械 edge 后 ML 的纪律顺序：
[1] 裸 BSP（信号次日开盘价，24/144bar）买卖双侧
[2] 确认买点 hold×stop 网格（全样本 + IS/OOS 2020-23/2024-26）
[3] 去尾测试（最优配置）
[4] 成本敏感性：12bp vs 8bp（maker 往返）
成本=币安 taker 0.05%×2 + 滑点 2bp = 12bp。
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

COST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0012
df = train.load(sys.argv[1] if len(sys.argv) > 1 else "events_b5")
bars = for_policy(load_many(["BTCUSDT"], "5m")["BTCUSDT"])
print(f"events={len(df)} BUY={(df.is_buy==1).sum()} SELL={(df.is_buy==0).sum()} cost={COST*1e4:.0f}bp")

# ---------- [1] 裸 BSP ----------
for hold in (24, 144):
    for side, m in (("BUY", df.is_buy == 1), ("SELL", df.is_buy == 0)):
        rets = []
        for r in df[m].itertuples():
            i = int(r.bar_idx) + 1
            if i + hold >= len(bars["close"]):
                continue
            ret = bars["close"][i + hold] / bars["open"][i] - 1
            rets.append(ret if side == "BUY" else -ret)
        rets = np.array(rets) - COST
        t = rets.mean() / (rets.std() / np.sqrt(len(rets)))
        print(f"[1] 裸{side} hold={hold}: n={len(rets)} E={rets.mean()*1e4:+.1f}bp t={t:+.2f}")

# ---------- [2] 确认买点网格 IS/OOS ----------
cb = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
cb["year"] = pd.to_datetime(cb.event_time).dt.year
print(f"\n[2] 确认买点 n={len(cb)} (IS {(cb.year<=2023).sum()} / OOS {(cb.year>=2024).sum()})")
print(f"{'hold':>5s} {'stop':>4s} | {'全样本E':>8s} {'n':>5s} | {'IS E':>8s} | {'OOS E':>8s} {'OOS胜率':>7s}")
for hold in (144, 288, 864, 1728, 5184):
    for stop in (3.0, 4.0):
        def ev(sub):
            rr = []
            for r in sub.itertuples():
                res = simulate_exit(bars, int(r.conf_entry_idx), 1, float(r.atr), hold=hold, stop_atr=stop, cost=COST)
                if res is not None:
                    rr.append(res.ret)
            return (len(rr), np.mean(rr) * 1e4, (np.array(rr) > 0).mean() * 100) if rr else (0, np.nan, np.nan)
        n0, e0, _ = ev(cb)
        n1, e1, _ = ev(cb[cb.year <= 2023])
        n2, e2, w2 = ev(cb[cb.year >= 2024])
        print(f"{hold:>5d} {stop:>4.1f} | {e0:>+8.1f} {n0:>5d} | {e1:>+8.1f} | {e2:>+8.1f} {w2:>6.1f}%")

# ---------- [3] 去尾（144/3.0 基准） ----------
rets = []
for r in cb.itertuples():
    res = simulate_exit(bars, int(r.conf_entry_idx), 1, float(r.atr), hold=144, stop_atr=3.0, cost=COST)
    if res is not None:
        rets.append(res.ret)
r = np.sort(np.array(rets))
print(f"\n[3] 144/3.0 去尾: n={len(r)} E={r.mean()*1e4:+.1f}bp 中位={np.median(r)*1e4:+.1f}bp "
      f"去top1%={r[:int(len(r)*0.99)].mean()*1e4:+.1f}bp 去top5%={r[:int(len(r)*0.95)].mean()*1e4:+.1f}bp")
