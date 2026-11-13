"""摆动点 OOS 显著性 + 随机入场安慰剂对照（A股 #7 同法）。
对选定配置（a1.5, 288/3.0 与 576/3.0）在 OOS(2024-2026)：
  信号 E vs 随机 bar 入场 E（同期、同政策、2000 次抽样），附 t 值。
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

COST = 0.0012
rng = np.random.default_rng(7)
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])
n_bars = len(b["close"])
if "atr" not in b:
    c, h, l = b["close"], b["high"], b["low"]
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    b["atr"] = atr

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")
df["year"] = pd.to_datetime(df.event_time).dt.year
oos = df[df.year >= 2024]

# OOS 时间范围的 bar 索引边界（随机入场必须落在同期）
times = pd.to_datetime(pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "bars_crypto" / "30m/BTCUSDT.parquet")["time"])
oos_mask = (times >= "2024-01-01").to_numpy()
oos_idx = np.where(oos_mask)[0]
lo, hi = int(oos_idx.min()), int(oos_idx.max()) - 700

for hold in (288, 576):
    for side, label in ((1, "BUY"), (0, "SELL-short")):
        sub = oos[oos.is_buy == side]
        rets = []
        for r in sub.itertuples():
            res = simulate_exit(b, int(r.conf_entry_idx), side, float(r.atr), hold=hold, stop_atr=3.0, cost=COST)
            if res is not None:
                rets.append(res.ret)
        rets = np.array(rets)
        t_stat = rets.mean() / (rets.std(ddof=1) / np.sqrt(len(rets))) if len(rets) > 2 else np.nan

        rnd = []
        idxs = rng.integers(lo, hi, size=2000)
        for i in idxs:
            a = float(b["atr"][i]) if "atr" in b else 0.0
            if a <= 0:
                continue
            res = simulate_exit(b, int(i), side, a, hold=hold, stop_atr=3.0, cost=COST)
            if res is not None:
                rnd.append(res.ret)
        rnd = np.array(rnd)
        print(f"hold={hold} {label:11s} 信号: n={len(rets):4d} E={rets.mean()*1e4:+7.1f}bp t={t_stat:+.2f} | "
              f"随机: E={rnd.mean()*1e4:+7.1f}bp | edge={rets.mean()*1e4 - rnd.mean()*1e4:+7.1f}bp")
