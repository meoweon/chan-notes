"""门控内 Brooks 背景过滤组合终审 —— OOS 2024-2026（冻结 frenzy 阈值 0.00717）。

假设（来自 Brooks 理论与 bg_check 单变量证据）：
  H1 bg_ttr==0（不在 TTR 内开仓）：TTR 压倒一切，区间内信号均值 -32bp
  H2 bg_sell_zone==1（区间上 1/3 才开）：趋势彩票吃动量，卖区信号 +57/+147bp
  H3 H1+H2 组合
对照：frozen_gate 基线（+79.5%/1.21）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.eval import bars as bars_mod
from chan_notes.eval.metrics import expectancy

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2024-01-01")

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"])
fz["prev5"] = fz.frenzy5.shift(1)
thr = float(fz.loc[fz.d < CUTOFF, "frenzy5"].median())
prev_map = dict(zip(fz["d"].dt.strftime("%Y-%m-%d"), fz["prev5"]))

ev = pd.read_parquet(ROOT / "data" / "events" / "events_v8.parquet")
ev["event_time"] = pd.to_datetime(ev["event_time"])
ev = ev[ev.event_time >= CUTOFF].copy()
ev["_d"] = ev.event_time.dt.strftime("%Y-%m-%d")
ev["fz"] = ev._d.map(prev_map)
buys = ev[(ev.is_buy == 1) & (ev.conf_ok == 1) & (ev.fz >= thr)].copy()
print(f"门控内买点 {len(buys)}")

codes = sorted(buys.code.unique())
bars = {c: df[df.time >= CUTOFF].reset_index(drop=True)
        for c, df in bars_mod.load_frames(codes).items()}
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=False)


def bt(label, sub):
    res = portfolio.run(sub, bars, p, entry_mode="confirm")
    m, tr = res["metrics"], res["trades"]
    ex = expectancy(tr.ret.to_numpy()) if len(tr) else {}
    print(f"  {label:22s} total={m.get('total_ret', 0) * 100:+7.1f}% sharpe={m.get('sharpe', 0):5.2f} "
          f"mdd={m.get('mdd', 0) * 100:6.1f}% n={m.get('n_trades', 0):4d} "
          f"E={ex.get('expectancy_bp', 0):+7.1f}bp win={ex.get('win_rate', 0) * 100:4.1f}%")
    res["equity"].to_parquet(ROOT / "data" / "bt" / f"bg_{label}_equity.parquet")


print("\n===== OOS 2024-2026 门控内背景过滤（10×10%，144bar/3ATR）=====")
bt("基线_全做", buys)
bt("TTR过滤", buys[buys.bg_ttr == 0])
bt("卖区过滤", buys[buys.bg_sell_zone == 1])
bt("TTR+卖区", buys[(buys.bg_ttr == 0) & (buys.bg_sell_zone == 1)])
bt("非买区", buys[buys.bg_buy_zone == 0])
