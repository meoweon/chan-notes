"""门控内"只做高存活信号"终审 —— OOS 2024-2026（冻结 frenzy 阈值 0.00717）。

用户问题：只对高存活的买点和卖点交易，收益是否更可靠？
两组变体（与 #19 冻结门控基线同引擎、同参数）：
  V1 门控 + 存活过滤（买侧 score ≥ q0.6/q0.8，样本外 WF 分数）
  V2 门控 + 高存活卖点离场（卖侧 score ≥ q0.5/q0.8 的 BSP 才触发离场）
对照：frozen_gate 基线（无信号离场）、门控+全部卖点离场。
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

# ── 冻结阈值（与 #19 完全一致）──
fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"])
fz["prev5"] = fz.frenzy5.shift(1)
prev_map = dict(zip(fz["d"].dt.strftime("%Y-%m-%d"), fz["prev5"]))
thr = float(fz.loc[fz.d < CUTOFF, "frenzy5"].median())
print(f"frozen threshold = {thr:.5f}")

# ── 事件：门控内 OOS 确认买点 + 全部卖点 ──
ev = pd.read_parquet(ROOT / "data" / "events" / "events_v7.parquet")
ev["event_time"] = pd.to_datetime(ev["event_time"])
ev = ev[ev.event_time >= CUTOFF].copy()
ev["_d"] = ev.event_time.dt.strftime("%Y-%m-%d")
ev["fz"] = ev._d.map(prev_map)
gated = ev[ev.fz >= thr].copy()
buys = gated[(gated.is_buy == 1) & (gated.conf_ok == 1)].copy()
sells = gated[gated.is_buy == 0].copy()
print(f"gate on: {buys._d.nunique()} 天 / {ev._d.nunique()} 天；门控内买点 {len(buys)}、卖点 {len(sells)}")

# ── OOF 存活分数合并 ──
sc = pd.read_parquet(ROOT / "data" / "events" / "wf_surv.parquet")
sc["event_time"] = pd.to_datetime(sc["event_time"])
key = ["code", "event_time", "bsp_type", "is_buy"]
sc = sc[key + ["score"]].rename(columns={"score": "surv_score"})
buys = buys.merge(sc, on=key, how="left")
sells = sells.merge(sc, on=key, how="left")
print(f"买侧有分数 {buys.surv_score.notna().sum()}/{len(buys)}；卖侧 {sells.surv_score.notna().sum()}/{len(sells)}")

codes = sorted(set(buys.code) | set(sells.code))
bars = {c: df[df.time >= CUTOFF].reset_index(drop=True)
        for c, df in bars_mod.load_frames(codes).items()}
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=False)


def bt(label, ev_df, params):
    res = portfolio.run(ev_df, bars, params, entry_mode="confirm")
    m, tr = res["metrics"], res["trades"]
    ex = expectancy(tr.ret.to_numpy()) if len(tr) else {}
    print(f"  {label:28s} total={m.get('total_ret', 0) * 100:+7.1f}% cagr={m.get('cagr', 0) * 100:+5.1f}% "
          f"sharpe={m.get('sharpe', 0):5.2f} mdd={m.get('mdd', 0) * 100:6.1f}% "
          f"n={m.get('n_trades', 0):4d} E={ex.get('expectancy_bp', 0):+7.1f}bp win={ex.get('win_rate', 0) * 100:4.1f}%")
    res["equity"].to_parquet(ROOT / "data" / "bt" / f"gs_{label.replace(' ', '_').replace('=', '')}_equity.parquet")
    return m


print("\n===== OOS 2024-2026 门控内组合对照（10×10%，144bar/3ATR）=====")
bt("frozen_gate_基线", buys, p)

for q in (0.6, 0.8):
    sub = buys[buys.surv_score >= buys.surv_score.quantile(q)]
    bt(f"存活过滤_q{q}", sub, p)

# 卖点离场组：events = 买点 + 卖点（signal_exit=True）
p_exit = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=True)
bt("门控+全部卖点离场", pd.concat([buys, sells]), p_exit)
for q in (0.5, 0.8):
    hs = sells[sells.surv_score >= sells.surv_score.quantile(q)]
    bt(f"门控+高存活卖点_q{q}", pd.concat([buys, hs]), p_exit)

# 事件级参照：门控内买点的分数分位收益
print("\n===== 事件级（门控内，pol144_ret）=====")
for q in (0.5, 0.6, 0.7, 0.8):
    sub = buys[buys.surv_score >= buys.surv_score.quantile(q)]
    print(f"  top{int((1 - q) * 100):2d}%: n={len(sub):4d} E={sub.pol144_ret.mean() * 1e4:+7.1f}bp "
          f"win={(sub.pol144_ret > 0).mean() * 100:4.1f}%")
print(f"  全部门控内: n={len(buys)} E={buys.pol144_ret.mean() * 1e4:+.1f}bp")
