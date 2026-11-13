"""regime 切换系统终审：热=个股彩票策略（frozen_gate 权益），冷=_MARKET 指数。
日频拼接：gate-on 日取策略日收益，gate-off 日取指数日收益。OOS 2024-2026。
注意：切换按日粒度近似（策略持仓跨切换日的过渡简化为日切换），已声明。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"

eq = pd.read_parquet(ROOT / "data" / "bt" / "frozen_frozen_gate_equity.parquet")
eq.index = pd.to_datetime(eq.index)
eq = eq[eq.index >= SPLIT].sort_index()
sd = eq["equity"].resample("1D").last().dropna()
r_strat = sd.pct_change()

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"].shift(1)
thr = fz[fz.index < SPLIT].median()
hot = (fz >= thr)

mk = pd.read_parquet(ROOT / "data" / "bars" / "d" / "_MARKET.parquet")
mk["time"] = pd.to_datetime(mk["time"])
r_idx = mk.set_index("time")["close"].pct_change()

days = r_strat.index.intersection(r_idx.index)
r_strat, r_idx = r_strat.loc[days], r_idx.loc[days]
hot = hot.reindex(days).fillna(False)

r_sys = pd.Series(np.where(hot, r_strat.fillna(0), r_idx.fillna(0)), index=days)
r_base = r_strat.fillna(0)
r_bh = r_idx.fillna(0)

def stats(r, name):
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    yrs = len(r) / 242
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(242)
    mdd = ((eq / eq.cummax()) - 1).min()
    print(f"{name:<22s}: 总 {total*100:>+6.1f}%  CAGR {cagr*100:>5.1f}%  Sharpe {sharpe:>4.2f}  MDD {mdd*100:>6.1f}%")

stats(r_base, "基线(热彩票+冷空仓)")
stats(r_sys, "regime切换(热彩票+冷指数)")
stats(r_bh, "全程指数(对照)")
out = pd.DataFrame({"d": days, "hot": hot, "r_base": r_base, "r_sys": r_sys})
out.to_csv(ROOT / "reports" / "regime_switch_daily.csv", index=False)
print(f"\nsaved {ROOT / 'reports' / 'regime_switch_daily.csv'}  热天 {hot.sum()} / 冷天 {(~hot).sum()}")
