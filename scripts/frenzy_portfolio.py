"""frenzy regime 门控的组合级验证（本研究线最后一个开放假设）。

假设（来自台账 #8/#11/#18）：策略利润来自狂潮期暴露，入场过滤削减暴露有害，
正确用法是"什么时候开机"——低迷期关机、狂潮期满仓。

无未来函数设计：
- 事件（30m）用**前一交易日**的 frenzy5 值门控（当日值决策时未知）；
- 阈值两口径：静态全样本中位数（含泄漏，仅参考） vs 滚动 252 日中位数（PIT，主口径）。

对比：always-on 基线 / 静态门控 / 滚动门控，OOF 窗口 2021-02-01 起。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.eval.bars import load_frames
from chan_notes.eval.metrics import expectancy
from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]
START = "2021-02-01"

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"]
fz_prev = fz.shift(1)                                  # 前一交易日值（PIT）
fz_roll_med = fz.rolling(252, min_periods=60).median().shift(1)  # 滚动中位数（PIT）
fz_static_med = fz.median()

df = train.load("events_v7")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
df = df[df["event_time"] >= START]
df["d"] = pd.to_datetime(df["event_time"]).dt.normalize()
df["fz"] = df["d"].map(fz_prev)
df["fz_med"] = df["d"].map(fz_roll_med)
df = df.dropna(subset=["fz"])
print(f"OOF conf_buy events: {len(df)}")

variants = {
    "always_on": df,
    "static_gate": df[df.fz >= fz_static_med],
    "rolling_gate": df[df.fz >= df.fz_med],
}
uptime = {
    "always_on": 1.0,
    "static_gate": (df.fz >= fz_static_med).mean(),
    "rolling_gate": (df.fz >= df.fz_med).mean(),
}

bars = load_frames(df.code.unique())
bars = {c: b[b["time"] >= START] for c, b in bars.items()}
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=False)

rows = []
for tag, sub in variants.items():
    res = portfolio.run(sub, bars, p, entry_mode="confirm")
    m = res["metrics"]
    e = expectancy(res["trades"]["ret"]) if not res["trades"].empty else {}
    rows.append({
        "variant": tag, "trades": m.get("n_trades"), "uptime%": uptime[tag] * 100,
        "total%": m.get("total_ret", 0) * 100, "cagr%": m.get("cagr", 0) * 100,
        "sharpe": m.get("sharpe"), "mdd%": m.get("mdd", 0) * 100,
        "E_bp": e.get("expectancy_bp"), "win%": e.get("win_rate", 0) * 100,
        "R": e.get("payoff_R"), "kelly%": e.get("kelly", 0) * 100,
    })
    res["trades"].to_parquet(ROOT / "data" / "bt" / f"frenzy_{tag}_trades.parquet", index=False)
    res["equity"].to_parquet(ROOT / "data" / "bt" / f"frenzy_{tag}_equity.parquet")

a = pd.DataFrame(rows)
print(a.round(2).to_string(index=False))
a.to_csv(ROOT / "reports" / "frenzy_portfolio.csv", index=False)
print(f"saved {ROOT / 'reports' / 'frenzy_portfolio.csv'}")
