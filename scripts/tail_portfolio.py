"""#33 组合级终审：冻结 frenzy 门控 + 尾部分数过滤（OOS 2024-2026）。
变体：frozen_gate（#19 基线）vs frozen_gate + tail score top20%。
尾部分数来自 wf_tail30（OOF 2021-02 起，12m 滑动，mcw=2+spw=52 解瘫痪）。
注意：分位数阈值在 OOS 窗口内全局计算（与 #18/#22 同口径，轻度前视，已声明）。
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
SPLIT = "2024-01-01"

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"]
fz_prev = fz.shift(1)
thr = fz[fz.index < SPLIT].median()

df = train.load("events_v7")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
df = df[df["event_time"] >= SPLIT]
df["d"] = pd.to_datetime(df["event_time"]).dt.normalize()
df["fz"] = df["d"].map(fz_prev)
df = df.dropna(subset=["fz"])
gate = df[df.fz >= thr].copy()

sc = pd.read_parquet(ROOT / "data" / "events" / "wf_tail30.parquet")[["code", "event_time", "score"]]
gate = gate.merge(sc, on=["code", "event_time"], how="left")
print(f"门控内 {len(gate)} 事件，带尾部分数 {gate.score.notna().sum()}")

q80 = gate.score.quantile(0.8)
q60 = gate.score.quantile(0.6)
top20 = gate[gate.score >= q80]
top40 = gate[gate.score >= q60]

bars = load_frames(df.code.unique())
bars = {c: b[b["time"] >= SPLIT] for c, b in bars.items()}
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=False)

rows = []
for tag, sub in (("gate_all", gate), ("gate_tail_q60", top40), ("gate_tail_q80", top20)):
    res = portfolio.run(sub, bars, p, entry_mode="confirm")
    m = res["metrics"]
    e = expectancy(res["trades"]["ret"])
    rows.append({
        "variant": tag, "trades": m["n_trades"],
        "total%": m["total_ret"] * 100, "cagr%": m["cagr"] * 100,
        "sharpe": m["sharpe"], "mdd%": m["mdd"] * 100,
        "E_bp": e["expectancy_bp"], "win%": e["win_rate"] * 100,
        "R": e["payoff_R"], "kelly%": e["kelly"] * 100,
    })
    res["trades"].to_parquet(ROOT / "data" / "bt" / f"tail_{tag}_trades.parquet", index=False)
    res["equity"].to_parquet(ROOT / "data" / "bt" / f"tail_{tag}_equity.parquet")

a = pd.DataFrame(rows)
print(a.round(2).to_string(index=False))
a.to_csv(ROOT / "reports" / "tail_portfolio.csv", index=False)
print(f"saved {ROOT / 'reports' / 'tail_portfolio.csv'}")
