"""冻结 IS 阈值的 frenzy 门控——真样本外检验（2024-01 起）。

动机（frenzy_portfolio.py 结果）：静态全样本中位数门控 OOF 很强（+87.9%/夏普0.67），
但阈值含未来信息；滚动中位数门控反而最差——滚动中位数度量"相对近期热"，
而策略需要的是"绝对热"（妖股在场）。本脚本用 IS 期（2020-2023）中位数冻结阈值，
在从未参与阈值估计的 OOS 期（2024-2026）检验。
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
print(f"IS(2020-2023) frenzy5 中位数冻结阈值: {thr:.5f}")

df = train.load("events_v7")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
df = df[df["event_time"] >= SPLIT]
df["d"] = pd.to_datetime(df["event_time"]).dt.normalize()
df["fz"] = df["d"].map(fz_prev)
df = df.dropna(subset=["fz"])
gate = df[df.fz >= thr]
print(f"OOS(2024-2026) conf_buy: 全体 {len(df)} / 门控后 {len(gate)}（开机率 {len(gate)/len(df)*100:.0f}%）")

bars = load_frames(df.code.unique())
bars = {c: b[b["time"] >= SPLIT] for c, b in bars.items()}
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=3.0, hold_bars=144, signal_exit=False)

rows = []
for tag, sub in (("always_on", df), ("frozen_gate", gate)):
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
    res["trades"].to_parquet(ROOT / "data" / "bt" / f"frozen_{tag}_trades.parquet", index=False)
    res["equity"].to_parquet(ROOT / "data" / "bt" / f"frozen_{tag}_equity.parquet")

a = pd.DataFrame(rows)
print(a.round(2).to_string(index=False))
a.to_csv(ROOT / "reports" / "frenzy_frozen_oos.csv", index=False)
print(f"saved {ROOT / 'reports' / 'frenzy_frozen_oos.csv'}")
