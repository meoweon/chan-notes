"""自适应退出 vs 固定 144bar 对照（OOS 2024-2026 + 冻结 frenzy 门控口径）。

变体（均保留 3ATR 止损，simulate_exit 同一引擎）：
- F144   固定 144 bar 时间退出（现任，pol144_ret 复用）
- CLEAR  结构失效退出：触发 BSP 被框架清除（clear_bar）时离场，上限 288 bar；
         clear_bar <= 入场 bar 的事件 = 入场前结构已证伪，整笔跳过（额外的入场过滤器）
- REGIME 环境退出：前一交易日 frenzy5 跌破冻结阈值（门控关闭）时离场，上限 144 bar

事件级对照 + 最优者的组合级（hold_col 动态持有）验证。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.core import policy
from chan_notes.eval.bars import load_frames
from chan_notes.eval.metrics import expectancy
from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"
THR = 0.00717  # IS(2020-2023) 冻结阈值

df = train.load("events_v7")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
df = df[df["event_time"] >= SPLIT]

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"]
fz_prev = fz.shift(1)
df["d"] = pd.to_datetime(df["event_time"]).dt.normalize()
df["fz"] = df["d"].map(fz_prev)
df = df.dropna(subset=["fz"])
gated = df[df.fz >= THR].copy()
print(f"OOS conf_buy {len(df)} / 门控后 {len(gated)}")

bars = load_frames(df.code.unique())
bars = {c: b[b["time"] >= "2023-11-01"].reset_index(drop=True) for c, b in bars.items()}  # 留足入场前缓冲

# 每股 bar 时间 -> 索引；bar_idx/conf_entry_idx/clear_bar 是原始全序列索引，需要偏移对齐
# events_v7 的索引基于完整 30m 序列（2020 起），此处切片会错位 —— 改用完整序列
bars = load_frames(df.code.unique())

codes = gated.code.unique()
print("模拟事件级退出 …")

# frenzy 日期 -> bar 映射（每股自己算）：需要每股 time 序列
rows = []
for code in codes:
    b = bars[code]
    ind = {k: b[k].to_numpy() for k in ("open", "high", "low", "close")}
    times = pd.to_datetime(b["time"])
    dnorm = times.dt.normalize()
    fz_bar = dnorm.map(fz_prev).to_numpy()  # 每根 bar 对应的前一交易日 frenzy
    for r in gated[gated.code == code].itertuples():
        e = int(r.conf_entry_idx)
        atr = float(r.conf_atr)
        if not (atr > 0) or e >= len(ind["close"]):
            continue
        rec = {"code": code, "event_time": r.event_time, "entry_idx": e}
        # F144：复用 pol144_ret（同一引擎同一参数）
        rec["ret_F144"] = float(r.pol144_ret)
        r144 = policy.simulate_exit(ind, e, 1, atr, hold=144, stop_atr=3.0)
        rec["bars_F144"] = r144.bars if r144 else np.nan
        # CLEAR：结构失效退出（cap 288）
        cb = int(r.clear_bar)
        if 0 < cb <= e:
            rec["ret_CLEAR"] = np.nan  # 入场前已证伪 → 不交易
            rec["bars_CLEAR"] = np.nan
            rec["skip_CLEAR"] = 1
        else:
            hold = 288 if cb == -1 else min(cb - e + 1, 288)
            rc = policy.simulate_exit(ind, e, 1, atr, hold=hold, stop_atr=3.0)
            rec["ret_CLEAR"] = rc.ret if rc else np.nan
            rec["bars_CLEAR"] = rc.bars if rc else np.nan
            rec["skip_CLEAR"] = 0
        # REGIME：门控关闭退出（cap 144）
        end = min(e + 144 - 1, len(ind["close"]) - 1)
        off = e + int(np.argmax(fz_bar[e:end + 1] < THR)) if (fz_bar[e:end + 1] < THR).any() else None
        hold_r = (off - e + 1) if off is not None else 144
        rr = policy.simulate_exit(ind, e, 1, atr, hold=hold_r, stop_atr=3.0)
        rec["ret_REGIME"] = rr.ret if rr else np.nan
        rec["bars_REGIME"] = rr.bars if rr else np.nan
        rows.append(rec)

res = pd.DataFrame(rows)
print(f"\n事件数 {len(res)}，CLEAR 入场前证伪跳过 {int(res.skip_CLEAR.sum())} 笔（{res.skip_CLEAR.mean()*100:.1f}%）")

print("\n=== 事件级对照（OOS 2024-2026，冻结门控子集）===")
out = []
for v in ("F144", "CLEAR", "REGIME"):
    r = res[f"ret_{v}"].dropna()
    ex = expectancy(r)
    out.append({
        "variant": v, "n": len(r), "E_bp": ex["expectancy_bp"], "win%": ex["win_rate"] * 100,
        "R": ex["payoff_R"], "avg_bars": res.loc[r.index, f"bars_{v}"].mean(),
        "top5%share%": ex["top5pct_profit_share"] * 100,
        "tail-1%E": r.sort_values().iloc[: int(len(r) * 0.99)].mean() * 1e4,
        "tail-5%E": r.sort_values().iloc[: int(len(r) * 0.95)].mean() * 1e4,
    })
a = pd.DataFrame(out)
print(a.round(1).to_string(index=False))
res.to_parquet(ROOT / "data" / "bt" / "adaptive_exit_events.parquet", index=False)
a.to_csv(ROOT / "reports" / "adaptive_exit.csv", index=False)
print("saved reports/adaptive_exit.csv + data/bt/adaptive_exit_events.parquet")
