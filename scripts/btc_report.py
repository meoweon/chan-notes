"""BTC 策略成绩单：净值曲线级指标（总收益/年化/夏普/最大回撤/胜率/盈亏比/利润因子/Kelly）。
规则：单资产顺序执行——一仓未平不开新仓；每笔全仓复利；成本 12bp 已含在 ret 内。
对象：摆动点发生器 30m（a1.5）买侧/卖侧做空/多空合并，最优格 576/3.0；对照=买入持有。
"""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["CHAN_NOTES_BARS"] = str(Path(__file__).resolve().parents[1] / "data" / "bars_crypto")  # 须在 import chan_notes 前
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many

COST = 0.0012
HOLD, STOP = 576, 3.0
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])
n_bars = len(b["close"])
times = pd.to_datetime(pd.read_parquet(
    Path(__file__).resolve().parents[1] / "data" / "bars_crypto" / "30m" / "BTCUSDT.parquet")["time"])

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")

def trades_for(side):
    out = []
    for r in df[df.is_buy == side].itertuples():
        res = simulate_exit(b, int(r.conf_entry_idx), side, float(r.atr), hold=HOLD, stop_atr=STOP, cost=COST)
        if res is not None:
            eb = int(r.conf_entry_idx) + res.bars - 1
            out.append({"entry_bar": int(r.conf_entry_idx), "exit_bar": eb,
                        "entry_time": times[int(r.conf_entry_idx)], "exit_time": times[eb],
                        "ret": res.ret})
    return pd.DataFrame(out)

def run(trades, start=None, end=None):
    """顺序执行：跳过与持仓重叠的信号；逐笔全仓复利。"""
    t = trades.sort_values("entry_bar")
    if start is not None:
        t = t[(t.entry_time >= start) & (t.entry_time < end)]
    taken, busy_until = [], -1
    for r in t.itertuples():
        if r.entry_bar < busy_until:
            continue
        taken.append(r)
        busy_until = r.exit_bar
    if not taken:
        return None
    taken = pd.DataFrame([x._asdict() for x in taken])
    rets = taken["ret"].values
    eq = np.cumprod(1 + rets)
    years = (taken["exit_time"].iloc[-1] - taken["entry_time"].iloc[0]).days / 365.25
    total = eq[-1] - 1
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    tpy = len(rets) / years if years > 0 else 0
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(tpy)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    w, l = rets[rets > 0], rets[rets <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    rr = w.mean() / abs(l.mean()) if len(l) and l.mean() != 0 else float("nan")
    p = len(w) / len(rets)
    kelly = p - (1 - p) / rr if rr and rr > 0 else float("nan")
    return {"笔数": len(rets), "跳过": len(t) - len(rets), "总收益": total, "年化": cagr,
            "夏普": sharpe, "最大回撤": mdd, "胜率": p, "盈亏比": rr,
            "每笔期望bp": rets.mean() * 1e4, "利润因子": pf, "Kelly": kelly}

def show(name, s):
    print(f"{name:<14s} 笔数{s['笔数']:>4d}(跳{s['跳过']:>4d}) 总收益{s['总收益']*100:>+8.1f}% 年化{s['年化']*100:>+6.1f}% "
          f"夏普{s['夏普']:>5.2f} 回撤{s['最大回撤']*100:>6.1f}% 胜率{s['胜率']*100:>4.1f}% "
          f"盈亏比{s['盈亏比']:>4.2f} 期望{s['每笔期望bp']:>+6.1f}bp PF{s['利润因子']:>5.2f} Kelly{s['Kelly']*100:>5.1f}%")

buy, sell = trades_for(1), trades_for(0)
both = pd.concat([buy, sell])

bh0, bh1 = b["close"][0], b["close"][-1]
years = (times.iloc[-1] - times.iloc[0]).days / 365.25
print(f"买入持有对照: 总收益{(bh1/bh0-1)*100:+,.0f}%  年化{((bh1/bh0)**(1/years)-1)*100:+.1f}%\n")

for nm, tr in (("买侧(做多)", buy), ("卖侧(做空)", sell), ("多空合并", both)):
    for tag, st, en in (("全期", None, None), ("样本内20-23", "2020-01-01", "2024-01-01"), ("样本外24-26", "2024-01-01", "2027-01-01")):
        s = run(tr, pd.Timestamp(st) if st else None, pd.Timestamp(en) if en else None)
        if s:
            show(f"{nm}/{tag}", s)
    print()
