"""TTR fade scalp（Brooks 区间打法）终审：range+冷 regime 格子内，个股 TTR 中触及下沿 1/3 做多。
纪律：信号与安慰剂同格对照；IS(2020-2023) 选参数，OOS(2024-2026) 只评一次；每股信号去重（持有期内不再触发）。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.data import store
from chan_notes.eval.bars import for_policy
from chan_notes.features import indicators

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"
COST = 0.0013
ZONE = 1 / 3  # 区间下沿 1/3 为买区

rg = pd.read_csv(ROOT / "reports" / "market_regime.csv", parse_dates=["d"]).set_index("d")["state"].shift(1)
fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"].shift(1)
thr = fz[fz.index < SPLIT].median()

codes = [c for c in store.list_codes("30m") if not c.startswith("_")]
rng = np.random.default_rng(7)

sig_events, rnd_events = [], []
for ci, code in enumerate(codes):
    df = store.load(code, "30m")
    if len(df) < 500:
        continue
    ind = indicators.precompute(df)
    close, atr = ind["close"], ind["atr14"]
    hh20p = pd.Series(ind["hh20"]).shift(1).to_numpy()
    ll20p = pd.Series(ind["ll20"]).shift(1).to_numpy()
    width = hh20p - ll20p
    in_zone = close <= ll20p + ZONE * width
    cell = (ind["ttr"] == 1) & (ind["ttr_bars"] >= 10) & np.isfinite(atr) & (width > 0)

    days = pd.to_datetime(df["time"]).dt.normalize()
    st = days.map(rg).to_numpy()
    fz_v = days.map(fz).to_numpy()
    cold_range = (st == "range") & (fz_v < thr)

    sig_mask = cell & in_zone & cold_range
    sig_idx = np.flatnonzero(sig_mask)
    pool_idx = np.flatnonzero(cell & cold_range & ~sig_mask)

    # 每股去重：信号后 24 bar 内不再触发；安慰剂同步抽样（2× 信号数）
    taken, last = [], -10**9
    for i in sig_idx:
        if i - last > 24 and i + 30 < len(df):
            taken.append(i)
            last = i
    if taken:
        bars = for_policy(df)
        for i in taken:
            sig_events.append((code, days.iloc[i], i, float(atr[i])))
        k = min(len(pool_idx), 2 * len(taken))
        if k > 0:
            for i in rng.choice(pool_idx, size=k, replace=False):
                if i + 30 < len(df) and np.isfinite(atr[i]):
                    rnd_events.append((code, days.iloc[i], int(i), float(atr[i]), bars))
    if (ci + 1) % 60 == 0:
        print(f"  [{ci+1}/{len(codes)}] sig={len(sig_events)} rnd={len(rnd_events)}", flush=True)

print(f"\n信号 {len(sig_events)} / 安慰剂 {len(rnd_events)}")

def run_set(events, with_bars=None, hold=24, stop=1.0):
    out = {"IS": [], "OOS": []}
    cache = {}
    for ev in events:
        code, d, i, a = ev[:4]
        if code not in cache:
            cache[code] = for_policy(store.load(code, "30m"))
        r = simulate_exit(cache[code], i + 1, 1, a, hold=hold, stop_atr=stop, cost=COST)
        if r is None:
            continue
        out["IS" if d < pd.Timestamp(SPLIT) else "OOS"].append(r.ret)
    return {k: np.array(v) for k, v in out.items()}

print(f"\n{'组':<10s}{'H':>4s}{'stop':>5s} | {'IS n':>6s}{'IS E':>8s}{'IS win':>7s} | {'OOS n':>6s}{'OOS E':>8s}{'OOS win':>7s}")
for hold in (12, 24):
    for stop in (1.0, 1.5):
        for tag, evs in (("信号", sig_events), ("安慰剂", rnd_events)):
            res = run_set(evs, hold=hold, stop=stop)
            i1, i2 = res["IS"], res["OOS"]
            if len(i1) == 0 or len(i2) == 0:
                continue
            print(f"{tag:<10s}{hold:>4d}{stop:>5.1f} | {len(i1):>6d}{i1.mean()*1e4:>+8.1f}{(i1>0).mean()*100:>6.1f}% | "
                  f"{len(i2):>6d}{i2.mean()*1e4:>+8.1f}{(i2>0).mean()*100:>6.1f}%")
