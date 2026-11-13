"""自定义摆动高低点事件生成器（全因果，不依赖 framework BSP）。

定义：bar t-k 是 [t-2k, t] 窗口最高/最低点 → t 时刻确认（右侧 k 根），t+1 开盘入场。
严格交替（低点事件后只认高点，反之亦然），不回溯替换（替换=未来函数）。
振幅过滤：距上一个反向确认摆动 >= amp × ATR(pivot)。

用法: python scripts/build_swing_events.py --codes BTCUSDT --freq 30m --k 3 --amp 1.5 --name swing_b30_k3
"""
import argparse
import time as _time

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chan_notes.data import store
from chan_notes.features import indicators, pa, patterns, brooks, context
from chan_notes.features.market import MarketContext
from chan_notes.labels.label import forward_labels
from chan_notes.core.policy import policy_labels


def detect_events(ind, k: int, amp: float, warmup: int = 200):
    """回撤确认之字转向（全因果，无未来函数）。

    状态机：SEEK_LOW 时追踪 running_min；当收盘从 running_min 反弹 >= amp*ATR(t)
    → 低点在 argmin bar 确认，t 时刻发出买事件。SEEK_HIGH 对称。
    k 参数保留兼容但不再使用（分形窗口被回撤确认取代）。
    返回 [(confirm_bar t, pivot_idx p, is_buy, pivot_price)]
    """
    h, l, c, atr = ind["high"], ind["low"], ind["close"], ind["atr14"]
    n = len(h)
    events = []
    state = 0  # 0=未定, 1=SEEK_HIGH(已有低点), -1=SEEK_LOW(已有高点)
    ext_price = 0.0   # 当前摆动的极值
    ext_idx = -1
    for t in range(warmup, n - 1):
        a = atr[t]
        if not (a == a) or a <= 0:
            continue
        if state >= 0:  # 未定或找高点：追踪运行最高
            if h[t] >= ext_price or state == 0:
                if state == 0:
                    # 初始：先建立参考
                    ext_price, ext_idx = float(h[t]), t
                    state = 1
                else:
                    ext_price, ext_idx = float(h[t]), t
            elif c[t] <= ext_price - amp * a:
                # 高点确认：从 running_max 回撤 amp*ATR
                events.append((t, ext_idx, 0, ext_price))
                state = -1
                ext_price, ext_idx = float(l[t]), t
        elif state == -1:  # 找低点
            if l[t] <= ext_price:
                ext_price, ext_idx = float(l[t]), t
            elif c[t] >= ext_price + amp * a:
                # 低点确认：从 running_min 反弹 amp*ATR
                events.append((t, ext_idx, 1, ext_price))
                state = 1
                ext_price, ext_idx = float(h[t]), t
    return events


def build(code: str, freq: str, k: int, amp: float, warmup: int = 200):
    df = store.load(code, freq)
    ind = indicators.precompute(df)
    times = df["time"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    opens = ind["open"]
    mkt = MarketContext(store.load("_MARKET", freq)) if store.has("_MARKET", freq) else None
    evs = detect_events(ind, k, amp, warmup)
    rows = []
    for (t, p, is_buy, px) in evs:
        entry = t + 1
        a = float(ind["atr14"][t])
        if not (a == a):
            a = 0.0
        feats = {}
        try:
            feats.update(pa.price_action_features(ind, t))
            feats.update(patterns.pattern_features(ind, t))
            feats.update(brooks.swing_features(ind, t))
            feats.update(brooks.regime_features(ind, t))
            feats.update(brooks.bar_count_features(ind, t))
            feats.update(brooks.breakout_strength_features(ind, t))
            feats.update(brooks.pattern2_features(ind, t))
            feats.update(context.context_features(ind, t))
        except Exception:
            continue
        r = {
            "code": code, "freq": freq, "bar_idx": t, "event_time": times[t],
            "bsp_bi_idx": -1, "bsp_type": "SWING", "is_buy": is_buy,
            "bsp_klu_idx": p, "delay_bars": t - p,
            "entry_idx": entry, "entry_time": times[entry], "entry_price": float(opens[entry]),
            "clear_bar": -1, "atr": a,
            "conf_ok": 1.0, "conf_wait": 0.0, "conf_entry_idx": float(entry),
            "conf_entry_time": times[entry], "conf_entry_price": float(opens[entry]),
            "conf_atr": a,
        }
        r.update(feats)
        if mkt is not None:
            r.update(mkt.features_at(times[t]))
        r.update(forward_labels(ind, entry, is_buy, a))
        r.update(policy_labels(ind, entry, is_buy, a))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="BTCUSDT")
    ap.add_argument("--freq", default="30m")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--amp", type=float, default=1.5)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    t0 = _time.time()
    frames = [build(c, args.freq, args.k, args.amp) for c in args.codes.split(",")]
    out = pd.concat(frames, ignore_index=True)
    from chan_notes.config import EVENTS
    p = EVENTS / f"{args.name}.parquet"
    out.to_parquet(p, index=False)
    buys = out[out.is_buy == 1]
    print(f"saved {p} rows={len(out)} cols={out.shape[1]} ({_time.time()-t0:.1f}s) "
          f"buy={len(buys)} sell={len(out)-len(buys)}")


if __name__ == "__main__":
    main()
