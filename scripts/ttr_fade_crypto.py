"""TTR fade scalp 在加密 1h 上的复检（台账 #39）。

背景：#32/#35 在 A 股把 TTR fade 判死，死因是"区间内 ATR 太小，喂不饱 13bp 成本"。
加密的单 bar 波动更大、成本更低（taker 约 10bp/边），所以这是一次有条件复检，
不是重开已死方向。

预注册口径（跑之前写死，不改）：
  信号：1h 上 TTR = width20 < 0.7 * rolling120 中位 width；区间够宽 = (hh20-ll20) >= 2.5*ATR14；
        贴下沿 = pos = (close-ll20)/(hh20-ll20) <= 0.25。次 bar 开盘入场（无未来函数）。
  出场（scalp，三选一先到）：
        目标 = ll20 + 0.5*(hh20-ll20)（区间中点）；要求目标距入场 >= 0.2%，否则跳过
        止损 = 入场 - stop_mult*ATR（stop_mult 取 1.0 / 1.5）
        时间 = max_hold 根 bar（24 / 48 / 96）后收盘离场
        同一根 bar 同时触及止损与目标时，按止损处理（保守）
  成本：单边 tc_bp，净收益已扣 2*tc
  切分：IS = 2020-10~2023-12；OOS = 2024-01~2026-10
  对照：
    ① 安慰剂：在同样的 TTR&够宽 bar 池里随机入场，同出场逻辑，20 次取均值
    ② 非 TTR 下沿 fade：pos<=0.25 且 TTR=0，隔离 TTR 本身有没有增量
  判据：OOS 净 E 显著为正，且 信号 − 安慰剂 > 成本量级（>=10bp），才算存活。
        只做事件级 discovery，不做组合级主张（重叠持仓未处理）。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

BARS = Path("data/bars/1h")
SPLIT = pd.Timestamp("2024-01-01")


def load_bars() -> list[tuple[str, pd.DataFrame]]:
    out = []
    for f in sorted(glob.glob(str(BARS / "*USDT.parquet"))):
        b = pd.read_parquet(f).sort_values("time").reset_index(drop=True)
        out.append((Path(f).stem, b))
    return out


def indicators(b: pd.DataFrame):
    h, l, c = b.high.to_numpy(float), b.low.to_numpy(float), b.close.to_numpy(float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    hh20 = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
    ll20 = pd.Series(l).rolling(20, min_periods=20).min().to_numpy()
    width = (hh20 - ll20) / c
    med = pd.Series(width).rolling(120, min_periods=20).median().to_numpy()
    ttr = width < 0.7 * med
    wide = (hh20 - ll20) >= 2.5 * atr
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (c - ll20) / (hh20 - ll20)
    return atr, hh20, ll20, ttr, wide, pos


def run_trade(o, h, l, c, atr, hh20, ll20, i, stop_mult, max_hold):
    """i = 信号 bar，次 bar 开盘入场。返回净前收益（比例）或 None。"""
    e_i = i + 1
    if e_i >= len(c) - 1:
        return None
    entry = o[e_i]
    rng = hh20[i] - ll20[i]
    target = ll20[i] + 0.5 * rng
    if not (target / entry - 1.0 >= 0.002):
        return None
    stop = entry - stop_mult * atr[i]
    end = min(e_i + max_hold, len(c))
    for j in range(e_i, end):
        if l[j] <= stop:            # 保守：同 bar 先判止损
            return stop / entry - 1.0
        if h[j] >= target:
            return target / entry - 1.0
    return c[end - 1] / entry - 1.0


def collect(data, idxs, stop_mult, max_hold):
    rows = []
    for code, b, o, h, l, c, atr, hh20, ll20 in data:
        t = b.time.to_numpy()
        for i in idxs.get(code, []):
            r = run_trade(o, h, l, c, atr, hh20, ll20, i, stop_mult, max_hold)
            if r is not None:
                rows.append((code, pd.Timestamp(t[i]), r))
    return pd.DataFrame(rows, columns=["code", "t", "ret"])


def stats(df: pd.DataFrame, tc_bp: float) -> dict:
    if len(df) == 0:
        return dict(n=0)
    net = df.ret - 2 * tc_bp / 1e4
    win = net[net > 0]
    loss = net[net <= 0]
    return dict(
        n=len(net),
        E_bp=net.mean() * 1e4,
        win=len(win) / len(net) * 100,
        R=(win.mean() / abs(loss.mean())) if len(loss) and loss.mean() != 0 else np.nan,
        total=(np.prod(1 + net) - 1) * 100,
    )


def main():
    data = []
    pools, sig_ttr, sig_nottr = {}, {}, {}
    atr_ttr, atr_all = [], []
    for code, b in load_bars():
        o, h, l, c = (b[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        atr, hh20, ll20, ttr, wide, pos = indicators(b)
        ok = np.isfinite(atr) & np.isfinite(pos) & np.isfinite(hh20) & wide
        atr_pct = atr / c * 1e4
        atr_all.append(atr_pct[ok])
        atr_ttr.append(atr_pct[ok & ttr])
        pools[code] = np.where(ok & ttr)[0]
        sig_ttr[code] = np.where(ok & ttr & (pos <= 0.25))[0]
        sig_nottr[code] = np.where(ok & (~ttr) & (pos <= 0.25))[0]
        data.append((code, b, o, h, l, c, atr, hh20, ll20))

    print("=== 环境：ATR%% 中位数（1h） ===")
    print("全样本 %.1fbp | TTR 内 %.1fbp" % (np.median(np.concatenate(atr_all)),
                                             np.median(np.concatenate(atr_ttr))))
    print("TTR&够宽 bar 数 = %d" % sum(len(v) for v in pools.values()))

    rng = np.random.default_rng(42)
    for stop_mult in (1.0, 1.5):
        for max_hold in (24, 48, 96):
            for tc in (10,):
                sig = collect(data, sig_ttr, stop_mult, max_hold)
                nottr = collect(data, sig_nottr, stop_mult, max_hold)
                plac = []
                for _ in range(20):
                    ridx = {k: rng.choice(v, size=max(1, len(sig_ttr[k])), replace=False).tolist()
                            for k, v in pools.items() if len(v)}
                    p = collect(data, ridx, stop_mult, max_hold)
                    plac.append(stats(p, tc))
                pd_ = pd.DataFrame(plac).dropna()

                for tag, df in (("TTR下沿", sig), ("非TTR下沿", nottr)):
                    for seg, sub in (("IS", df[df.t < SPLIT]), ("OOS", df[df.t >= SPLIT])):
                        s = stats(sub, tc)
                        if s["n"] == 0:
                            continue
                        print(f"stop={stop_mult} hold={max_hold} {tag}-{seg}: "
                              f"n={s['n']:5d} E={s['E_bp']:+7.1f}bp 胜率={s['win']:4.1f}% "
                              f"R={s['R']:4.2f} 累计={s['total']:+8.1f}%")
                print(f"    安慰剂(同池随机)      : n≈{int(pd_.n.mean()):5d} "
                      f"E={pd_.E_bp.mean():+7.1f}bp 胜率={pd_.win.mean():4.1f}% 累计={pd_.total.mean():+8.1f}%")
                print()


if __name__ == "__main__":
    main()
