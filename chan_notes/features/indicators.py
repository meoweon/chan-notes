"""因果指标预计算：整条序列一次算完，第 i 个值只用 0..i 的数据。"""
import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def precompute(df: pd.DataFrame) -> dict:
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()
    vol = df["volume"].astype(float).to_numpy()

    cs, hs, ls = pd.Series(close), pd.Series(high), pd.Series(low)
    prev_close = cs.shift(1)

    # ATR / 波动
    tr = pd.concat([hs - ls, (hs - prev_close).abs(), (ls - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # MACD
    dif = _ema(cs, 12) - _ema(cs, 26)
    dea = _ema(dif, 9)
    macd_bar = 2 * (dif - dea)

    # RSI (Wilder)
    delta = cs.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi14 = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # BOLL
    ma20 = cs.rolling(20).mean()
    std20 = cs.rolling(20).std(ddof=0)
    boll_up, boll_dn = ma20 + 2 * std20, ma20 - 2 * std20

    # 均线
    ma5, ma10, ma60 = cs.rolling(5).mean(), cs.rolling(10).mean(), cs.rolling(60).mean()

    # 区间（不含当前 bar）
    hh20 = hs.shift(1).rolling(20).max()
    ll20 = ls.shift(1).rolling(20).min()
    hh50 = hs.shift(1).rolling(50).max()
    ll50 = ls.shift(1).rolling(50).min()

    # 20 bar 线性回归（斜率、R²），窗口末端为当前 bar
    n = 20
    x = np.arange(n, dtype=float)
    sx, sxx = x.sum(), (x ** 2).sum()
    conv_k = x[::-1].copy()
    sxy = pd.Series(np.convolve(close, conv_k, mode="full")[: len(close)])
    sy = cs.rolling(n).sum()
    syy = (cs ** 2).rolling(n).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    cov = n * sxy - sx * sy
    varx = n * sxx - sx ** 2
    vary = n * syy - sy ** 2
    r2 = (cov ** 2) / (varx * vary.replace(0, np.nan))

    # 成交量
    vma5, vma20 = pd.Series(vol).rolling(5).mean(), pd.Series(vol).rolling(20).mean()

    def arr(s):
        return s.to_numpy(dtype=float)

    # ── Brooks 背景层预计算（全部因果）──
    n = len(close)
    # Kaufman 效率系数：净位移 / 路径总长（趋势 vs 区间的一阶度量）
    diff1 = np.abs(np.diff(close, prepend=close[0]))
    dsum = pd.Series(diff1)

    def _er(win):
        net = (cs - cs.shift(win)).abs()
        den = dsum.rolling(win).sum()
        return (net / den.replace(0, np.nan)).to_numpy(dtype=float)

    er20, er60 = _er(20), _er(60)

    # 市场周期阶段：±2=趋势(spike/channel 强)、±1=弱趋势/通道、0=区间(barb-wire)；符号=60bar 方向
    dirn60 = np.sign((cs - cs.shift(60)).fillna(0.0).to_numpy())
    stage = np.where(er60 >= 0.45, 2.0, np.where(er60 >= 0.25, 1.0, 0.0)) * dirn60

    # 当前阶段已持续的 bar 数
    regime_age = np.zeros(n)
    for i in range(1, n):
        regime_age[i] = regime_age[i - 1] + 1 if stage[i] == stage[i - 1] else 0.0

    # TTR（紧密交易区间）：20bar 宽度 < 120bar 中位宽度的 0.7 倍
    width20 = (hh20 - ll20) / cs
    med_w = width20.rolling(120, min_periods=20).median()
    ttr = (width20 < 0.7 * med_w).fillna(False).to_numpy()
    ttr_bars = np.zeros(n)
    for i in range(1, n):
        ttr_bars[i] = ttr_bars[i - 1] + 1 if ttr[i] else 0.0

    # 高潮：TR > 2*ATR 且实体占比 > 0.6；连续同向计数（相邻链）+ 距上次高潮的签名 bar 数
    rng_hl = (hs - ls).replace(0, np.nan)
    body_ratio = ((cs - pd.Series(open_)).abs() / rng_hl).fillna(0.0)
    is_climax = ((tr > 2 * atr14) & (body_ratio > 0.6)).fillna(False).to_numpy()
    cdir = np.sign((cs - pd.Series(open_)).fillna(0.0).to_numpy())
    climax_cnt = np.zeros(n)
    climax_age = np.zeros(n)
    last_dir, last_idx, chain = 0.0, -1, 0.0
    for i in range(n):
        if is_climax[i]:
            d = cdir[i] if cdir[i] != 0 else (last_dir if last_dir != 0 else 1.0)
            if d == last_dir and last_idx == i - 1:
                chain += d
            else:
                chain = d
            last_dir, last_idx = d, i
            climax_cnt[i] = chain
            climax_age[i] = 0.0
        else:
            if last_idx >= 0:
                climax_age[i] = (i - last_idx) * (last_dir if last_dir != 0 else 1.0)

    out = {
        "close": close, "open": open_, "high": high, "low": low, "vol": vol,
        "atr14": arr(atr14), "tr": arr(tr),
        "dif": arr(dif), "dea": arr(dea), "macd_bar": arr(macd_bar),
        "rsi14": arr(rsi14),
        "ma5": arr(ma5), "ma10": arr(ma10), "ma20": arr(ma20), "ma60": arr(ma60),
        "std20": arr(std20), "boll_up": arr(boll_up), "boll_dn": arr(boll_dn),
        "hh20": arr(hh20), "ll20": arr(ll20), "hh50": arr(hh50), "ll50": arr(ll50),
        "slope20": arr(slope), "r2_20": arr(r2),
        "vma5": arr(vma5), "vma20": arr(vma20),
        "range20": arr((hh20 - ll20) / cs),
        "er20": er20, "er60": er60, "cycle_stage": stage, "regime_age": regime_age,
        "ttr": ttr.astype(float), "ttr_bars": ttr_bars,
        "climax_cnt": climax_cnt, "climax_age": climax_age,
    }

    # 日线背景广播：bar 属于第 k 天时，只用第 k-1 天及之前的已完成日线统计（严格无泄漏）
    if "time" in df.columns:
        dates = pd.to_datetime(df["time"]).dt.normalize()
        day_codes, _ = pd.factorize(dates)
        gc = pd.Series(close).groupby(day_codes)
        gh = pd.Series(high).groupby(day_codes)
        gl = pd.Series(low).groupby(day_codes)
        dc, dh, dl = gc.last(), gh.max(), gl.min()
        d_ma20 = dc.rolling(20).mean()
        d_slope5 = ((d_ma20 - d_ma20.shift(5)) / dc.replace(0, np.nan))
        d_hh60 = dh.rolling(60).max()
        d_ll60 = dl.rolling(60).min()
        ddiff = dc.diff().abs()
        d_er20 = ((dc - dc.shift(20)).abs() / ddiff.rolling(20).sum().replace(0, np.nan))

        def prev_day(s):
            return s.shift(1).to_numpy(dtype=float)[day_codes]

        out["d_ma_slope5"] = prev_day(d_slope5)
        out["d_hh60"] = prev_day(d_hh60)
        out["d_ll60"] = prev_day(d_ll60)
        out["d_er20"] = prev_day(d_er20)
    else:
        for k in ("d_ma_slope5", "d_hh60", "d_ll60", "d_er20"):
            out[k] = np.full(n, np.nan)

    return out
