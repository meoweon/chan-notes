"""标签：前向收益 / 最大有利不利偏移 / 止损命中，全部在入场后计算。

统一口径（2026-11-03 修正，台账 #37）：
- 入场价 = 决策 bar 下一根的开盘价
- 持有 H 根，第 H 根收盘离场
- 成本 = 往返费率（默认 13bps），多空两侧都**扣除**
  （旧版 bug：卖侧 dir_ret = -q+c，高估 2c≈26bp）
- dir_ret = 买点取多头净收益，卖点取 -(价差收益)-成本
- mfe/mae 为物理有利/不利偏移：mfe≥0（买=向上/卖=向下的最大幅度），
  mae≤0（买=向下/卖=向上的最大幅度）
  （旧版 bug：卖侧 mfe/mae 符号写反）
- 数据末端不足 H 根 → 该档 5 个键全部 NaN（删失，不伪造完整标签）
"""
import numpy as np

HORIZONS = (8, 24, 48)     # 30分钟：约 1 / 3 / 6 个交易日
FORWARD_KEYS = tuple(f"{k}_{h}" for h in HORIZONS for k in ("ret", "dir_ret", "mfe", "mae", "stop"))


def forward_labels(ind: dict, entry_idx: int, is_buy: int, atr: float,
                   horizons=HORIZONS, cost: float = 0.0013, stop_atr: float = 2.0) -> dict:
    n = len(ind["close"])
    out = {}
    if entry_idx >= n:
        return out
    entry = ind["open"][entry_idx]
    if entry <= 0:
        return out
    sign = 1.0 if is_buy else -1.0
    stop_dist = stop_atr * atr / entry if entry > 0 else 0.0

    for H in horizons:
        full_end = entry_idx + H - 1
        end = min(full_end, n - 1)
        if end < entry_idx or end < full_end:   # 删失：观察窗不完整
            for k in ("ret", "dir_ret", "mfe", "mae", "stop"):
                out[f"{k}_{H}"] = np.nan
            continue
        hi = ind["high"][entry_idx:end + 1].max()
        lo = ind["low"][entry_idx:end + 1].min()
        close = ind["close"][end]
        q = close / entry - 1.0                 # 多头视角价差收益
        out[f"ret_{H}"] = q - cost
        out[f"dir_ret_{H}"] = sign * q - cost
        out[f"mfe_{H}"] = (hi / entry - 1.0) if is_buy else (1.0 - lo / entry)
        out[f"mae_{H}"] = (lo / entry - 1.0) if is_buy else -(hi / entry - 1.0)
        hit = (lo <= entry * (1 - stop_dist)) if is_buy else (hi >= entry * (1 + stop_dist))
        out[f"stop_{H}"] = 1.0 if hit else 0.0
    return out
