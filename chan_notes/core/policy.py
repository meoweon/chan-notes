"""统一的退出规则模拟。

标签、评估脚本、组合回测共用这一份实现，避免同一规则被复制多份后走样。

口径（2026-11-03 修正，台账 #37）：
- ret 按**信号方向**取号并扣成本：
  做多 = 平仓价/入场价 - 1 - cost
  做空 = -(平仓价/入场价 - 1) - cost（做空成本必须再扣一次，而非加回）
- 数据末端不足持有期且未触止损 → reason="censored"（删失），调用方必须剔除，
  不得当作完整时间退出标签（旧版 bug：一律返回 "time"）。
"""
from dataclasses import dataclass

DEFAULT_COST = 0.0013      # 往返成本（佣金+印花税+滑点粗估）
DEFAULT_HOLD = 24
DEFAULT_STOP_ATR = 2.0
POLICY_KEYS = ("pol_ret", "pol_dir_ret", "pol_bars", "pol_stop")


@dataclass(frozen=True)
class ExitResult:
    ret: float        # 按信号方向取号的净收益（含成本；做空已取负并扣成本）
    bars: int         # 实际持有 bar 数
    reason: str       # stop / trail / time / censored


def simulate_exit(ind: dict, entry_idx: int, is_buy: int, atr: float,
                  hold: int = DEFAULT_HOLD, stop_atr: float = DEFAULT_STOP_ATR,
                  trail_atr: float = 0.0, cost: float = DEFAULT_COST) -> ExitResult | None:
    """按「入场后先触止损则止损、否则第 hold 根收盘离场」模拟单笔交易。

    trail_atr > 0 时启用移动止损：用**已收 bar** 的极值更新止损，
    再在下一根检查是否触发（避免同一根内的未来信息）。
    """
    n = len(ind["close"])
    if entry_idx >= n or entry_idx < 0:
        return None
    entry = ind["open"][entry_idx]
    if entry <= 0:
        return None
    sign = 1.0 if is_buy else -1.0
    stop = entry - stop_atr * atr if is_buy else entry + stop_atr * atr
    peak = ind["high"][entry_idx] if is_buy else ind["low"][entry_idx]
    full_end = entry_idx + hold - 1
    end = min(full_end, n - 1)
    for i in range(entry_idx, end + 1):
        if is_buy and ind["low"][i] <= stop:
            raw = stop / entry - 1.0
            return ExitResult(sign * raw - cost, i - entry_idx + 1,
                              "trail" if trail_atr > 0 and stop > entry - stop_atr * atr else "stop")
        if (not is_buy) and ind["high"][i] >= stop:
            raw = stop / entry - 1.0
            return ExitResult(sign * raw - cost, i - entry_idx + 1,
                              "trail" if trail_atr > 0 and stop < entry + stop_atr * atr else "stop")
        if trail_atr > 0:
            if is_buy:
                peak = max(peak, ind["high"][i])
                stop = max(stop, peak - trail_atr * atr)
            else:
                peak = min(peak, ind["low"][i])
                stop = min(stop, peak + trail_atr * atr)
    raw = ind["close"][end] / entry - 1.0
    return ExitResult(sign * raw - cost, end - entry_idx + 1,
                      "time" if end >= full_end else "censored")


def policy_labels(ind: dict, entry_idx: int, is_buy: int, atr: float, **kw) -> dict:
    """供数据集构建使用的标签字典（去掉 None 情况）。

    ret 已按方向取号并扣成本，pol_ret 与 pol_dir_ret 同值（保留两个键名以兼容旧 schema）。
    """
    r = simulate_exit(ind, entry_idx, is_buy, atr, **kw)
    if r is None:
        return {}
    return {"pol_ret": r.ret, "pol_dir_ret": r.ret,
            "pol_bars": float(r.bars), "pol_stop": 1.0 if r.reason in ("stop", "trail") else 0.0}
