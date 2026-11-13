"""组合级逐 bar 盯市回测（A股多头）。

规则：
- 信号 bar 的下一根开盘入场（event.entry_time 对应的 bar 开盘价）
- 单笔仓位 = 当前净值 × pos_pct
- 最多同时持有 max_pos
- 止损：入场价 - stop_atr × ATR（bar 内 low 触发，按止损价成交）
- 时间退出：持有 H 根后该 bar 收盘离场
- 反向信号：持仓股票出现卖点 BSP 时收盘离场
- 成本：往返 cost（默认 13bps）
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BtParams:
    capital: float = 1_000_000.0
    pos_pct: float = 0.10
    max_pos: int = 10
    stop_atr: float = 2.0
    take_profit_atr: float | None = None  # None 表示不启用止盈
    hold_bars: int = 24
    cost: float = 0.0013
    signal_exit: bool = True


@dataclass
class Position:
    code: str
    entry_time: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    take_profit: float | None = None
    bars_held: int = 0
    entry_idx: int = 0


def run(events: pd.DataFrame, bars: dict[str, pd.DataFrame], p: BtParams = BtParams(),
        date_col: str = "entry_time", entry_mode: str = "signal"):
    """events: 需要 code/entry_time/entry_price/is_buy/atr；bars: code -> DataFrame(time,open,high,low,close)

    entry_mode="confirm" 时只用 conf_ok==1 的事件，按 conf_entry_time/price 入场。
    """
    ev = events[events["is_buy"] == 1].copy()
    if entry_mode == "confirm":
        ev = ev[ev["conf_ok"] == 1].copy()
        ev["entry_time"] = pd.to_datetime(ev["conf_entry_time"])
        ev["entry_price"] = ev["conf_entry_price"]
        ev["atr"] = ev["conf_atr"]
    ev["entry_time"] = pd.to_datetime(ev[date_col] if entry_mode != "confirm" else ev["entry_time"])
    if "score" in ev.columns:
        ev = ev.sort_values(["entry_time", "score"], ascending=[True, False])
    sell_times = {}
    for r in events[events["is_buy"] == 0].itertuples():
        sell_times.setdefault(r.code, set()).add(pd.Timestamp(getattr(r, date_col)))

    # 每股时间->索引
    idx_map = {}
    all_times = set()
    for code, df in bars.items():
        ts = pd.to_datetime(df["time"]).to_numpy()
        idx_map[code] = {t: i for i, t in enumerate(ts)}
        all_times.update(ts)
    timeline = sorted(all_times)

    entries_by_time = {}
    for r in ev.itertuples():
        entries_by_time.setdefault(pd.Timestamp(r.entry_time), []).append(r)

    cash = p.capital
    positions: dict[str, Position] = {}
    equity_curve = []
    trades = []

    for t in timeline:
        # 1) 已有持仓的盘中止损 / 时间退出
        for code in list(positions):
            pos = positions[code]
            df = bars[code]
            i = idx_map[code].get(t)
            if i is None:
                continue
            o, h, l, c = df["open"].iat[i], df["high"].iat[i], df["low"].iat[i], df["close"].iat[i]
            pos.bars_held += 1
            exit_price, reason = None, None
            if l <= pos.stop:
                exit_price, reason = pos.stop, "stop"
            elif pos.take_profit is not None and h >= pos.take_profit:
                exit_price, reason = pos.take_profit, "take_profit"
            elif pos.bars_held >= p.hold_bars:
                exit_price, reason = c, "time"
            elif p.signal_exit and t in sell_times.get(code, set()):
                exit_price, reason = c, "signal"
            if exit_price is not None:
                cash += pos.shares * exit_price * (1 - p.cost / 2)
                trades.append({"code": code, "entry_time": pos.entry_time, "exit_time": t,
                               "entry": pos.entry_price, "exit": exit_price,
                               "ret": exit_price / pos.entry_price - 1 - p.cost,
                               "bars": pos.bars_held, "reason": reason})
                del positions[code]

        # 2) 新信号入场
        for r in entries_by_time.get(t, []):
            if r.code in positions or len(positions) >= p.max_pos:
                continue
            df = bars[r.code]
            i = idx_map[r.code].get(t)
            if i is None:
                continue
            price = df["open"].iat[i]
            atr = r.atr if r.atr == r.atr and r.atr > 0 else price * 0.01
            equity = cash + sum(x.shares * bars[c]["close"].iat[idx_map[c][t]]
                                for c, x in positions.items() if t in idx_map[c])
            budget = min(equity * p.pos_pct, cash)
            shares = budget / (price * (1 + p.cost / 2))
            if shares <= 0:
                continue
            cash -= shares * price * (1 + p.cost / 2)
            take_profit = None
            if p.take_profit_atr is not None:
                take_profit = price + p.take_profit_atr * atr
            positions[r.code] = Position(code=r.code, entry_time=t, entry_price=price,
                                         shares=shares, stop=price - p.stop_atr * atr,
                                         take_profit=take_profit, entry_idx=i)

        # 3) 盯市
        eq = cash
        for code, pos in positions.items():
            i = idx_map[code].get(t)
            if i is not None:
                eq += pos.shares * bars[code]["close"].iat[i]
        equity_curve.append((t, eq))

    eq_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    tr_df = pd.DataFrame(trades)
    return {"equity": eq_df, "trades": tr_df, "params": p,
            "metrics": metrics(eq_df, tr_df, p)}


def metrics(eq: pd.DataFrame, trades: pd.DataFrame, p: BtParams) -> dict:
    if eq.empty:
        return {}
    e = eq["equity"]
    ret = e.iloc[-1] / p.capital - 1
    bars = len(e)
    years = (eq.index[-1] - eq.index[0]).days / 365.25 if bars > 1 else np.nan
    bars_per_year = bars / years if years and years > 0 else 8 * 242
    cagr = (1 + ret) ** (1 / years) - 1 if years and years > 0 and ret > -1 else np.nan
    r = e.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else np.nan
    dd = (e / e.cummax() - 1).min()
    out = {"total_ret": ret, "cagr": cagr, "sharpe": sharpe, "mdd": dd, "bars": bars}
    if not trades.empty:
        out.update({
            "n_trades": len(trades), "win_rate": (trades["ret"] > 0).mean(),
            "avg_ret": trades["ret"].mean(), "avg_win": trades.loc[trades["ret"] > 0, "ret"].mean(),
            "avg_loss": trades.loc[trades["ret"] <= 0, "ret"].mean(),
            "avg_bars": trades["bars"].mean(),
        })
    return out
