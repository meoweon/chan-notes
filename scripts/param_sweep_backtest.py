"""参数扫描回测 v2（干净版）。

相对 v1 的修正：
1. frenzy 指数用 **T-1 收盘**（原版用当日收盘，等于偷看最多 24 小时）。
2. frenzy / score 阈值用 **扩张分位**（只用事件当时之前的信息），不是全样本分位。
3. score 用 **逐月 walk-forward 的样本外预测**（`train_wf_oos.py` 落盘），
   不是用最后一个 fold 的模型给全历史打分。
4. 全网格 3×4×2×5×4 = 480 组，参数选择只在样本内窗口（IS）上做，
   另留 2025-07~2026-10 作为样本外（OOS）校验。

用法：
  python scripts/param_sweep_backtest.py           # 全网格
  python scripts/param_sweep_backtest.py --quick   # 小网格自检
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.config import DATA
from chan_notes.eval.bars import load_frames
from chan_notes.models import train

FREQ = "1h"
NAME = os.environ.get("EVENTS_NAME", "crypto_21_v1")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
MODEL_PATH = DATA / "models" / f'{os.environ.get("MODEL_NAME", "wf_crypto_21_xgb_oos")}.pkl'

ROOT = Path(__file__).resolve().parents[1]
BAR_DIR = ROOT / "data" / "bars" / FREQ
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

WINDOW_START = pd.Timestamp("2021-01-01")      # 样本外打分从 2021-01 起
IS_END = pd.Timestamp("2025-07-01")            # [WINDOW_START, IS_END) 为样本内
OOS_END = pd.Timestamp("2026-11-01")

HOLDS = [24, 48, 96]
STOPS = [1.0, 2.0, 3.0, 5.0]
TPS = [5.0, 8.0]
FRENZY_QS = [("无", None), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)]
SCORE_QS = [("无", None), ("q40", 0.40), ("q60", 0.60), ("q80", 0.80)]

MIN_GATE_EVENTS = 200
FRENZY_MIN_DAYS = 180


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_events(df, d):
    """把样本外分数按行号贴回事件表。"""
    s = pd.Series(d["oos_prob"], index=d["oos_idx"])
    df = df.copy()
    df["score"] = df.index.map(s)
    return df


def build_frenzy_index():
    """T-1 收盘的“狂热指数”：当天 >8% 的币占比，5 日平滑，右移一天。

    事件在 D 日入场时，只能用到 D-1 收盘为止的信息。
    """
    daily = {}
    for f in sorted(BAR_DIR.glob("*.parquet")):
        if f.stem == "_MARKET":
            continue
        b = pd.read_parquet(f, columns=["time", "close"])
        b["d"] = pd.to_datetime(b["time"]).dt.strftime("%Y-%m-%d")
        daily[f.stem] = b.groupby("d")["close"].last()
    px = pd.DataFrame(daily).sort_index()
    rets = px.pct_change()
    frenzy = (rets > 0.08).sum(axis=1) / rets.notna().sum(axis=1)
    frenzy = frenzy.rolling(5, min_periods=1).mean()
    # T-1：当天可用的值来自昨天的收盘
    return frenzy.shift(1)


def expanding_quantile_series(series, q, min_periods=FRENZY_MIN_DAYS):
    """对日频序列做扩张分位；第 t 个值只用 [0, t) 的样本。"""
    vals = series.to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        hist = vals[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_periods:
            out[i] = np.quantile(hist, q)
    return pd.Series(out, index=series.index)


def monthly_score_thresholds(df, q):
    """按自然月给 score 阈值：第 m 月的阈值只用 m 月之前事件的分数。"""
    thr = {}
    months = pd.period_range(df["t"].min().to_period("M"), df["t"].max().to_period("M"), freq="M")
    for m in months:
        hist = df.loc[df["t"] < m.start_time, "score"]
        if hist.notna().sum() >= MIN_GATE_EVENTS:
            thr[str(m)] = float(hist.quantile(q))
    return thr


def slice_metrics(eq, trades, p, start, end):
    e = eq.loc[(eq.index >= start) & (eq.index < end), "equity"]
    if len(e) < 3:
        return None
    ret = e.iloc[-1] / e.iloc[0] - 1
    years = (e.index[-1] - e.index[0]).days / 365.25
    bp_year = len(e) / years if years > 0 else 8760
    r = e.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(bp_year) if r.std() > 0 else np.nan
    # Sortino：只罚下行波动，分母用 sqrt(mean(min(r,0)^2))
    ddev = float(np.sqrt((np.minimum(r, 0) ** 2).mean())) if len(r) else np.nan
    sortino = r.mean() / ddev * np.sqrt(bp_year) if ddev and ddev > 0 else np.nan
    mdd = (e / e.cummax() - 1).min()
    cagr = (1 + ret) ** (1 / years) - 1 if years > 0 and ret > -1 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    # 最大回撤的恢复：从谷底回到前高所需天数；窗口内未恢复则 NaN
    peak = e.cummax()
    dd = e / peak - 1
    if mdd < 0:
        trough = dd.idxmin()
        prior = float(peak.loc[trough])
        before = e.loc[:trough]
        peak_t = before[before >= prior].index[-1]
        after = e.loc[e.index > trough]
        rec = after[after >= prior]
        if len(rec):
            rec_days = float((rec.index[0] - trough).days)
            ddlen_days = float((rec.index[0] - peak_t).days)
        else:
            rec_days = np.nan
            ddlen_days = np.nan
    else:
        rec_days = 0.0
        ddlen_days = 0.0
    tr = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
    n = int(len(tr))
    d = {"ret": ret, "sharpe": sharpe, "mdd": mdd, "n": n,
         "sortino": sortino, "cagr": cagr, "calmar": calmar,
         "rec_days": rec_days, "ddlen_days": ddlen_days,
         "freq": n / years if years > 0 else np.nan}
    if n:
        wr = tr["ret"] > 0
        win = float(wr.mean())
        avg_win = float(tr.loc[wr, "ret"].mean()) if wr.any() else np.nan
        avg_loss = float(tr.loc[~wr, "ret"].mean()) if (~wr).any() else np.nan
        d["win"] = win
        d["avg_ret"] = float(tr["ret"].mean())
        d["payoff"] = (abs(avg_win / avg_loss)
                       if np.isfinite(avg_loss) and avg_loss < 0 else np.nan)
    else:
        d["win"] = d["avg_ret"] = d["payoff"] = np.nan
    return d


_CTX = {}


def build_context(verbose=True):
    if verbose:
        print("加载数据...", flush=True)
    raw = train.load(NAME)
    d = load_model()
    df = raw[(raw.is_buy == 1) & (raw.conf_ok == 1)].copy()
    df = score_events(df, d)
    if verbose:
        print(f"确认买点事件 {len(df)}，有样本外分数 {df.score.notna().sum()}", flush=True)

    df["t"] = pd.to_datetime(df["event_time"])
    df = df[(df.t >= WINDOW_START) & (df.t < OOS_END)].copy()
    # 没有样本外分数的事件直接剔除（2020 年）
    df = df[df.score.notna()].copy()
    if verbose:
        print(f"回测窗口事件 {len(df)}：{df.t.min().date()} ~ {df.t.max().date()}", flush=True)

    frenzy_daily = build_frenzy_index()
    df["d"] = df["t"].dt.strftime("%Y-%m-%d")
    df["frenzy"] = df["d"].map(frenzy_daily)
    df = df.dropna(subset=["frenzy"]).copy()

    # 扩张分位阈值（只用过去）
    f_thr = {name: (expanding_quantile_series(frenzy_daily, q) if q is not None else None)
             for name, q in FRENZY_QS}
    s_thr = {name: (monthly_score_thresholds(df, q) if q is not None else None)
             for name, q in SCORE_QS}
    bars = load_frames(df.code.unique(), FREQ)
    return {"df": df, "f_thr": f_thr, "s_thr": s_thr, "bars": bars, "d": d}


def run_combo(ctx, hold, stop, tp, f_name, s_name, return_equity=False,
              return_monthly=False, return_tstat=False):
    df, f_thr, s_thr, bars = ctx["df"], ctx["f_thr"], ctx["s_thr"], ctx["bars"]
    fmap = f_thr[f_name]
    f_series = df["d"].map(fmap) if fmap is not None else pd.Series(-np.inf, index=df.index)
    if s_name == "无":
        s_series = pd.Series(-np.inf, index=df.index)
    else:
        month = df["t"].dt.to_period("M").astype(str)
        s_series = month.map(s_thr[s_name]).fillna(-np.inf)
    mask = (df["frenzy"] >= f_series) & (df["score"] >= s_series)
    sub = df[mask]
    if len(sub) < 10:
        return None
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=stop,
                           take_profit_atr=tp, hold_bars=hold, signal_exit=True)
    out = portfolio.run(sub, bars, p, entry_mode="confirm")
    if out is None:
        return None
    eq, tr, m = out["equity"], out["trades"], out["metrics"]
    row = {
        "hold": hold, "stop": stop, "take_profit": tp,
        "frenzy": f_name, "score": s_name,
        "ret_all": m.get("total_ret"), "sharpe_all": m.get("sharpe"),
        "mdd_all": m.get("mdd"), "n_all": m.get("n_trades", 0),
        "win_all": m.get("win_rate"),
    }
    for tag, a, b in (("is", WINDOW_START, IS_END), ("oos", IS_END, OOS_END)):
        sm = slice_metrics(eq, tr, p, a, b)
        if sm:
            row[f"ret_{tag}"] = sm["ret"]; row[f"sharpe_{tag}"] = sm["sharpe"]
            row[f"mdd_{tag}"] = sm["mdd"]; row[f"n_{tag}"] = sm["n"]
            row[f"win_{tag}"] = sm["win"]; row[f"avgret_{tag}"] = sm["avg_ret"]
            row[f"sortino_{tag}"] = sm["sortino"]; row[f"cagr_{tag}"] = sm["cagr"]
            row[f"calmar_{tag}"] = sm["calmar"]
            row[f"payoff_{tag}"] = sm["payoff"]; row[f"freq_{tag}"] = sm["freq"]
            row[f"rec_{tag}"] = sm["rec_days"]; row[f"ddlen_{tag}"] = sm["ddlen_days"]
    if return_equity:
        row["__equity"] = eq["equity"]
    if return_monthly:
        row["__monthly"] = eq["equity"].resample("ME").last()
    if return_tstat:
        if len(tr):
            t = tr.copy()
            t["m"] = pd.to_datetime(t["entry_time"]).dt.to_period("M").astype(str)
            t["w"] = (t["ret"] > 0).astype(int)
            t["gw"] = t["ret"].clip(lower=0)
            t["gl"] = (-t["ret"]).clip(lower=0)
            row["__tstat"] = t.groupby("m").agg(
                n=("ret", "size"), wins=("w", "sum"),
                sw=("gw", "sum"), sl=("gl", "sum"), sr=("ret", "sum"))
        else:
            row["__tstat"] = pd.DataFrame(columns=["n", "wins", "sw", "sl", "sr"])
    return row


def _init_worker():
    _CTX.update(build_context(verbose=False))


def _worker_combo(task):
    return run_combo(_CTX, *task)


def main(quick=False):
    ctx = build_context()
    df, bars, d = ctx["df"], ctx["bars"], ctx["d"]
    f_thr, s_thr = ctx["f_thr"], ctx["s_thr"]

    holds, stops, tps = (HOLDS, STOPS, TPS) if not quick else ([48], [2.0, 5.0], [5.0])
    grid = list(product(holds, stops, tps, FRENZY_QS, SCORE_QS))
    if quick:
        grid = grid[:4]
    grid = [(h, s, t, f[0], sc[0]) for h, s, t, f, sc in grid]
    print(f"网格 {len(grid)} 组", flush=True)

    rows = []
    n_workers = min(8, (os.cpu_count() or 2) - 1)
    if quick or n_workers <= 1 or len(grid) < 8:
        for i, g in enumerate(grid, 1):
            r = run_combo(ctx, *g)
            if r:
                rows.append(r)
            if i % 20 == 0 or i == len(grid):
                print(f"  {i}/{len(grid)}", flush=True)
    else:
        print(f"并行 {n_workers} 进程", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
            for i, r in enumerate(ex.map(_worker_combo, grid, chunksize=4), 1):
                if r:
                    rows.append(r)
                if i % 40 == 0 or i == len(grid):
                    print(f"  {i}/{len(grid)}", flush=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(REPORTS / f"param_sweep_results{OUT_SUFFIX}.csv", index=False)
    print(f"保存 {REPORTS/f'param_sweep_results{OUT_SUFFIX}.csv'}  ({len(res_df)} 行)", flush=True)

    # --- 关键曲线 ---
    curves = {}

    def add_curve(tag, kw):
        if kw.get("hold") is None:
            return
        p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=kw["stop"],
                               take_profit_atr=kw.get("tp"), hold_bars=kw["hold"], signal_exit=True)
        sub = df
        if kw.get("f_name") not in (None, "无"):
            fmap = f_thr[kw["f_name"]]
            sub = sub[sub["d"].map(fmap) <= sub["frenzy"]]
        if kw.get("s_name") not in (None, "无"):
            month = sub["t"].dt.to_period("M").astype(str)
            sub = sub[sub["score"] >= month.map(s_thr[kw["s_name"]]).fillna(-np.inf)]
        out = portfolio.run(sub, bars, p, entry_mode="confirm")
        if out is None:
            print(f"  [curve] {tag}: 无有效结果")
            return
        curves[tag] = out["equity"]["equity"]

    is_df = res_df[res_df.n_is >= 30] if "n_is" in res_df else res_df
    best_is = is_df.loc[is_df.sharpe_is.idxmax()] if len(is_df) else None
    best_ret = is_df.loc[is_df.ret_is.idxmax()] if len(is_df) else None

    add_curve("基准 (hold=48/stop=2/tp=5)", dict(hold=48, stop=2.0, tp=5.0))
    if best_is is not None:
        add_curve("★ IS 最优夏普", dict(hold=int(best_is.hold), stop=float(best_is.stop),
                                     tp=float(best_is.take_profit),
                                     f_name=best_is.frenzy, s_name=best_is.score))
        add_curve("同组但去掉 score 门控", dict(hold=int(best_is.hold), stop=float(best_is.stop),
                                          tp=float(best_is.take_profit),
                                          f_name=best_is.frenzy, s_name="无"))
    if best_ret is not None:
        add_curve("IS 最优收益", dict(hold=int(best_ret.hold), stop=float(best_ret.stop),
                                   tp=float(best_ret.take_profit),
                                   f_name=best_ret.frenzy, s_name=best_ret.score))

    if curves:
        eq_df = pd.DataFrame(curves)
        eq_df = eq_df.resample("D").last().ffill().dropna(how="all")
        eq_df.to_csv(REPORTS / f"param_sweep_curves{OUT_SUFFIX}.csv", index_label="time")
        print(f"保存 {REPORTS/f'param_sweep_curves{OUT_SUFFIX}.csv'}  {eq_df.shape}", flush=True)

    # 给报告用的摘要
    summary = {
        "n_events": int(len(df)),
        "n_combos": int(len(res_df)),
        "window": [str(df.t.min().date()), str(df.t.max().date())],
        "is_window": [str(WINDOW_START.date()), str(IS_END.date())],
        "oos_window": [str(IS_END.date()), str(OOS_END.date())],
        "score_oos_events": int(d["oos_idx"].size),
        "score_auc_mean": float(np.nanmean(d["meta"]["auc"])),
    }
    (REPORTS / f"param_sweep_meta{OUT_SUFFIX}.json").write_text(
        pd.Series(summary).to_json(force_ascii=False, indent=1), encoding="utf-8")
    print("完成", flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
