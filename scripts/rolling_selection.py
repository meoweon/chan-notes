"""滚动选参流程净值：用"每笔期望"每年重选参数，产出可落地的流程曲线。

与 param_sweep_backtest.py 的区别：
- 不做事后最优对比（那是作弊上限），而是模拟"每年初只用过去数据选参、下一年照用"。
- 顺带保存 480 组的月度净值与月度交易统计，供画图复用。

产物：
- reports/rolling_grid.pkl     480 组的月度净值 + 月度交易统计
- reports/rolling_selection.json 选参记录、分年指标与 plotly HTML 图表

用法：python scripts/rolling_selection.py [--quick]
"""
import json
import pickle
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import param_sweep_backtest as psb

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
SFX = os.environ.get("OUT_SUFFIX", "")

def chart_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


BASE = dict(hold=48, stop=2.0, tp=5.0, f_name="无", s_name="无")
MIN_N = 30                     # 选参时要求的历史成交笔数
MIN_FREQ = 100                 # 选参时要求的历史频率（笔/年）：不设下限会选到几乎不交易的组合
SEL_YEARS = [2022, 2023, 2024, 2025, 2026]
START = pd.Timestamp("2022-01-01")   # 第一条选参决策生效日，三组曲线共同起点


def _init_worker():
    psb._CTX.update(psb.build_context(verbose=False))


def _work(task):
    return psb.run_combo(psb._CTX, *task, return_monthly=True, return_tstat=True)


def build_grid(quick=False):
    holds, stops, tps = (psb.HOLDS, psb.STOPS, psb.TPS) if not quick else ([48], [2.0, 5.0], [5.0])
    grid = [(h, s, t, f, sc) for h, s, t, (f, _), (sc, _)
            in product(holds, stops, tps, psb.FRENZY_QS, psb.SCORE_QS)]
    return grid[:4] if quick else grid


def key_of(row):
    return (int(row["hold"]), float(row["stop"]), float(row["take_profit"]),
            str(row["frenzy"]), str(row["score"]))


def collect(grid):
    cache = REPORTS / f"rolling_grid{SFX}.pkl"
    if cache.exists() and "--force" not in sys.argv:
        with open(cache, "rb") as f:
            old = pickle.load(f)
        if len(old) >= len(grid):
            print(f"复用缓存 {cache}（{len(old)} 组）", flush=True)
            return old
    workers = min(8, (__import__("os").cpu_count() or 2) - 1)
    print(f"回测 {len(grid)} 组（{workers} 进程）...", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for i, row in enumerate(ex.map(_work, grid, chunksize=4), 1):
            if row:
                out[key_of(row)] = {"monthly": row["__monthly"], "tstat": row["__tstat"]}
            if i % 40 == 0 or i == len(grid):
                print(f"  {i}/{len(grid)}", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"保存 {cache}  {len(out)} 组", flush=True)
    return out


def month_returns(monthly):
    """月度净值 -> 月度收益率（相邻月末）。"""
    m = monthly.dropna()
    return m.pct_change().dropna()


def hist_stats(tstat, until):
    """截至 until（不含）：每笔期望、成交笔数、年化频率。"""
    if tstat is None or not len(tstat):
        return np.nan, 0, 0.0
    idx = pd.PeriodIndex(tstat.index, freq="M")
    mask = idx < until.to_period("M")
    sel = tstat.loc[mask]
    n = float(sel["n"].sum())
    months = max(int(mask.sum()), 1)
    if n <= 0:
        return np.nan, 0, 0.0
    return float(sel["sr"].sum()) / n, int(n), n / (months / 12.0)


def pick(grid, data, until, min_freq=MIN_FREQ):
    best, best_avg, best_key = None, -np.inf, None
    for k in grid:
        if k not in data:
            continue
        avg, n, freq = hist_stats(data[k]["tstat"], until)
        if n < MIN_N or freq < min_freq or not np.isfinite(avg):
            continue
        if avg > best_avg:
            best, best_avg, best_key = avg, avg, k
    return best_key, best_avg


def flow_curve(data, choices):
    """按 (生效日, 参数) 拼接月度收益，返回月末净值序列。"""
    rets = {}
    dates = []
    for i, (d, k, _) in enumerate(choices):
        end = choices[i + 1][0] if i + 1 < len(choices) else pd.Timestamp("2027-01-01")
        r = month_returns(data[k]["monthly"])
        sel = r.loc[(r.index >= d) & (r.index < end)]
        for t, v in sel.items():
            rets[t] = v
            dates.append(t)
    s = pd.Series(rets).sort_index()
    if not len(s):
        return pd.Series(dtype=float)
    return (1 + s).cumprod().rename("equity")


def fixed_curve(data, k):
    return flow_curve(data, [(START, k, None)])


def clip_daily(s):
    return s


def drawdown(s):
    return s / s.cummax() - 1


def metrics_from_tstat(tstat, months):
    sel = tstat[tstat.index.isin(months)]
    n = float(sel["n"].sum())
    if n <= 0:
        return dict(n=0, win=np.nan, payoff=np.nan, avg_ret=np.nan, freq=0.0)
    wins = float(sel["wins"].sum())
    sw, sl = float(sel["sw"].sum()), float(sel["sl"].sum())
    payoff = np.nan
    if wins > 0 and (n - wins) > 0 and sl > 0:
        payoff = (sw / wins) / (sl / (n - wins))
    years = max(len(months), 1) / 12.0
    return dict(n=int(n), win=wins / n, payoff=payoff,
                avg_ret=float(sel["sr"].sum()) / n, freq=n / years)


def main(quick=False):
    grid = build_grid(quick)
    data = collect(grid)
    base_key = (int(BASE["hold"]), float(BASE["stop"]), float(BASE["tp"]),
                BASE["f_name"], BASE["s_name"])
    if base_key not in data:
        raise SystemExit("基线组合缺失")

    # 逐年选参：2022-01-01 起，每年初只用当年之前的数据
    choices, picks = [], []
    for y in SEL_YEARS:
        d = pd.Timestamp(f"{y}-01-01")
        k, avg = pick(grid, data, d)
        if k is None:
            print(f"  {y}: 历史不足，跳过")
            continue
        _, _, hf = hist_stats(data[k]["tstat"], d)
        choices.append((d, k, avg))
        picks.append({"year": y, "hold": k[0], "stop": k[1], "tp": k[2],
                      "frenzy": k[3], "score": k[4], "hist_avg_ret": avg,
                      "hist_freq": hf})
    print("选参记录：")
    for p in picks:
        print(f"  {p['year']}: {p['hold']}/{p['stop']}/{p['tp']}/{p['frenzy']}/{p['score']}"
              f"  历史每笔期望 {p['hist_avg_ret']:+.2%}  频率 {p['hist_freq']:.0f}/年")

    # 对照：不加频率下限会选到什么（写进脚注，不画线）
    free_choices = [(d, *pick(grid, data, d, min_freq=0)) for d in
                    [pd.Timestamp(f"{y}-01-01") for y in SEL_YEARS]]
    free_choices = [(d, k, a) for d, k, a in free_choices if k]
    flow_free = flow_curve(data, free_choices)

    flow = flow_curve(data, choices)
    base = fixed_curve(data, base_key)
    # 事后最优：拿 IS 全段每笔期望最高的组合跑全程（作弊上限）
    ep_key, ep_avg = pick(grid, data, psb.IS_END)
    epost = fixed_curve(data, ep_key)
    print(f"事后最优（IS 每笔期望最高）: {ep_key}  IS 每笔 {ep_avg:+.2%}")

    # --- 图 1：净值 + 水下回撤 ---
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.68, 0.32],
                         subplot_titles=["净值（起点 = 1.0）", "水下回撤（%）"])
    series = [("基准 48/2/5（不选参）", base, "#7f7f7f", "solid"),
              ("滚动选参（每笔期望，频率≥100/年）", flow, "#2a9d8f", "solid"),
              ("事后最优（IS 内每笔期望最高，作弊上限）", epost, "#e76f51", "dash")]
    for name, s, color, dash in series:
        if not len(s):
            continue
        fig1.add_trace(go.Scatter(x=s.index, y=s.values, name=name,
                                  line=dict(color=color, width=2, dash=dash)),
                       row=1, col=1)
        dd = drawdown(s)
        fig1.add_trace(go.Scatter(x=dd.index, y=(dd.values * 100).tolist(), name=name,
                                  line=dict(color=color, width=1.4, dash=dash),
                                  showlegend=False),
                       row=2, col=1)
    fig1.update_yaxes(type="log", row=1, col=1)
    fig1.update_xaxes(title_text="选参流程净值 vs 不选参 vs 事后最优（2022-01 起）", row=1, col=1)
    fig1.update_yaxes(title_text="水下回撤（%）", zeroline=True, row=2, col=1)
    # 标注各曲线最大回撤与恢复
    notes = []
    for i, (name, s, color, dash) in enumerate(series):
        if not len(s):
            continue
        dd = drawdown(s)
        trough = dd.idxmin()
        depth = dd.min()
        peak = s.loc[:trough].max()
        after = s.loc[s.index > trough]
        rec = after[after >= peak]
        rec_txt = f"{(rec.index[0] - trough).days} 天" if len(rec) else "未恢复"
        fig1.add_trace(go.Scatter(x=[trough], y=[float(depth * 100)], mode="markers",
                                  marker=dict(color=color, size=6), showlegend=False),
                       row=2, col=1)
        notes.append((name, depth, rec_txt))
        dy = 15 if depth < -0.10 else -18 - 15 * i
        dx = -100 if name.startswith("滚动") else (50 if name.startswith("基准") else 0)
        fig1.add_annotation(x=trough, y=float(depth * 100), text=f"{depth:.1%} / {rec_txt}",
                            xref="x2", yref="y2", xanchor="center",
                            yanchor="bottom" if dy > 0 else "top",
                            ax=dx, ay=-dy, axref="pixel", ayref="pixel",
                            font=dict(size=10, color=color),
                            arrowhead=2, arrowsize=0.8, arrowwidth=1, arrowcolor=color)
    fig1.update_layout(height=650, width=960, template="plotly_white",
                       legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"))
    _dd_lo = min((drawdown(s).min() for _, s, _, _ in series if len(s)), default=-0.1)
    fig1.update_yaxes(range=[_dd_lo * 100 * 1.35, 3], row=2, col=1)

    # --- 图 2：交易型指标分年 ---
    years = [y for y in SEL_YEARS if y >= flow.index[0].year]
    rows = []
    for y in years:
        months = [f"{y}-{m:02d}" for m in range(1, 13)]
        rows.append({"year": y, "who": "基准 48/2/5",
                     **metrics_from_tstat(data[base_key]["tstat"], months)})
        if y in [c[0].year for c in choices]:
            k = next(k for d, k, _ in choices if d.year == y)
            rows.append({"year": y, "who": "滚动选参",
                         **metrics_from_tstat(data[k]["tstat"], months)})
    mdf = pd.DataFrame(rows)

    fig2 = make_subplots(rows=2, cols=2, shared_xaxes=True,
                         subplot_titles=["每笔期望", "胜率", "盈亏比", "频率（笔/年）"])
    specs = [("avg_ret", 100, "%"), ("win", 100, "%"),
             ("payoff", 1, ""), ("freq", 1, "")]
    for idx, (col, scale, unit) in enumerate(specs):
        r, c = idx // 2 + 1, idx % 2 + 1
        for who, color, symbol in [("基准 48/2/5", "#7f7f7f", "circle"),
                                   ("滚动选参", "#2a9d8f", "square")]:
            sub = mdf[mdf.who == who]
            if not len(sub):
                continue
            fig2.add_trace(go.Scatter(x=sub.year, y=(sub[col] * scale).tolist(),
                                      name=who, line=dict(color=color, width=1.8),
                                      marker=dict(color=color, symbol=symbol, size=7)),
                           row=r, col=c)
        ytitle = specs[idx][1]
        fig2.update_yaxes(title_text=f"{ytitle}{unit}", row=r, col=c)
    fig2.update_layout(height=520, width=960, title_text="交易型指标逐年对比（净值线看不出的那四项）",
                       template="plotly_white")
    fig2.update_xaxes(title_text="年份", row=2, col=1)
    fig2.update_xaxes(title_text="年份", row=2, col=2)

    out = {"picks": picks, "free_picks": [[str(d.date()), list(k)] for d, k, _ in free_choices],
           "free_end": float(flow_free.iloc[-1]) if len(flow_free) else None,
           "ex_post": {"key": list(ep_key), "is_avg_ret": ep_avg},
           "fig1": notes,
           "fig1_html": chart_div(fig1),
           "fig2_html": chart_div(fig2),
           "years": mdf.to_dict(orient="records"),
           "flow_end": float(flow.iloc[-1]) if len(flow) else None,
           "base_end": float(base.iloc[-1]) if len(base) else None,
           "epost_end": float(epost.iloc[-1]) if len(epost) else None}
    with open(REPORTS / f"rolling_selection{SFX}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print("\n最终净值：基准 %.3f  滚动 %.3f  事后最优 %.3f"
          % (out["base_end"] or np.nan, out["flow_end"] or np.nan, out["epost_end"] or np.nan))
    for n, d, r in notes:
        print(f"  最大回撤 {n}: {d:.1%} 恢复 {r}")
    print(f"写出 {REPORTS/'rolling_selection.json'}（含 plotly HTML div）")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
