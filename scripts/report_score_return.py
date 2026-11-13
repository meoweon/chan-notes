"""模型打分与收益率关系：交互式 HTML 可视化报告。

口径完全沿用 analyze_score_return_oos.py / analyze_bsppoint.py 的 load()：
确认买点（is_buy==1 且 conf_ok==1）、有逐月 walk-forward 样本外分数、2021-01~2026-10。
前向收益主列 c_dir_ret_48（确认入场、持有 48bar 收盘离场、按方向取号、已扣 13bp、无止损、无组合交互）。
本脚本不重训、不改标签，只对既有事件表与样本外分数做可视化剖析。

用法：
    EVENTS_NAME=crypto_31_v1 MODEL_NAME=wf_crypto_31_xgb_oos \
        .venv/bin/python scripts/report_score_return.py

产物：
    reports/score_return_report{OUT_SUFFIX}.html（plotly.js 走 CDN，打开需联网）
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

sys.path.insert(0, os.path.dirname(__file__))
from analyze_score_return_oos import load, IS_END, WINDOW_START, OOS_END, FW, REPORTS

OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
NAME = os.environ.get("EVENTS_NAME", "crypto_31_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "wf_crypto_31_xgb_oos")

C = {
    "is": "#2563eb", "oos": "#ea580c", "true": "#16a34a", "false": "#dc2626",
    "all": "#475569", "surv": "#7c3aed", "mfe": "#0891b2", "mae": "#db2777",
    "pol": "#ca8a04", "grid": "#e2e8f0",
}
TEMPLATE = "plotly_white"


def atr_frac(g):
    px = g["conf_entry_price"] if "conf_entry_price" in g else g["entry_price"]
    return (g["conf_atr"] / px).replace([np.inf, -np.inf], np.nan)


def decile_table(g):
    q = pd.qcut(g["score"], 10, duplicates="drop", labels=False)
    af = atr_frac(g)
    rows = []
    for b in sorted(pd.Series(q).dropna().unique()):
        x = g[q == b]
        r = x[FW]
        rows.append({
            "bin": int(b), "n": int(len(x)),
            "score_mid": float(x.score.mean()),
            "survive": float(x.survive_144.mean()),
            "ret_all": float(r.mean()),
            "ret_true": float(x.loc[x.survive_144 == 1, FW].mean()),
            "ret_false": float(x.loc[x.survive_144 == 0, FW].mean()),
            "win": float((r > 0).mean()),
            "mfe": float(x["c_mfe_48"].mean()),
            "mae": float(x["c_mae_48"].mean()),
            "volnorm": float((r / af.loc[x.index]).mean()),
            "pol": float(x["c_pol_dir_ret"].mean()),
            "stop": float(x["c_stop_48"].mean()),
            "ret8": float(x["c_dir_ret_8"].mean()),
            "ret24": float(x["c_dir_ret_24"].mean()),
            "ret48": float(r.mean()),
            "n_true": int((x.survive_144 == 1).sum()),
        })
    return pd.DataFrame(rows)


def base_layout(fig, title, height=420):
    fig.update_layout(
        template=TEMPLATE, title=dict(text=title, font=dict(size=16)),
        height=height, margin=dict(l=60, r=30, t=60, b=50),
        font=dict(family="-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    return fig


def pct(fig, axis="y"):
    fig.update_yaxes(tickformat=".1%") if axis == "y" else fig.update_xaxes(tickformat=".1%")


FIGS = []  # (anchor, title, narrative, fig)


def add(anchor, title, narrative, fig):
    FIGS.append((anchor, title, narrative, fig))


def build(df):
    full = df.dropna(subset=[FW]).copy()
    isl = full[full.t < IS_END].copy()
    oos = full[full.t >= IS_END].copy()
    d_full, d_is, d_oos = decile_table(full), decile_table(isl), decile_table(oos)
    x = d_full.bin.astype(str)

    # 1 分数分布
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=isl.score, name=f"IS (n={len(isl)})", marker_color=C["is"],
                               opacity=0.6, xbins=dict(size=0.05)))
    fig.add_trace(go.Histogram(x=oos.score, name=f"OOS (n={len(oos)})", marker_color=C["oos"],
                               opacity=0.6, xbins=dict(size=0.05)))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="模型分数（样本外预测概率）", tickformat=".0%")
    fig.update_yaxes(title="事件数")
    add("dist", "1. 分数分布", "分数是逐月 walk-forward 的样本外预测概率（预测「结构会不会存活」），"
        f"不是预测涨跌。IS/OOS 分布形态接近，全期均值约 {full.score.mean():.2f}。", base_layout(fig, "模型分数分布（IS vs OOS）"))

    # 2 分数 vs 存活率
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d_is.survive, name="IS", mode="lines+markers", line=dict(color=C["is"])))
    fig.add_trace(go.Scatter(x=x, y=d_oos.survive, name="OOS", mode="lines+markers", line=dict(color=C["oos"])))
    fig.add_trace(go.Scatter(x=x, y=d_full.survive, name="全期", mode="lines+markers",
                             line=dict(color=C["surv"], dash="dot")))
    fig.update_xaxes(title="分数十分位（D0 最低 → D9 最高）")
    fig.update_yaxes(title="存活率（survive_144）")
    pct(fig)
    add("surv", "2. 分数 → 存活率：单调正梯度",
        "这是模型真正学到的东西：分数越高，买卖点越不容易被引擎擦除。IS/OOS 都单调上升，"
        f"D0 约 {d_full.survive.iloc[0]:.0%} → D9 约 {d_full.survive.iloc[-1]:.0%}。"
        f"事件级 Spearman IS {spearmanr(isl.score, isl.survive_144).statistic:+.3f} / "
        f"OOS {spearmanr(oos.score, oos.survive_144).statistic:+.3f}，区间均排除 0（见 E003）。",
        base_layout(fig, "分数十分位 vs 结构存活率"))

    # 3 抵消机制：真/假/全部收益
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=d_full.ret_true, name="真点收益", mode="lines+markers",
                             line=dict(color=C["true"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=d_full.ret_false, name="假点收益", mode="lines+markers",
                             line=dict(color=C["false"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=d_full.ret_all, name="全部收益（净）", mode="lines+markers",
                             line=dict(color=C["all"], width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=d_full.survive, name="存活率", mode="lines+markers",
                             line=dict(color=C["surv"], dash="dot")), secondary_y=True)
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="前向收益 c_dir_ret_48", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title="存活率", tickformat=".0%", secondary_y=True, showgrid=False)
    fig.update_layout(hovermode="x unified")
    add("cancel", "3. 抵消机制：为什么净收益没有梯度",
        "核心图。分数升高 → 存活率上升（紫）、真点占比变大；但真点收益（绿）反而下降、假点收益（红）始终为负。"
        "两股力量抵消，全部收益（灰粗线）基本走平。恒等式：全部收益 = 存活率×真点收益 + (1−存活率)×假点收益。",
        base_layout(fig, "分数十分位：真点 / 假点 / 全部收益 与存活率", height=460))

    # 4 事件级散点
    samp = full.sample(min(len(full), 2264), random_state=1)
    fig = go.Figure()
    for val, name, col in [(0, "假点（被擦除）", C["false"]), (1, "真点（存活）", C["true"])]:
        s = samp[samp.survive_144 == val]
        fig.add_trace(go.Scattergl(x=s.score, y=s[FW], mode="markers", name=name,
                                   marker=dict(color=col, size=5, opacity=0.4)))
    fig.update_xaxes(title="模型分数", tickformat=".0%")
    fig.update_yaxes(title="c_dir_ret_48（单点净收益）", tickformat=".0%")
    add("scatter", "4. 事件级散点：噪声远大于信号",
        f"每个点是一个买点（n={len(full)}）。绿=存活、红=被擦除。可见同一分数下收益散布极宽（±20% 常见），"
        f"分数轴上几乎看不出斜率——事件级「分数→全部收益」Spearman 仅 "
        f"{spearmanr(isl.score, isl[FW]).statistic:+.3f}（IS）/ {spearmanr(oos.score, oos[FW]).statistic:+.3f}（OOS），区间跨 0。",
        base_layout(fig, "分数 vs 单点前向收益（事件级）", height=480))

    # 5 多持有期期限结构
    fig = go.Figure()
    for col, name, h in [("ret8", "持有 8bar", "#93c5fd"), ("ret24", "持有 24bar", "#3b82f6"),
                         ("ret48", "持有 48bar", "#1e3a8a")]:
        fig.add_trace(go.Scatter(x=x, y=d_full[col], name=name, mode="lines+markers", line=dict(color=h)))
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="平均前向收益", tickformat=".1%")
    add("term", "5. 多持有期：负梯度是否稳健",
        "把持有期换成 8 / 24 / 48bar 各算一遍。若高分档收益在所有持有期都偏低，说明不是「48bar」挑出来的假象。"
        "三条线高分段整体走低，方向一致。", base_layout(fig, "不同持有期下的分档收益"))

    # 6 胜率
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d_is.win, name="IS", mode="lines+markers", line=dict(color=C["is"])))
    fig.add_trace(go.Scatter(x=x, y=d_oos.win, name="OOS", mode="lines+markers", line=dict(color=C["oos"])))
    fig.add_hline(y=0.5, line=dict(color="#94a3b8", dash="dot"))
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="胜率（收益>0 占比）", tickformat=".0%")
    add("win", "6. 胜率：只看方向，不看幅度",
        "胜率把每个点压成「赚/亏」一个比特，免疫极端值。高分档胜率并未系统性更高——"
        "模型预测的是结构存活，不是涨跌方向（作者 experiment_log #3：方向不可预测，AUC≈0.5）。",
        base_layout(fig, "分数十分位 vs 胜率"))

    # 7 MFE / MAE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d_full.mfe, name="MFE 最大有利偏移", mode="lines+markers",
                             line=dict(color=C["mfe"])))
    fig.add_trace(go.Scatter(x=x, y=d_full.mae, name="MAE 最大不利偏移", mode="lines+markers",
                             line=dict(color=C["mae"])))
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="持有期内最大偏移", tickformat=".1%")
    add("excursion", "7. 最大有利 / 不利偏移（MFE/MAE）",
        "MFE=持有期内最多曾赚多少，MAE=最多曾亏多少。把「方向对不对」和「幅度大不大」拆开看。"
        "高分档的 MFE 并未更大，MAE 也未更小——结构稳 ≠ 行情爆发。", base_layout(fig, "分数十分位 vs MFE / MAE"))

    # 8 波动率归一化
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=d_full.volnorm, marker_color=C["all"], name="收益 / ATR"))
    fig.add_hline(y=0, line=dict(color="#94a3b8"))
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="收益（ATR 倍数）")
    add("volnorm", "8. 波动率归一化收益",
        "原始百分比收益会被高波动币主导。除以 ATR（换算成「承担一份波动赚多少」）后跨币更公平。"
        "归一化后高分档仍无正梯度，说明抵消不是波动率差异伪造的。", base_layout(fig, "分数十分位 vs 波动率归一化收益"))

    # 9 实盘对照：固定持有 vs 策略退出
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d_full.ret48, name="固定持有 48bar", mode="lines+markers",
                             line=dict(color=C["all"])))
    fig.add_trace(go.Scatter(x=x, y=d_full.pol, name="策略退出（2ATR止损/24bar）", mode="lines+markers",
                             line=dict(color=C["pol"])))
    fig.add_hline(y=0, line=dict(color="#94a3b8", dash="dot"))
    fig.update_xaxes(title="分数十分位")
    fig.update_yaxes(title="平均收益", tickformat=".1%")
    add("policy", "9. 实盘逻辑对照：固定持有 vs 带止损退出",
        "回应「为什么不用实盘止损逻辑」：黄线是带 2ATR 止损/24bar 时间退出的策略收益（c_pol_dir_ret）。"
        "两条线在高分档都没有抬升——换成带止损的实盘口径，分数依旧没有收益梯度。",
        base_layout(fig, "固定持有 vs 策略退出（带止损）"))

    # 10 IS vs OOS 小 multiples
    fig = make_subplots(rows=1, cols=3, subplot_titles=["存活率", "真点收益", "全部收益"])
    for j, (col, fmt) in enumerate([("survive", ".0%"), ("ret_true", ".1%"), ("ret_all", ".1%")], start=1):
        fig.add_trace(go.Scatter(x=d_is.bin.astype(str), y=d_is[col], name="IS", legendgroup="is",
                                 mode="lines+markers", line=dict(color=C["is"])), row=1, col=j)
        fig.add_trace(go.Scatter(x=d_oos.bin.astype(str), y=d_oos[col], name="OOS", legendgroup="oos",
                                 mode="lines+markers", line=dict(color=C["oos"])), row=1, col=j)
        fig.update_yaxes(tickformat=fmt, row=1, col=j)
        fig.update_xaxes(title="分位", row=1, col=j)
    add("isoos", "10. 跨窗口稳定性（IS vs OOS）",
        f"前段 IS（2021-01~2025-06，n={len(isl)}）与后段 OOS（2025-07~2026-10，n={len(oos)}）各自分档。"
        "存活率正梯度、真点收益负梯度、全部收益走平——三段在两个窗口同号，机制跨时间稳定（E003 H1 成立）。",
        base_layout(fig, "IS / OOS 分档对比", height=400))

    # 11 分类型
    tp = full.groupby("bsp_type").agg(
        n=("score", "size"), ret=(FW, "mean"), survive=("survive_144", "mean"),
        score=("score", "mean")).sort_values("n", ascending=False).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=tp.bsp_type, y=tp.ret, name="平均收益", marker_color=C["all"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=tp.bsp_type, y=tp.survive, name="存活率", mode="markers+lines",
                             line=dict(color=C["surv"])), secondary_y=True)
    fig.update_xaxes(title="买卖点类型")
    fig.update_yaxes(title="平均收益", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title="存活率", tickformat=".0%", secondary_y=True, showgrid=False)
    for r in tp.itertuples():
        fig.add_annotation(x=r.bsp_type, y=r.ret, text=f"n={r.n}", showarrow=False,
                           yshift=14, font=dict(size=10, color="#64748b"))
    top3 = "、".join(f"{r.bsp_type}({r.n})" for r in tp.head(3).itertuples())
    add("type", "11. 分买卖点类型",
        f"缠论买点分 1/1p/2/2s/3a/3b 及组合类型。样本集中在 {top3}。"
        "各类型收益与存活率无一致排序，类型本身不是收益的可靠分层。", base_layout(fig, "各买卖点类型的收益与存活率"))

    # 12 门控增量
    keeps = [0.2, 0.4, 0.5, 0.6, 0.8]
    rows = []
    for tag, g in [("IS", isl), ("OOS", oos)]:
        base = g[FW].mean()
        for k in keeps:
            thr = g.score.quantile(1 - k)
            sel = g[g.score >= thr][FW].mean()
            rows.append({"win": tag, "keep": f"前{int(k*100)}%", "lift": sel - base})
    ld = pd.DataFrame(rows)
    fig = go.Figure()
    for tag, col in [("IS", C["is"]), ("OOS", C["oos"])]:
        s = ld[ld.win == tag]
        fig.add_trace(go.Bar(x=s.keep, y=s.lift, name=tag, marker_color=col))
    fig.add_hline(y=0, line=dict(color="#94a3b8"))
    fig.update_xaxes(title="按分数取前 X% 门控")
    fig.update_yaxes(title="相对不筛的每笔收益增量", tickformat=".2%")
    add("gating", "12. 门控增量：挑高分点能多赚吗",
        "相对「确认后不筛」，按分数取前 X% 的每笔均值收益增量。IS/OOS 各档增量都很小且正负交替；"
        "E003 按月聚类自举显示五档区间全部跨 0——挑高分点没有可证实的增益。", base_layout(fig, "分数门控的收益增量"))

    # 13 时间序列
    m = full.set_index("t").resample("ME").agg(score=("score", "mean"), ret=(FW, "mean"),
                                              n=("score", "size")).dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=m.index, y=m.score, name="月均分数", line=dict(color=C["surv"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=m.index, y=m.ret, name="月均收益", line=dict(color=C["all"])), secondary_y=True)
    fig.add_vline(x=IS_END.timestamp() * 1000, line=dict(color="#94a3b8", dash="dash"),
                  annotation_text="IS | OOS")
    fig.update_xaxes(title="月份")
    fig.update_yaxes(title="月均分数", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title="月均收益", tickformat=".1%", secondary_y=True, showgrid=False)
    add("time", "13. 时间序列：分数与收益按月",
        "月均分数（紫）与月均收益（灰）随时间变化。竖线为 IS/OOS 分界（2025-07）。"
        "收益的起伏主要由市场 regime 驱动，与分数高低无稳定对应——这与作者「利润=狂潮窗口暴露密度」一致。",
        base_layout(fig, "月均分数 vs 月均收益", height=420))

    return full, isl, oos


def corr_table():
    p = REPORTS / f"score_return_oos{OUT_SUFFIX}.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    rows = []
    for tag, lab in [("is", "IS"), ("oos", "OOS")]:
        w = d["windows"][tag]
        se = w["spearman_event"]
        def fmt(k):
            v = se[k]
            lo, hi = v["ci_month"]
            return f"{v['rho']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(f"<tr><td>{lab}（n={w['n']}）</td><td>{fmt('survive')}</td>"
                    f"<td>{fmt('fw_true')}</td><td>{fmt('fw_all')}</td></tr>")
    return ("<table class='tbl'><thead><tr><th>窗口</th><th>分数↔存活</th>"
            "<th>分数↔真点收益</th><th>分数↔全部收益</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            "<p class='note'>事件级 Spearman + 按月聚类 95% 区间（B=4000，源自 E003 产物）。"
            "两个分量各自显著、方向相反，净效应区间跨 0。</p>")


def render(df, full, isl, oos):
    css = """
    :root{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--acc:#2563eb;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);
      font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;line-height:1.65;}
    .wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px;}
    header{border-bottom:2px solid var(--line);padding-bottom:20px;margin-bottom:24px;}
    h1{font-size:26px;margin:0 0 8px;}
    .sub{color:var(--mut);font-size:14px;}
    .banner{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px 18px;margin:18px 0;}
    .banner b{color:var(--acc);}
    .meta{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0;}
    .chip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px;}
    .chip b{font-size:16px;display:block;color:var(--acc);}
    section{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:20px 22px;margin:20px 0;}
    section h2{font-size:19px;margin:0 0 6px;}
    .nar{color:#334155;font-size:14px;margin:0 0 12px;}
    .note{color:var(--mut);font-size:12.5px;}
    .tbl{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0;}
    .tbl th,.tbl td{border:1px solid var(--line);padding:8px 10px;text-align:center;}
    .tbl th{background:#f1f5f9;}
    .toc{columns:2;font-size:13.5px;}
    .toc a{color:var(--acc);text-decoration:none;}
    footer{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:16px;}
    """
    parts = [f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>模型打分与收益率关系研究</title><style>{css}</style></head><body><div class='wrap'>"]
    n_coins = int(full.code.nunique())
    w0, w1 = f"{full.t.min():%Y-%m}", f"{full.t.max():%Y-%m}"
    parts.append("<header><h1>模型打分与收益率关系研究</h1>"
                 f"<div class='sub'>加密 {n_coins + 1} 个 USDT 永续（{n_coins} 币种 + _MARKET）· 1h · "
                 f"逐月 walk-forward 样本外分数 · {w0} ~ {w1}</div></header>")
    parts.append("<div class='banner'>核心结论：模型分数对<b>「结构会不会被擦掉」</b>有真实判别力"
                 "（分数→存活率单调上升），但对<b>「期望收益」没有梯度</b>——存活率的提升被真点收益的下降抵消。"
                 "这是标签（结构存活）与目标（市场收益）错位，已在 E003 样本外复核，并与原作者 A股/BTC 结论跨数据集一致。</div>")
    parts.append("<div class='meta'>"
                 f"<div class='chip'><b>{len(full)}</b>确认买点（有收益）</div>"
                 f"<div class='chip'><b>{len(isl)} / {len(oos)}</b>IS / OOS</div>"
                 f"<div class='chip'><b>{full.survive_144.mean():.1%}</b>整体存活率</div>"
                 f"<div class='chip'><b>{full[FW].mean():.2%}</b>整体平均收益</div>"
                 f"<div class='chip'><b>13bp</b>已扣往返成本</div></div>")
    parts.append("<section><h2>口径</h2><p class='nar'>收益主列 <code>c_dir_ret_48</code>：确认 bar 下一根开盘入场、"
                 "持有 48bar 收盘离场、按方向取号、已扣 13bp、<b>无止损、无组合交互</b>（单点期望，非策略净值）。"
                 "分数为逐月 walk-forward 样本外预测概率，预测目标是有没有被引擎擦除（survive_144），不是涨跌。"
                 "IS/OOS 分界 2025-07-01。本报告不重训、不改标签。</p>")
    parts.append("<div class='toc'>" + "".join(
        f"<div><a href='#{a}'>{t}</a></div>" for a, t, _, _ in FIGS) + "</div></section>")
    parts.append("<section><h2>相关性总览（E003）</h2>" + corr_table() + "</section>")

    first = True
    for a, t, nar, fig in FIGS:
        html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if first else False,
                           config={"displaylogo": False, "responsive": True})
        first = False
        parts.append(f"<section id='{a}'><h2>{t}</h2><p class='nar'>{nar}</p>{html}</section>")

    parts.append(f"<footer>产物：scripts/report_score_return.py · 数据 {NAME}.parquet + "
                 f"{MODEL_NAME}.pkl · 图表 plotly 7.1.0（CDN 加载，需联网）· 生成 {pd.Timestamp.now():%Y-%m-%d}<br>"
                 "边界：单点前向收益不含组合交互；OOS 仅 16 个月；"
                 f"{n_coins} 币种存在幸存者偏差。"
                 "详见 docs/30_实验/E003-分数收益抵消机制样本外复核.md。</footer>")
    parts.append("</div></body></html>")
    dst = REPORTS / f"score_return_report{OUT_SUFFIX}.html"
    dst.write_text("".join(parts), encoding="utf-8")
    print(f"保存 {dst}（{dst.stat().st_size/1e6:.1f} MB，{len(FIGS)} 图）")


def main():
    df = load()
    full, isl, oos = build(df)
    render(df, full, isl, oos)


if __name__ == "__main__":
    main()