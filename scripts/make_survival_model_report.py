#!/usr/bin/env python
"""Generate survival model performance report HTML with plotly interactive charts.

Data: data/events/{EVENTS_NAME}.parquet (with survive_144 labels)
      data/models/{MODEL_NAME}.pkl (monthly walk-forward out-of-sample predictions)
Caliber: survive_144 = 1 iff clear_bar == -1 or clear_bar > bar_idx + 144;
         censored samples (remaining bars < 144) are NaN, excluded from training and evaluation.
Only out-of-sample predictions are used for metrics.

Charts are generated with plotly and embedded as HTML divs in the report.

Environment variables: EVENTS_NAME / MODEL_NAME / OUT_SUFFIX.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
EVENTS_NAME = os.environ.get("EVENTS_NAME", "crypto_31_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "wf_crypto_31_xgb_oos")
SFX = os.environ.get("OUT_SUFFIX", "")
ASHARE_EVENTS = os.environ.get("ASHARE_EVENTS", "ashare_5_30m")
ASHARE_MODEL = os.environ.get("ASHARE_MODEL", "wf_ashare_xgb_oos")
ASHARE_FREQ = os.environ.get("ASHARE_FREQ", "30m")
# 可选：旧口径模型，用于第 10 节“新旧模型对比”；默认仍是 21 币，不存在则跳过该节。
OLD_EVENTS_NAME = os.environ.get("OLD_EVENTS_NAME", "crypto_21_v1")
OLD_MODEL_NAME = os.environ.get("OLD_MODEL_NAME", "wf_crypto_21_xgb_oos")

C_BLUE, C_GREY, C_RED, C_GREEN, C_ORANGE = "#1a73e8", "#9aa0a6", "#b3261e", "#0a7d32", "#e8710a"

CSS = """
 body{font-family:-apple-system,"PingFang SC",sans-serif;max-width:1080px;margin:32px auto;
      padding:0 20px;color:#222;line-height:1.6}
 h1{font-size:24px;margin-bottom:4px} h2{font-size:17px;margin-top:34px;
      border-left:4px solid #1a73e8;padding-left:8px}
 h3{font-size:14px;margin-top:22px;color:#333}
 .sub{color:#666;font-size:13px;margin-bottom:16px}
 .warn{background:#fdecea;border:1px solid #f5c6c2;color:#a50e0e;padding:12px 14px;
       border-radius:6px;font-size:13px;margin:16px 0}
 .note{background:#f1f3f4;border-radius:6px;padding:10px 14px;font-size:13px;color:#444}
 .card{border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:14px 0}
 .grid{display:flex;gap:28px;flex-wrap:wrap;margin-bottom:6px}
 .k{color:#888;font-size:12px;display:block}
 .grid b{font-size:17px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border-bottom:1px solid #eee;padding:5px 8px;text-align:left}
 th{background:#fafafa;font-weight:600;color:#444}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 .chart-container{border:1px solid #eee;border-radius:6px;margin:6px 0;overflow:hidden}
 li{margin:6px 0} code{background:#f6f8fa;padding:1px 4px;border-radius:3px;font-size:12px}
 details{margin:10px 0} summary{cursor:pointer;color:#1a73e8;font-size:13px}
 .two{display:flex;gap:18px;flex-wrap:wrap} .two>div{flex:1 1 360px;min-width:0}
"""

PLOTLY_LAYOUT = dict(
    font=dict(family="PingFang SC, -apple-system, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    plot_bgcolor="white",
    hovermode="closest",
)


def chart_div(fig, height=None):
    """宽度自适应；极简模板；显式高度。"""
    h = height or fig.layout.height or 440
    fig.update_layout(template="none", height=h, width=None, autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", default_height=f"{h}px",
                       config={"responsive": True, "displaylogo": False})


def load():
    ev = pd.read_parquet(ROOT / "data/events" / f"{EVENTS_NAME}.parquet")
    d = pickle.load(open(ROOT / "data/models" / f"{MODEL_NAME}.pkl", "rb"))
    idx = np.asarray(d["oos_idx"])
    df = ev.iloc[idx].copy()
    df["pred"] = np.asarray(d["oos_prob"])
    df["month"] = np.asarray(d["oos_month"])
    df = df.dropna(subset=["survive_144"]).reset_index(drop=True)
    return ev, df, d


def external_block(events_name, model_name):
    ep = ROOT / "data/events" / f"{events_name}.parquet"
    mp = ROOT / "data/models" / f"{model_name}.pkl"
    if not ep.exists() or not mp.exists():
        return None
    ev = pd.read_parquet(ep)
    d = pickle.load(open(mp, "rb"))
    df = ev.iloc[np.asarray(d["oos_idx"])].copy()
    df["pred"] = np.asarray(d["oos_prob"])
    df["month"] = np.asarray(d["oos_month"])
    df = df.dropna(subset=["survive_144"]).reset_index(drop=True)
    y, p_ = df["survive_144"].values.astype(float), df["pred"].values
    m = metrics(y, p_)
    auc_ci, _, _ = boot_ci(y, p_)
    e, _ = ece(y, p_)
    mo = df.groupby("month").apply(
        lambda g: roc_auc_score(g["survive_144"], g["pred"])
        if g["survive_144"].nunique() > 1 and len(g) >= 8 else np.nan,
        include_groups=False).dropna()
    q = pd.qcut(df["pred"].rank(method="first"), 10, labels=False)
    dec = pd.DataFrame({"q": q, "y": y}).groupby("q")["y"].mean()
    by_t = df.groupby("bsp_type").apply(
        lambda g: pd.Series({"n": len(g), "auc": roc_auc_score(g["survive_144"], g["pred"])
                             if g["survive_144"].nunique() > 1 and len(g) >= 20 else np.nan}),
        include_groups=False).dropna()
    w_auc = float((by_t["auc"] * by_t["n"]).sum() / by_t["n"].sum()) if len(by_t) else np.nan
    ref = df.groupby("bsp_type")["survive_144"].transform("mean").values
    type_only = float(roc_auc_score(y, ref))
    by_c = df.groupby("code").apply(
        lambda g: roc_auc_score(g["survive_144"], g["pred"])
        if g["survive_144"].nunique() > 1 and len(g) >= 20 else np.nan,
        include_groups=False).dropna()
    return dict(name=events_name, n=m["n"], n_sym=df["code"].nunique(), base=m["base"],
                auc=m["auc"], ci=auc_ci, brier=m["brier"],
                bss=1 - m["brier"] / (m["base"] * (1 - m["base"])),
                ece=e, months=len(mo), m_mean=float(mo.mean()), m_std=float(mo.std()),
                dec=dec, w_auc=w_auc, type_only=type_only,
                coin_min=float(by_c.min()), coin_max=float(by_c.max()), n_coin=len(by_c))


def load_old_model():
    """Load the optional older-universe model for the comparison section."""
    old_events = ROOT / "data/events" / f"{OLD_EVENTS_NAME}.parquet"
    old_model = ROOT / "data/models" / f"{OLD_MODEL_NAME}.pkl"
    if not old_events.exists() or not old_model.exists():
        return None
    ev = pd.read_parquet(old_events)
    d = pickle.load(open(old_model, "rb"))
    idx = np.asarray(d["oos_idx"])
    df = ev.iloc[idx].copy()
    df["pred"] = np.asarray(d["oos_prob"])
    df["month"] = np.asarray(d["oos_month"])
    df = df.dropna(subset=["survive_144"]).reset_index(drop=True)
    y, p = df["survive_144"].values.astype(float), df["pred"].values
    m = metrics(y, p)
    mo = df.groupby("month").apply(
        lambda g: roc_auc_score(g["survive_144"], g["pred"])
        if g["survive_144"].nunique() > 1 and len(g) >= 8 else np.nan,
        include_groups=False).dropna()
    auc_ci, _, _ = boot_ci(y, p)
    return dict(
        n=m["n"], base=m["base"], auc=m["auc"], ci=auc_ci,
        brier=m["brier"], bss=1 - m["brier"] / (m["base"] * (1 - m["base"])),
        months=len(mo), m_mean=float(mo.mean()), m_std=float(mo.std()),
        m_min=float(mo.min()), m_max=float(mo.max()),
        n_coin=df["code"].nunique(),
        buy_n=int((df["is_buy"] == 1).sum()), sell_n=int((df["is_buy"] == 0).sum()),
        buy_surv=float(df[df["is_buy"] == 1]["survive_144"].mean()),
        sell_surv=float(df[df["is_buy"] == 0]["survive_144"].mean()),
    )


def fig_compare_decile(a, b, la, lb):
    x = list(range(1, 11))
    fig = go.Figure()
    # barmode="group" 自己会错开，不要再手动 ±0.2（否则双重偏移）
    fig.add_trace(go.Bar(x=x, y=a.values.tolist(), name=la, marker_color=C_BLUE))
    fig.add_trace(go.Bar(x=x, y=b.values.tolist(), name=lb, marker_color=C_ORANGE))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="十分位存活率：加密 vs A 股",
        xaxis=dict(title="模型分数十分位（1=最低，10=最高）",
                   tickvals=x, ticktext=[str(i) for i in x]),
        yaxis=dict(title="实际存活率", range=[0, 1]),
        barmode="group",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
    )
    return chart_div(fig)


def metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return dict(
        n=len(y), base=y.mean(), auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
        brier=brier_score_loss(y, p), logloss=log_loss(y, p),
        mcc=matthews_corrcoef(y, (p >= 0.5).astype(int)),
    )


def ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"bin": q, "y": y, "p": p}).groupby("bin").agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["p"] - g["y"]).abs()).sum()), g


def boot_ci(y, p, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    auc, brier = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        auc.append(roc_auc_score(y[i], p[i]))
        brier.append(brier_score_loss(y[i], p[i]))
    return (np.percentile(auc, [2.5, 97.5]), np.percentile(brier, [2.5, 97.5]),
            float(np.std(auc)))


def fig_monthly(g):
    g = g.copy()
    g["month"] = pd.PeriodIndex(g["month"], freq="M").to_timestamp()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g["month"], y=g["n"], name="样本数（左）",
                         marker_color="#e8eaed", opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=g["month"], y=g["auc"], name="月度 AUC",
                             line=dict(color=C_BLUE, width=2), mode="lines+markers",
                             marker=dict(size=4)), secondary_y=True)
    auc_mean = g["auc"].mean()
    fig.add_hline(y=auc_mean, line=dict(color=C_BLUE, dash="dash", width=1),
                  annotation_text=f"均值 {auc_mean:.3f}", annotation_position="top left",
                  secondary_y=True)
    fig.add_hline(y=0.5, line=dict(color=C_GREY, dash="dot", width=1),
                  annotation_text="随机 0.5", annotation_position="bottom left",
                  secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="月度样本外 AUC（2021-01 ~ 2026-10）",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    fig.update_yaxes(title_text="样本数", secondary_y=False)
    fig.update_yaxes(title_text="AUC", range=[0.4, min(0.95, g["auc"].max() + 0.05)],
                     secondary_y=True)
    fig.update_xaxes(range=[g["month"].min() - pd.Timedelta(days=18),
                            g["month"].max() + pd.Timedelta(days=18)])
    return chart_div(fig)


def fig_drift(g):
    g = g.copy()
    g["month"] = pd.PeriodIndex(g["month"], freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["month"], y=g["base"], name="实际存活率",
                             line=dict(color=C_GREEN, width=2), mode="lines"))
    fig.add_trace(go.Scatter(x=g["month"], y=g["pred"], name="平均预测",
                             line=dict(color=C_ORANGE, width=2, dash="dash"), mode="lines"))
    fig.add_hline(y=g["base"].mean(), line=dict(color=C_GREY, dash="dot", width=1))
    fig.update_layout(**PLOTLY_LAYOUT, title="基准率漂移：实际 vs 预测",
                      yaxis=dict(title="存活率", range=[0, 1]),
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    fig.update_xaxes(range=[g["month"].min() - pd.Timedelta(days=18),
                            g["month"].max() + pd.Timedelta(days=18)])
    return chart_div(fig)


def fig_calibration(y, p, e):
    _, g = ece(y, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="完美校准",
                             line=dict(color=C_GREY, dash="dash", width=1), mode="lines"))
    fig.add_trace(go.Scatter(x=g["p"], y=g["y"], name="十分位分箱",
                             line=dict(color=C_BLUE, width=2), mode="lines+markers",
                             marker=dict(size=6)))
    fig.add_hline(y=y.mean(), line=dict(color=C_RED, dash="dot", width=1),
                  annotation_text=f"基准率 {y.mean():.1%}", annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"校准曲线（ECE {e:.3f}）",
                      xaxis=dict(title="预测存活概率", range=[0, 1]),
                      yaxis=dict(title="实际存活比例", range=[0, 1]),
                      legend=dict(x=0.02, y=0.98, font=dict(size=11)))
    return chart_div(fig)


def fig_dist(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df.loc[df.survive_144 == 0, "pred"], nbinsx=40, histnorm="probability density",
        name="未存活", marker_color=C_RED, opacity=0.55))
    fig.add_trace(go.Histogram(
        x=df.loc[df.survive_144 == 1, "pred"], nbinsx=40, histnorm="probability density",
        name="存活", marker_color=C_GREEN, opacity=0.55))
    fig.add_vline(x=df["pred"].median(), line=dict(color=C_GREY, dash="dot", width=1))
    fig.update_layout(**PLOTLY_LAYOUT, title="按真实标签的预测分布",
                      xaxis=dict(title="预测存活概率"),
                      yaxis=dict(title="密度"), barmode="overlay",
                      legend=dict(font=dict(size=11)))
    return chart_div(fig)


def fig_decile(df):
    q = pd.qcut(df["pred"].rank(method="first"), 10, labels=False)
    g = pd.DataFrame({"q": q, "y": df["survive_144"], "p": df["pred"]}).groupby("q").agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=(g.index + 1).tolist(), y=g["y"].tolist(),
                         name="实际存活率", marker_color=C_GREEN, opacity=0.8))
    fig.add_trace(go.Scatter(x=(g.index + 1).tolist(), y=g["p"].tolist(),
                             name="平均预测", line=dict(color=C_ORANGE, width=2, dash="dash"),
                             mode="lines+markers", marker=dict(size=6)))
    base_rate = df["survive_144"].mean()
    fig.add_hline(y=base_rate, line=dict(color=C_RED, dash="dot", width=1),
                  annotation_text=f"基准率 {base_rate:.1%}", annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, title="十分位单调性：分数越高，存活率越高",
                      xaxis=dict(title="分数十分位（1=最低，10=最高）",
                                 tickvals=list(range(1, 11))),
                      yaxis=dict(title="存活率", range=[0, 1]),
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return chart_div(fig), g


def fig_by_coin(df, overall):
    rows = []
    for k, g in df.groupby("code"):
        if len(g) < 30 or g["survive_144"].nunique() < 2:
            continue
        rows.append(dict(key=k, n=len(g), base=g["survive_144"].mean(),
                         auc=roc_auc_score(g["survive_144"], g["pred"])))
    g = pd.DataFrame(rows).sort_values("auc")
    colors = [C_GREEN if a >= 0.5 else C_GREY for a in g["auc"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=g["key"], x=g["auc"], orientation="h", marker_color=colors,
                         text=[f"{r.auc:.3f} (n={r.n})" for r in g.itertuples()],
                         textposition="outside", textfont=dict(size=10, color="#444")))
    fig.add_vline(x=0.5, line=dict(color=C_RED, dash="dot", width=1))
    fig.add_vline(x=overall, line=dict(color=C_BLUE, dash="dash", width=1),
                  annotation_text=f"总体 {overall:.3f}", annotation_position="top right")
    layout = {**PLOTLY_LAYOUT, "title": "分币种样本外 AUC（已排序）",
              "xaxis": dict(title="AUC", range=[min(0.35, g["auc"].min() - 0.05),
                                                 max(0.85, g["auc"].max() + 0.08)]),
              "yaxis": dict(autorange="reversed"), "showlegend": False,
              "margin": dict(l=60, r=120, t=50, b=50),
              "height": max(380, 24 * len(g) + 120)}
    fig.update_layout(layout)
    return chart_div(fig), g


def fig_by_type(df, overall):
    g = df.groupby("bsp_type").agg(n=("survive_144", "size"), surv=("survive_144", "mean"))
    g["auc"] = [roc_auc_score(x["survive_144"], x["pred"]) if x["survive_144"].nunique() > 1 else np.nan
                for _, x in df.groupby("bsp_type")]
    g = g[g["n"] >= 30]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g.index.tolist(), y=g["surv"].tolist(), name="存活率（左）",
                         marker_color=C_GREEN, opacity=0.75), secondary_y=False)
    base_rate = df["survive_144"].mean()
    fig.add_hline(y=base_rate, line=dict(color=C_RED, dash="dot", width=1),
                  annotation_text=f"全部 {base_rate:.1%}", annotation_position="top right",
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=g.index.tolist(), y=g["auc"].tolist(), name="类型内 AUC（右）",
                             line=dict(color=C_BLUE, width=2, dash="dash"), mode="lines+markers",
                             marker=dict(size=6)), secondary_y=True)
    fig.add_hline(y=overall, line=dict(color=C_BLUE, dash="dot", width=1), secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="按信号类型：存活率与类型内 AUC",
                      yaxis=dict(title="实际存活率", range=[0, 1.02]),
                      legend=dict(x=0.02, y=0.98, font=dict(size=11)))
    fig.update_yaxes(title_text="类型内 AUC", range=[0.3, 0.95], showgrid=False, secondary_y=True)
    fig.update_xaxes(type="category")      # "1"/"2" 这类标签别被当成数值轴
    return chart_div(fig), g


def fig_lift(df):
    rows = []
    for cov in np.arange(0.05, 0.96, 0.05):
        thr = df["pred"].quantile(1 - cov)
        s = df[df["pred"] >= thr]
        rows.append(dict(cov=cov, prec=s["survive_144"].mean(), n=len(s),
                         rec=s["survive_144"].sum() / df["survive_144"].sum()))
    t = pd.DataFrame(rows)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t["cov"], y=t["prec"], name="保留部分存活率",
                             line=dict(color=C_BLUE, width=2), mode="lines+markers",
                             marker=dict(size=5)), secondary_y=False)
    base_rate = df["survive_144"].mean()
    fig.add_hline(y=base_rate, line=dict(color=C_RED, dash="dot", width=1),
                  annotation_text=f"整体基准 {base_rate:.1%}", annotation_position="top right",
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=t["cov"], y=t["rec"], name="召回率（右）",
                             line=dict(color=C_GREY, width=1.5, dash="dash"), mode="lines"),
                  secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="阈值：只保留高分前 X% 时的存活率",
                      xaxis=dict(title="保留比例（由高分到低分）"),
                      legend=dict(x=0.75, y=0.95, font=dict(size=11)))
    fig.update_yaxes(title_text="存活率 / 精确率", range=[0.3, 1.0], secondary_y=False)
    fig.update_yaxes(title_text="召回率", secondary_y=True)
    return chart_div(fig), t


def fig_pr(df):
    y = df["survive_144"].values.astype(int)
    p = df["pred"].values
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, name=f"PR 曲线（AP {ap:.3f}）",
                             line=dict(color=C_BLUE, width=2), mode="lines"))
    fig.add_hline(y=y.mean(), line=dict(color=C_RED, dash="dot", width=1.5),
                  annotation_text=f"随机基线（{y.mean():.1%}）", annotation_position="top right")
    pts = [("0.5", 0.5), ("q60", float(df["pred"].quantile(0.6))),
           ("q80", float(df["pred"].quantile(0.8)))]
    for lab, thr in pts:
        sel = p >= thr
        tp = int((sel & (y == 1)).sum())
        fp = int((sel & (y == 0)).sum())
        fn = int((~sel & (y == 1)).sum())
        pr_, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        fig.add_trace(go.Scatter(x=[rc], y=[pr_], name=lab,
                                 mode="markers+text", marker=dict(size=8, color="#111"),
                                 text=[lab], textposition="top right", textfont=dict(size=10)))
    fig.update_layout(**PLOTLY_LAYOUT, title="精确率-召回率曲线（样本外）",
                      xaxis=dict(title="召回率", range=[0, 1], gridcolor="#eee"),
                      yaxis=dict(title="精确率", range=[0, 1], gridcolor="#eee"),
                      legend=dict(x=0.02, y=0.2, font=dict(size=11)))
    return chart_div(fig), ap


def fig_threshold_curves(df):
    ths = np.linspace(0.02, 0.98, 97)
    per = pd.PeriodIndex(df["month"], freq="M")
    periods = [
        ("验证期 2021-01 ~ 2025-06", df[per <= pd.Period("2025-06", "M")]),
        ("测试期 2025-07 ~ 2026-10", df[per >= pd.Period("2025-07", "M")]),
    ]
    subtitles = []
    for lab, g in periods:
        yy = g["survive_144"].values.astype(int)
        pp = g["pred"].values
        auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
        subtitles.append(f"{lab} (n={len(g):,}, AUC={auc:.4f})")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=subtitles)
    for i, (lab, g) in enumerate(periods, 1):
        y = g["survive_144"].values.astype(int)
        p = g["pred"].values
        pr, rc, ac = [], [], []
        for t in ths:
            sel = p >= t
            tp = int((sel & (y == 1)).sum())
            fp = int((sel & (y == 0)).sum())
            fn = int((~sel & (y == 1)).sum())
            tn = int((~sel & (y == 0)).sum())
            pr.append(tp / max(tp + fp, 1))
            rc.append(tp / max(tp + fn, 1))
            ac.append((tp + tn) / max(len(y), 1))
        fig.add_trace(go.Scatter(x=ths, y=pr, name="Precision", showlegend=(i == 1),
                                 line=dict(color="#2e7d32", width=2), mode="lines"), row=i, col=1)
        fig.add_trace(go.Scatter(x=ths, y=rc, name="Recall", showlegend=(i == 1),
                                 line=dict(color="#c62828", width=2), mode="lines"), row=i, col=1)
        fig.add_trace(go.Scatter(x=ths, y=ac, name="Accuracy", showlegend=(i == 1),
                                 line=dict(color="#ef9a3a", width=2, dash="dashdot"), mode="lines"),
                      row=i, col=1)
        fig.add_vline(x=0.5, line=dict(color="#555", dash="dash", width=1), row=i, col=1)
        fig.update_yaxes(title_text="分值", range=[0, 1.02], row=i, col=1)
        fig.update_xaxes(gridcolor="#eee", row=i, col=1)
        fig.update_yaxes(gridcolor="#eee", row=i, col=1)
    fig.update_xaxes(title_text="阈值", row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"阈值扫描：Precision / Recall / Accuracy（虚线 = 0.5）<br>"
                            f"<sup>数据 {EVENTS_NAME} / 模型 {MODEL_NAME}</sup>",
                      height=600,
                      legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=11)))
    return chart_div(fig)


def fig_importance(gain, perm):
    fig = go.Figure()
    if perm is None:
        g = gain.head(15)[::-1]
        fig.add_trace(go.Bar(y=g.index.tolist(), x=g.values.tolist(), orientation="h",
                             marker_color=C_BLUE))
        fig.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="Gain 重要性"),
                          title="特征重要性 Top 15（置换重要性不可用：最后一个月样本太少）",
                          yaxis=dict(tickfont=dict(size=10)), showlegend=False)
        return chart_div(fig)
    perm_sorted = perm.sort_values("mean", ascending=False)
    gain_scaled = perm_sorted["gain"].values / perm_sorted["gain"].max() * perm_sorted["mean"].max()
    fig.add_trace(go.Bar(y=perm_sorted.index.tolist(), x=perm_sorted["mean"].values,
                         orientation="h", name="置换重要性（AUC 下降）",
                         marker_color=C_BLUE))
    fig.add_trace(go.Bar(y=perm_sorted.index.tolist(), x=gain_scaled.tolist(),
                         orientation="h", name="Gain 重要性（缩放）",
                         marker_color=C_GREY, opacity=0.7))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="AUC 下降（越高越重要）"),
                      title="置换 vs Gain 重要性（Top 15）", height=560,
                      yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                      legend=dict(x=0.65, y=0.1, font=dict(size=11)))
    return chart_div(fig)


def perm_importance(model, df, features, top, n_repeat=10, seed=3):
    rng = np.random.default_rng(seed)
    y = df["survive_144"].values
    base = roc_auc_score(y, model.predict_proba(df[features])[:, 1])
    rows = {}
    for f in top:
        drops = []
        orig = df[f].values.copy()
        for _ in range(n_repeat):
            v = orig.copy()
            rng.shuffle(v)
            df[f] = v
            p = model.predict_proba(df[features])[:, 1]
            drops.append(base - roc_auc_score(y, p))
        df[f] = orig
        rows[f] = dict(mean=float(np.mean(drops)), sd=float(np.std(drops)))
    return base, pd.DataFrame(rows).T


def main():
    ev_all, df, d = load()
    # 口径信息全部从数据推导，避免换 universe 后标题/表头写死。
    is_ashare = EVENTS_NAME.startswith("ashare")
    freq = os.environ.get("FREQ", "30m" if is_ashare else "1h")
    n_coin = int(df["code"].nunique())
    n_sym = n_coin if is_ashare else n_coin + 1
    unit = "只 A 股" if is_ashare else "币种"
    universe_label = (f"{n_coin} {unit}" if is_ashare
                      else f"{n_coin} 币种 + _MARKET")
    market_desc = (f"{n_coin} 只 A 股" if is_ashare
                   else f"{n_sym} 个 USDT 永续（{n_coin} 币种 + _MARKET）")
    _t = pd.to_datetime(df["event_time"])
    span = f"{_t.min():%Y-%m} ~ {_t.max():%Y-%m}"
    _bar_days = {"1h": "6 天", "30m": "3 天", "15m": "1.5 天", "5m": "0.5 天"}
    days_note = f"（{_bar_days[freq]}）" if freq in _bar_days else ""
    y, p = df["survive_144"].values.astype(float), df["pred"].values
    m = metrics(y, p)
    e, cal = ece(y, p)
    auc_ci, brier_ci, auc_se = boot_ci(y, p)
    brier_base = float(m["base"] * (1 - m["base"]))
    bss = 1 - m["brier"] / brier_base

    g = df.groupby("month").apply(lambda x: pd.Series({
        "auc": roc_auc_score(x["survive_144"], x["pred"]) if x["survive_144"].nunique() > 1 else np.nan,
        "n": len(x), "base": x["survive_144"].mean(), "pred": x["pred"].mean(),
    }), include_groups=False).reset_index()

    div_monthly = fig_monthly(g)
    div_drift = fig_drift(g)
    div_calibration = fig_calibration(y, p, e)
    div_dist = fig_dist(df)
    div_decile, dec = fig_decile(df)
    div_by_coin, g_coin = fig_by_coin(df, m["auc"])
    div_by_type, g_type = fig_by_type(df, m["auc"])
    div_lift, t_lift = fig_lift(df)
    div_pr, ap_pr = fig_pr(df)
    div_thr = fig_threshold_curves(df)

    feats = d["features"]
    model = d["model"]
    gain = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    last_month = str(df["month"].max())
    df_last = df[df["month"] == last_month].reset_index(drop=True)
    if len(df_last) >= 50 and df_last["survive_144"].nunique() == 2:
        base_auc_perm, perm = perm_importance(model, df_last, feats, list(gain.head(15).index))
    else:
        base_auc_perm, perm = float("nan"), None
    if perm is not None:
        perm["gain"] = gain.reindex(perm.index).values
    div_importance = fig_importance(gain, perm)

    ref_type = df.groupby("bsp_type")["survive_144"].transform("mean").values
    ref_rows = [
        ("常数（永远猜基准率）", m["base"], 0.5, brier_base, 0.0),
        ("仅类型（类型均值，样本内上界）", m["base"], roc_auc_score(y, ref_type),
         brier_score_loss(y, np.clip(ref_type, 1e-6, 1 - 1e-6)), 0.0),
        ("完整模型（XGBoost）", m["base"], m["auc"], m["brier"], bss),
    ]

    df["year"] = df["month"].str[:4]
    yr = df.groupby("year").apply(lambda x: pd.Series({
        "n": len(x), "base": x["survive_144"].mean(),
        "auc": roc_auc_score(x["survive_144"], x["pred"]),
        "brier": brier_score_loss(x["survive_144"], x["pred"]),
        "pred": x["pred"].mean(),
    }), include_groups=False).reset_index()

    buy = df.groupby("is_buy").apply(lambda x: pd.Series({
        "n": len(x), "base": x["survive_144"].mean(),
        "auc": roc_auc_score(x["survive_144"], x["pred"]),
    }), include_groups=False).reset_index()
    buy["label"] = buy["is_buy"].map({1: "买点", 0: "卖点"})
    dl = df.copy()
    dl["bucket"] = pd.cut(dl["delay_bars"], [-1, 0, 2, 1000], labels=["0 根", "1-2 根", "≥3 根"])
    dl = dl.groupby("bucket", observed=True).apply(lambda x: pd.Series({
        "n": len(x), "base": x["survive_144"].mean(),
        "auc": roc_auc_score(x["survive_144"], x["pred"]) if x["survive_144"].nunique() > 1 else np.nan,
    }), include_groups=False).reset_index()

    md = []
    for mo, gm in df.groupby("month"):
        if len(gm) < 40:
            continue
        top = gm[gm["pred"] >= gm["pred"].quantile(.9)]
        md.append(dict(month=mo, n=len(gm), base=gm["survive_144"].mean(),
                       top=top["survive_144"].mean()))
    md = pd.DataFrame(md)

    conf = []
    for name, cov in [("q60（保留 40%）", .40), ("q80（保留 20%）", .20)]:
        thr = df["pred"].quantile(1 - cov)
        sel = df["pred"] >= thr
        tp = int(((sel) & (df.survive_144 == 1)).sum())
        fp = int(((sel) & (df.survive_144 == 0)).sum())
        fn = int((~sel & (df.survive_144 == 1)).sum())
        tn = int((~sel & (df.survive_144 == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        conf.append(dict(name=name, thr=thr, n=int(sel.sum()), prec=prec, rec=rec,
                         f1=2 * prec * rec / max(prec + rec, 1e-9), tp=tp, fp=fp, fn=fn, tn=tn))

    ext = None if ASHARE_EVENTS == EVENTS_NAME else external_block(ASHARE_EVENTS, ASHARE_MODEL)
    div_compare = ""
    if ext is not None:
        div_compare = fig_compare_decile(
            dec["y"], ext["dec"],
            f"加密 {n_coin} 币（AUC {m['auc']:.2f}）", f"A 股 {ext['n_sym']} 只（AUC {ext['auc']:.2f}）")

    old = None if is_ashare else load_old_model()

    score_q = {q: float(df["pred"].quantile(q)) for q in (0.4, 0.6, 0.8, 0.9)}
    top_decile_note = (
        f"月度前 10% 存活率中位数 {md['top'].median():.1%}，月度基准率中位数 {md['base'].median():.1%}"
        f"（{len(md)} 个月）。" if len(md)
        else "月度前 10% 统计已跳过：多数月份样本不足 40。")

    def pct(x, d_=1):
        return "N/A" if pd.isna(x) else f"{x * 100:.{d_}f}%"

    def table(headers, rows, num_from=1):
        h = "".join(f"<th style='text-align:right'>{c}</th>" if i >= num_from else f"<th>{c}</th>"
                    for i, c in enumerate(headers))
        body = "".join("<tr>" + "".join(
            f"<td class='num'>{c}</td>" if i >= num_from else f"<td>{c}</td>"
            for i, c in enumerate(r)) + "</tr>" for r in rows)
        return f"<table><tr>{h}</tr>{body}</table>"

    mon_rows = [(r.month, int(r.n), f"{r.auc:.3f}", f"{r.base:.1%}", f"{r.pred:.1%}")
                for r in g.itertuples()]
    coin_rows = [(r.key, int(r.n), f"{r.base:.1%}", f"{r.auc:.3f}", f"{r.n / len(df):.1%}")
                 for r in g_coin.sort_values("auc", ascending=False).itertuples()]
    cal_rows = [(f"{r.p:.1%}", f"{r.y:.1%}", int(r.n)) for r in cal.itertuples()]
    lift_rows = [(f"{c:.0%}", f"{thr:.3f}", n, f"{prec:.1%}", f"{prec / y.mean():.2f}x", f"{rec:.0%}")
                 for c, thr, n, prec, rec in
                 [(r.cov, df['pred'].quantile(1 - r.cov), r.n, r.prec, r.rec) for r in t_lift.itertuples()]]
    type_rows = [(k, int(r.n), f"{r.surv:.1%}", "N/A" if pd.isna(r.auc) else f"{r.auc:.3f}",
                  f"{r.surv / y.mean():.2f}x") for k, r in g_type.sort_values("surv").iterrows()]
    w_auc = float((g_type["auc"] * g_type["n"]).sum() / g_type["n"].sum())
    yr_rows = [(r.year, int(r.n), f"{r.base:.1%}", f"{r.auc:.3f}", f"{r.brier:.3f}", f"{r.pred:.1%}")
               for r in yr.itertuples()]
    buy_rows = [(r.label, int(r.n), f"{r.base:.1%}", f"{r.auc:.3f}") for r in buy.itertuples()]
    dl_rows = [(str(r.bucket), int(r.n), f"{r.base:.1%}", "N/A" if pd.isna(r.auc) else f"{r.auc:.3f}")
               for r in dl.itertuples()]
    ref_html = "".join(
        f"<tr><td>{a}</td><td class='num'>{b:.1%}</td><td class='num'>{c:.3f}</td>"
        f"<td class='num'>{dd:.3f}</td><td class='num'>{ee:.1%}</td></tr>"
        for a, b, c, dd, ee in ref_rows)
    conf_rows = [(c["name"], f"{c['thr']:.3f}", c["n"], f"{c['prec']:.1%}", f"{c['rec']:.1%}",
                  f"{c['f1']:.3f}", c["tp"], c["fp"], c["fn"], c["tn"]) for c in conf]

    yv = df["survive_144"].values.astype(int)
    pr_rows = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sel = df["pred"].values >= thr
        tp = int((sel & (yv == 1)).sum()); fp = int((sel & (yv == 0)).sum())
        fn = int((~sel & (yv == 1)).sum()); tn = int((~sel & (yv == 0)).sum())
        pp = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
        pr_rows.append((f"{thr:.2f}", int(sel.sum()), f"{pp:.1%}", f"{rr:.1%}",
                        f"{2 * pp * rr / max(pp + rr, 1e-9):.3f}", tp, fp, fn, tn))
    _p, _r, _t = precision_recall_curve(yv, df["pred"].values)
    _f1 = np.divide(2 * _p * _r, _p + _r, out=np.zeros_like(_p), where=(_p + _r) > 0)
    _k = int(np.nanargmax(_f1))
    best_f1 = _f1[_k]
    best_thr = float(_t[min(_k, len(_t) - 1)])
    dec_rows = [(int(r.Index) + 1, int(r.n), f"{r.p:.1%}", f"{r.y:.1%}") for r in dec.itertuples()]
    imp_rows = [(k, f"{v:.4f}") for k, v in gain.head(18).items()]
    if ext is None:
        cross_bullet = ""
    else:
        cross_bullet = (
            f'<li><b>跨市场</b>：A 股 {ext["n_sym"]} 只 {ASHARE_FREQ} 的 AUC 为 {ext["auc"]:.3f}'
            f' （置信区间 {ext["ci"][0]:.3f}~{ext["ci"][1]:.3f}），与加密方向一致，'
            f' 但只有 {ext["n"]} 个样本，置信区间宽 {(ext["ci"][1] - ext["ci"][0]) / (auc_ci[1] - auc_ci[0]):.1f}倍，Brier 技能分接近 0'
            f' 仅作一致性检查，不是独立验证。</li>')
    if ext is None:
        compare_html = ('<div class="note">未找到 A 股数据集/模型（'
                        f'data/events/{ASHARE_EVENTS}.parquet, data/models/{ASHARE_MODEL}.pkl），跳过对比。'
                        '</div>')
    else:
        cmp_rows = [
            ("标的数", f"{df['code'].nunique()}", f"{ext['n_sym']}"),
            ("评估样本数", f"{m['n']:,}", f"{ext['n']:,}"),
            ("基准率", f"{m['base']:.1%}", f"{ext['base']:.1%}"),
            ("AUC", f"{m['auc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
             f"{ext['auc']:.3f} [{ext['ci'][0]:.3f}, {ext['ci'][1]:.3f}]"),
            ("月度 AUC 均值 ± 标准差", f"{g['auc'].mean():.3f} +/- {g['auc'].std():.3f}",
             f"{ext['m_mean']:.3f} +/- {ext['m_std']:.3f}"),
            ("月数（样本≥8）", f"{len(g)}", f"{ext['months']}"),
            ("Brier / 技能分", f"{m['brier']:.3f} / {bss:.3f}", f"{ext['brier']:.3f} / {ext['bss']:.3f}"),
            ("ECE", f"{e:.3f}", f"{ext['ece']:.3f}"),
            ("十分位极差", f"{dec['y'].iloc[-1] - dec['y'].iloc[0]:.1%}",
             f"{ext['dec'].iloc[-1] - ext['dec'].iloc[0]:.1%}"),
            ("加权类型内 AUC", f"{w_auc:.3f}", f"{ext['w_auc']:.3f}"),
            ("仅类型 AUC（上界）", f"{roc_auc_score(y, ref_type):.3f}", f"{ext['type_only']:.3f}"),
            ("分币种 AUC 范围", f"{g_coin['auc'].min():.3f} ~ {g_coin['auc'].max():.3f}",
             f"{ext['coin_min']:.3f} ~ {ext['coin_max']:.3f}"),
        ]
        compare_html = table(["指标", f"加密 {universe_label} / {freq}", f"A 股 {ext['n_sym']} 只 / {ASHARE_FREQ}"],
                             [(a, b, c) for a, b, c in cmp_rows], num_from=1) + (
            f'<div class="note">A 股组只能算“方向一致”，不是“独立验证”：'
            f'{ext["n"]} 个样本（为加密的 {ext["n"] / m["n"]:.1%}），{ext["n_sym"]} 只标的，'
            f'{ext["months"]} 个月；置信区间宽度是加密的 '
            f'{(ext["ci"][1] - ext["ci"][0]) / (auc_ci[1] - auc_ci[0]):.1f}倍。'
            f'Brier 技能分 {ext["bss"]:.3f}，几乎不比猜基准率好。'
            f'加密的 Brier 技能分为 {bss:.3f}。</div>')
    top_gain = str(gain.index[0])
    if perm is None:
        perm_note = (f'<div class="note">已跳过置换重要性：最后一个月（{last_month}）只有 '
                     f'{len(df_last)} 个样本，不足以做稳定的置换检验。'
                     f'仅展示 Gain 重要性（偏向高基数特征）。</div>')
        perm_table = table(["特征", "gain"], [(k, f"{v:.4f}") for k, v in gain.head(15).items()])
    else:
        top_perm = str(perm.sort_values("mean", ascending=False).index[0])
        perm_note = (
            f'<div class="note">Gain 最高：<code>{top_gain}</code> (perm={perm.loc[top_gain, "mean"]:.4f}); '
            f'置换最高：<code>{top_perm}</code> （AUC 从 {base_auc_perm:.3f} 降到 '
            f'{base_auc_perm - perm.loc[top_perm, "mean"]:.3f}). '
            f'两种排序不一致，说明特征存在冗余；负值是 {len(df_last)} 个样本带来的噪声。</div>')
        perm_table = table(["特征", "置换重要性（AUC 下降，均值 ± 标准差）", "gain"],
                           [(k, f"{r['mean']:.4f} +/- {r['sd']:.3f}", f"{r['gain']:.4f}")
                            for k, r in perm.sort_values("mean", ascending=False).iterrows()])

    compare_old_html = ""
    if old is not None:
        d_n = (m["n"] / old["n"] - 1) * 100
        d_auc = (g["auc"].mean() - old["m_mean"]) * 100
        d_vol = (g["auc"].std() / old["m_std"] - 1) * 100 if old["m_std"] else float("nan")
        compare_old_html = f"""
<h2>10. 新旧模型对比（{old['n_coin']} 币种 → {n_coin} 币种）</h2>
<div class="note">旧模型使用 {old['n_coin']} 币种训练，新模型扩展到 {n_coin} 币种（{n_sym} 个标的，含 _MARKET）。事件样本增加 {d_n:+.0f}%，月度 AUC 均值变化 {d_auc:+.2f} 个百分点，月度波动变化 {d_vol:+.0f}%。</div>
{table(["指标", f"旧模型 ({old['n_coin']} 币种)", f"新模型 ({n_coin} 币种)", "变化"], [
    ("事件样本数", f"{old['n']:,}", f"{m['n']:,}", f"+{(m['n']/old['n']-1)*100:.0f}%"),
    ("币种数", str(old['n_coin']), str(df['code'].nunique()), f"+{df['code'].nunique()-old['n_coin']}"),
    ("整体 AUC", f"{old['auc']:.4f}", f"{m['auc']:.4f}", f"{m['auc']-old['auc']:+.4f}"),
    ("AUC 95% CI", f"{old['ci'][0]:.3f}~{old['ci'][1]:.3f}", f"{auc_ci[0]:.3f}~{auc_ci[1]:.3f}", ""),
    ("月度 AUC 均值", f"{old['m_mean']:.4f}", f"{g['auc'].mean():.4f}", f"{g['auc'].mean()-old['m_mean']:+.4f}"),
    ("月度 AUC 标准差", f"{old['m_std']:.4f}", f"{g['auc'].std():.4f}", f"{g['auc'].std()-old['m_std']:+.4f}"),
    ("AUC 最小值", f"{old['m_min']:.4f}", f"{g['auc'].min():.4f}", f"{g['auc'].min()-old['m_min']:+.4f}"),
    ("AUC 最大值", f"{old['m_max']:.4f}", f"{g['auc'].max():.4f}", f"{g['auc'].max()-old['m_max']:+.4f}"),
    ("基准率", f"{old['base']:.1%}", f"{m['base']:.1%}", f"{m['base']-old['base']:+.1%}"),
    ("Brier 分", f"{old['brier']:.4f}", f"{m['brier']:.4f}", f"{m['brier']-old['brier']:+.4f}"),
    ("Brier 技能分", f"{old['bss']:.3f}", f"{bss:.3f}", f"{bss-old['bss']:+.3f}"),
    ("月均测试笔数", f"{old['n']//old['months']}", f"{m['n']//len(g)}", f"+{m['n']//len(g)-old['n']//old['months']}"),
    ("BUY 样本", f"{old['buy_n']:,}", f"{int((df['is_buy']==1).sum()):,}", f"+{(df['is_buy']==1).sum()-old['buy_n']:,}"),
    ("SELL 样本", f"{old['sell_n']:,}", f"{int((df['is_buy']==0).sum()):,}", f"+{(df['is_buy']==0).sum()-old['sell_n']:,}"),
    ("BUY survive 率", f"{old['buy_surv']:.1%}", f"{df[df['is_buy']==1]['survive_144'].mean():.1%}", f"{df[df['is_buy']==1]['survive_144'].mean()-old['buy_surv']:+.1%}"),
    ("SELL survive 率", f"{old['sell_surv']:.1%}", f"{df[df['is_buy']==0]['survive_144'].mean():.1%}", f"{df[df['is_buy']==0]['survive_144'].mean()-old['sell_surv']:+.1%}"),
], num_from=1)}
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>存活模型报告：{universe_label} {freq}</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>{CSS}</style></head><body>
<h1>存活模型报告：{universe_label} {freq}</h1>
<div class="sub">生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M} ｜ K 线：{market_desc}，{freq}，{span}<br>数据：{EVENTS_NAME}；模型：{MODEL_NAME}（逐月 walk-forward 样本外）；
标签：<code>survive_144</code> = 信号后 144 根 bar{days_note}内未被擦除。</div>

<div class="warn">
<b>判别力强 ≠ 可交易。</b>
本报告所有指标衡量的都是判别能力（AUC、校准、排序）。
E002 置换检验：按分数留前 20% 的交易夏普 −0.74，随机抽样中位数 +0.25。
本报告回答的是“模型能否区分存活”，不是“能不能赚钱”。
</div>

<h2>1. 摘要</h2>
<div class="card">
  <div class="grid">
    <div><span class="k">样本外 AUC</span><b>{m['auc']:.3f}</b></div>
    <div><span class="k">AUC 95% 置信区间</span><b>{auc_ci[0]:.3f} ~ {auc_ci[1]:.3f}</b></div>
    <div><span class="k">样本数</span><b>{m['n']:,}</b></div>
    <div><span class="k">基准率</span><b>{pct(m['base'])}</b></div>
    <div><span class="k">Brier 分</span><b>{m['brier']:.3f}</b></div>
    <div><span class="k">Brier 技能分</span><b>{bss:.3f}</b></div>
    <div><span class="k">PR-AUC</span><b>{m['ap']:.3f}</b></div>
    <div><span class="k">LogLoss</span><b>{m['logloss']:.3f}</b></div>
    <div><span class="k">ECE</span><b>{e:.3f}</b></div>
    <div><span class="k">MCC（p≥0.5）</span><b>{m['mcc']:.3f}</b></div>
    <div><span class="k">月度 AUC 均值</span><b>{g['auc'].mean():.3f} +/- {g['auc'].std():.3f}</b></div>
    <div><span class="k">月数</span><b>{len(g)}</b></div>
  </div>
</div>

<h2>2. 方法</h2>
<table>
<tr><th>项目</th><th>说明</th></tr>
<tr><td>样本</td><td>原始信号 {len(ev_all):,} 个 → 去掉 2021 年前 ({int((ev_all.event_time < '2021-01-01').sum()):,})、去掉右删失 ({int(ev_all.survive_144.isna().sum())}) → <b>{m['n']:,}</b> 个样本外预测</td></tr>
<tr><td>标签</td><td><code>survive_144</code>：信号后超过 144 根 bar 才被擦除或从未被擦除 = 1；144 根内被擦除 = 0</td></tr>
<tr><td>训练</td><td>逐月 walk-forward：每个月用此前全部数据训练，预测当月，共 {len(g)} 个月</td></tr>
<tr><td>模型</td><td>XGBoost 分类器，{len(feats)} 个特征， n_estimators=150, max_depth=5, lr=0.05</td></tr>
<tr><td>特征</td><td>缠论结构 + 价格行为 + 动态指标，共 {len(feats)} 列</td></tr>
<tr><td>评估</td><td>只看样本外（当月模型从未见过当月数据）</td></tr>
</table>

<h2>3. 判别力</h2>
<h3>3.1 月度 AUC</h3>
<div class="sub">AUC 低于 0.6 的月份 {int((g['auc'] < 0.6).sum())} 个，最小 {g['auc'].min():.3f}，最大 {g['auc'].max():.3f}。</div>
<div class="chart-container">{div_monthly}</div>
<details><summary>月度明细（{len(g)} 个月）</summary>
{table(["月份", "样本数", "AUC", "实际", "预测"], mon_rows)}</details>

<h3>3.2 十分位单调性</h3>
<div class="chart-container">{div_decile}</div>
{table(["十分位", "样本数", "平均预测", "实际存活率"], dec_rows)}

<h3>3.3 分年度</h3>
{table(["年份", "样本数", "基准率", "AUC", "Brier", "平均预测"], yr_rows)}

<h2>4. 校准</h2>
<div class="two">
  <div class="chart-container">{div_calibration}</div>
  <div class="chart-container">{div_dist}</div>
</div>
<details><summary>校准分箱</summary>
{table(["平均预测", "实际", "样本数"], cal_rows)}</details>

<h3>4.1 基准率漂移</h3>
<div class="chart-container">{div_drift}</div>
<div class="note">整体实际均值 {g['base'].mean():.1%}，平均预测 {g['pred'].mean():.1%}；
月度绝对偏差中位数 {float((g['pred'] - g['base']).abs().median()):.1%}.</div>

<h2>5. 参照模型</h2>
<table>
<tr><th>预测器</th><th style="text-align:right">基准率</th><th style="text-align:right">AUC</th>
<th style="text-align:right">Brier</th><th style="text-align:right">Brier 技能分</th></tr>
{ref_html}
</table>

<h2>6. 分组表现</h2>
<h3>6.1 分币种</h3>
<div class="chart-container">{div_by_coin}</div>
<details><summary>分币种明细</summary>
{table(["币种", "样本数", "存活率", "AUC", "占比"], coin_rows)}</details>

<h3>6.2 分信号类型</h3>
<div class="chart-container">{div_by_type}</div>
{table(["类型", "样本数", "存活率", "类型内 AUC", "相对基准"], type_rows)}

<h3>6.3 买卖方向与滞后</h3>
<div class="two">
<div>{table(["方向", "样本数", "基准率", "AUC"], buy_rows)}</div>
<div>{table(["滞后", "样本数", "基准率", "AUC"], dl_rows)}</div>
</div>

<h2>7. 阈值分析</h2>
<div class="chart-container">{div_lift}</div>
{table(["保留比例", "阈值", "样本数", "存活率", "相对基准", "召回率"], lift_rows)}
<h3>7.1 精确率-召回率</h3>
<div class="two">
<div class="chart-container">{div_pr}</div>
<div class="note">
<p>样本外 n={len(df)}，基准率 {y.mean():.1%}；AP=<b>{ap_pr:.3f}</b>。最佳 F1=<b>{best_f1:.3f}</b>，对应阈值 {best_thr:.3f}。</p>
</div>
</div>
<div class="chart-container">{div_thr}</div>
{table(["阈值", "保留数", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"], pr_rows)}

<h3>7.2 常用闸门的混淆矩阵</h3>
{table(["闸门", "阈值", "样本数", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"], conf_rows)}
<div class="note">分数分位：q40={score_q[0.4]:.3f}, q60={score_q[0.6]:.3f}, q80={score_q[0.8]:.3f}, q90={score_q[0.9]:.3f}.
{top_decile_note}</div>

<h2>8. 特征重要性：Gain vs 置换</h2>
<div class="sub">最后一个月 {last_month}，n={len(df_last)}。</div>
<div class="chart-container">{div_importance}</div>
{perm_table}
{perm_note}
<details><summary>Gain 重要性 Top 18</summary>
{table(["特征", "gain"], imp_rows)}</details>

<h2>9. A 股对比</h2>
{compare_html}
<h3>9.1 十分位对比</h3>
<div class=”chart-container”>{div_compare}</div>

{compare_old_html}

<h2>11. 结论</h2>
<ul>
<li><b>是</b>：模型有排序能力（AUC {m['auc']:.3f}，置信区间 {auc_ci[0]:.3f}~{auc_ci[1]:.3f}），{len(g)} 个月里表现稳定。</li>
<li><b>注意</b>：AUC 很大一部分来自信号类型（各类存活率从 {g_type['surv'].min():.0%} 到 {g_type['surv'].max():.0%}），类型内 AUC 只有 {w_auc:.2f}。</li>
<li><b>否</b>：”存活”不等于赚钱。本报告评估的是标签，不是交易盈亏。</li>
<li><b>否</b>：分数高不等于”该交易”。E002 置换检验显示，按分数筛选后的交易表现不如随机。</li>
{cross_bullet}
</ul>

<h2>12. 复现</h2>
<pre><code>cd chan-notes
.venv/bin/python scripts/train_wf_oos.py {EVENTS_NAME} {MODEL_NAME}
.venv/bin/python scripts/make_survival_model_report.py</code></pre>
<div class="sub" style="margin-top:20px">生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}；脚本：scripts/make_survival_model_report.py</div>
</body></html>"""

    out = REPORTS / f"survival_model_report{SFX}.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"AUC={m['auc']:.4f} CI[{auc_ci[0]:.3f},{auc_ci[1]:.3f}] base={m['base']:.4f} "
          f"brier={m['brier']:.4f} bss={bss:.3f} ap={m['ap']:.3f} ece={e:.3f} n={m['n']}")
    print(f"type-only AUC={roc_auc_score(y, ref_type):.4f} within-type weighted={w_auc:.3f}")
    print(f"top10% prec={conf[1]['prec']:.3f} recall={conf[1]['rec']:.3f}")
    if perm is not None:
        print("perm top:", perm.sort_values('mean', ascending=False).head(5)['mean'].round(4).to_dict())
    print("perm base auc (last month):", round(base_auc_perm, 4))


if __name__ == "__main__":
    main()
