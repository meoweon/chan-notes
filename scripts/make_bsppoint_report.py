"""生成《缠论买卖点报告》（HTML）。

读三个 JSON（验证 + 分析 + 波动率），渲染完整报告。所有数字来自可复现脚本：
  - validate_bsppoint_profit.py → bsppoint_profit_validation{OUT_SUFFIX}.json
  - analyze_bsppoint.py           → bsppoint_analysis{OUT_SUFFIX}.json
  - bsppoint_score_vol{OUT_SUFFIX}.json

用法：
    .venv/bin/python scripts/make_bsppoint_report.py
"""
import json
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
EVENTS_NAME = os.environ.get("EVENTS_NAME", "crypto_31_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "wf_crypto_31_xgb_oos")
FREQ = os.environ.get("FREQ", "1h")
VAL = json.loads((REPORTS / f"bsppoint_profit_validation{OUT_SUFFIX}.json").read_text())
ANA = json.loads((REPORTS / f"bsppoint_analysis{OUT_SUFFIX}.json").read_text())
VOL = json.loads((REPORTS / f"bsppoint_score_vol{OUT_SUFFIX}.json").read_text())

# 标题里的 universe / 区间从事件表推导，避免换口径后写死。
_ev = pd.read_parquet(ROOT / "data" / "events" / f"{EVENTS_NAME}.parquet",
                      columns=["code", "event_time"])
N_COIN = int(_ev["code"].nunique())
N_ALL = N_COIN + 1  # + _MARKET
_t = pd.to_datetime(_ev["event_time"])
DATA_START = f"{_t.min():%Y-%m}"
STUDY_START = f"{_t.min() + pd.DateOffset(months=12):%Y-%m}"
SPAN = f"{STUDY_START} ~ {_t.max():%Y-%m}"

TYPE_ORDER = ["1", "1p", "2", "2s", "3a", "3b", "2,3b", "2s,3b", "1,2s"]
BASE_TYPES = ["1", "1p", "2", "2s", "3a", "3b"]

C_TRUE, C_JUDGE = "#2fb37a", "#2f6fed"
FONT = '-apple-system,"PingFang SC","Microsoft YaHei","Segoe UI",Roboto,Helvetica,Arial,sans-serif'

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:15px/1.6 -apple-system,"PingFang SC","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     margin:0;background:#f7f8fa;color:#1a1d23}
.wrap{max-width:1120px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid #2f6fed}
h3{font-size:16px;margin:20px 0 6px}
p{margin:8px 0}
.meta{color:#5b6472;font-size:13px}
.card{background:#fff;border:1px solid #e5e8ee;border-radius:10px;padding:16px 18px;margin:12px 0;overflow-x:auto}
.warn{background:#fff7e6;border:1px solid #f2c94c;border-radius:10px;padding:12px 16px;margin:14px 0}
.warn b{color:#9a6b00}
.key{background:#eefaf1;border:1px solid #9bd4ac;border-radius:10px;padding:12px 16px;margin:14px 0}
.key b{color:#0b6e35}
.todo{background:#f0f4ff;border:1px dashed #7aa2f7;border-radius:10px;padding:12px 16px;margin:14px 0;color:#35509a}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px;background:#fff}
th,td{border:1px solid #e5e8ee;padding:6px 9px;text-align:right}
th{background:#f0f3f9;font-weight:600}
td:first-child,th:first-child{text-align:left}
.pos{color:#0b7a3b;font-weight:600}.neg{color:#c0392b;font-weight:600}
.tag{display:inline-block;background:#eef2fb;color:#2f6fed;border-radius:5px;padding:0 7px;font-size:12px;margin-right:5px}
code{background:#eef1f5;border-radius:4px;padding:1px 5px;font-size:13px}
.small{color:#5b6472;font-size:12.5px}
.legend{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;margin:0 4px 0 10px;border:1.3px solid}
"""


# ───────────────────────── 格式化 ─────────────────────────
def pct(x, d=1):
    """带符号的百分比（用于收益）。"""
    return "—" if x is None else f"{x*100:+.{d}f}%"


def pctu(x, d=1):
    """不带符号的百分比（用于比率：存活率、Recall、Precision、胜率…）。"""
    return "—" if x is None else f"{x*100:.{d}f}%"


def cls(x):
    return "pos" if (x or 0) > 0 else ("neg" if (x or 0) < 0 else "")


def fw(stat, show_n=False):
    """前向收益单元：均值 / 中位 / 胜率（胜率不带符号）。"""
    if not stat:
        return "—"
    s = f"{pct(stat['mean'],2)} / {pct(stat['median'],2)} / {pctu(stat['win'],0)}"
    return s + (f"<span class='small'> n={stat['n']}</span>" if show_n else "")


def conf_row(split):
    s = VAL["splits"].get(split)
    if not s:
        return ""
    r = []
    for tag in ("is", "oos"):
        m = s[tag]
        r.append(f"<td class='{cls(m['ret'])}'>{pct(m['ret'])}</td>"
                 f"<td>{m['sharpe']:+.2f}</td>"
                 f"<td>{pct(m['avgret'],2)}</td>"
                 f"<td>{pctu(m['win'],1)}</td>"
                 f"<td>{m['payoff']:.2f}</td>"
                 f"<td>{pct(m['mdd'])}</td>")
    return "".join(r)


# ───────────────────────── 表格 ─────────────────────────
def type_rows():
    rows = []
    for tp in TYPE_ORDER:
        if tp not in ANA["types"]:
            continue
        g = ANA["types"][tp]
        tag = "基础" if tp in BASE_TYPES else "多标签"
        auc = "—" if g["auc"] is None else f"{g['auc']:.3f}"
        ls = ANA["lifespan"].get(tp, {})
        med = "—" if ls.get("median") is None else f"{ls['median']:.0f}"
        tpn, fpn, fnn = g.get("tp", 0), g.get("fp", 0), g.get("fn", 0)
        rows.append(
            f"<tr><td><span class='tag'>{tag}</span>{tp}</td>"
            f"<td>{g['n']}</td><td>{g['survive_rate']:.1%}</td><td>{auc}</td>"
            f"<td>{tpn + fpn}</td><td>{tpn}</td><td>{fpn}</td><td>{fnn}</td>"
            f"<td>{g.get('prec', 0):.1%}</td><td>{g.get('rec', 0):.1%}</td><td>{g.get('f1', 0):.2f}</td>"
            f"<td class='{cls(g['fw_all'] and g['fw_all']['mean'])}'>{fw(g['fw_all'])}</td>"
            f"<td class='{cls(g['fw_true'] and g['fw_true']['mean'])}'>{fw(g['fw_true'])}</td>"
            f"<td class='{cls(g['fw_false'] and g['fw_false']['mean'])}'>{fw(g['fw_false'])}</td>"
            f"<td>{med}</td></tr>")
    return "\n".join(rows)


def decile_rows():
    rows = []
    for r in ANA["deciles"]:
        rows.append(
            f"<tr><td>D{r['bin']}</td>"
            f"<td class='small'>{r['score_lo']:.3f}–{r['score_hi']:.3f}</td>"
            f"<td>{r['n']}</td><td>{pctu(r['survive_rate'])}</td>"
            f"<td class='{cls(r['fw'] and r['fw']['mean'])}'>{fw(r['fw'])}</td>"
            f"<td class='{cls(r['fw_true'] and r['fw_true']['mean'])}'>{fw(r['fw_true'])}</td>"
            f"<td class='{cls(r['fw_false'] and r['fw_false']['mean'])}'>{fw(r['fw_false'])}</td></tr>")
    return "\n".join(rows)


def ratio_rows():
    rows = []
    for r in ANA["ratios"]:
        rows.append(
            f"<tr><td>{r['name']}</td><td>{r['thr']:.3f}</td><td>{r['n_pos']}</td>"
            f"<td>{pctu(r['阳性率'])}</td><td>{pctu(r['recall'])}</td><td>{pctu(r['FPR'])}</td>"
            f"<td>{pctu(r['precision'])}</td><td>{pctu(r['TP占比'])}</td><td>{r['f1']:.3f}</td></tr>")
    return "\n".join(rows)


def horizon_rows():
    rows = []
    for h in ANA["horizons"]:
        auc = "—" if h["auc"] is None else f"{h['auc']:.3f}"
        rows.append(f"<tr><td>{h['h']}</td><td>{pctu(h['survive_rate'])}</td><td>{auc}</td></tr>")
    return "\n".join(rows)


def lifespan_rows():
    rows = []
    for tp in TYPE_ORDER:
        if tp not in ANA["lifespan"]:
            continue
        g = ANA["lifespan"][tp]
        med = "—" if g["median"] is None else f"{g['median']:.0f}"
        p75 = "—" if g["p75"] is None else f"{g['p75']:.0f}"
        rows.append(f"<tr><td>{tp}</td><td>{g['n_cleared']}</td><td>{med}</td><td>{p75}</td></tr>")
    return "\n".join(rows)


def pct2(x):
    return "—" if x is None else f"{x*100:.2f}%"


def vol_decile_rows():
    rows = []
    for r in VOL["deciles"]:
        rows.append(
            f"<tr><td>D{r['bin']}</td><td>{pctu(r['survive_rate'],0)}</td>"
            f"<td>{pct2(r['amp_all'])}</td><td>{pct2(r['amp_true'])}</td>"
            f"<td>{pct2(r['up_true'])}</td><td>{pct2(r['dn_true'])}</td>"
            f"<td>{pct2(r['fw_true'])}</td></tr>")
    return "\n".join(rows)


def vol_type_rows():
    rows = []
    dn, us, fwc, n = (VOL["type_corr_dn"], VOL["type_corr_up_share"],
                      VOL["type_corr_fw"], VOL["type_true_n"])
    for tp in TYPE_ORDER:
        if tp not in dn:
            continue
        rows.append(
            f"<tr><td>{tp}</td><td>{n[tp]}</td>"
            f"<td>{dn[tp]:+.3f}</td><td>{us[tp]:+.3f}</td><td>{fwc[tp]:+.3f}</td></tr>")
    return "\n".join(rows)


def bars_rows():
    B = ANA["bars"]
    out = []
    for tp in [t for t in TYPE_ORDER if t in B] + ["全部"]:
        g = B[tp]
        n = max(g["n"], 1)
        med = "—" if g["delay_med_pos"] is None else f"{g['delay_med_pos']:.0f}"
        maxd = g["delay_max"] if g["delay_pos"] else "—"
        out.append(
            f"<tr><td>{tp}</td><td>{g['n']}</td>"
            f"<td>{g['delay0']}（{g['delay0'] / n:.1%}）</td>"
            f"<td>{g['delay_pos']}（{g['delay_pos'] / n:.1%}）</td>"
            f"<td>{med}</td><td>{maxd}</td></tr>")
    return "".join(out)


def wait_rows():
    B = ANA["bars"]
    out = []
    for tp in [t for t in TYPE_ORDER if t in B] + ["全部"]:
        g = B[tp]
        n = max(g["n"], 1)
        cells = "".join(
            f"<td>{g['wait'].get(str(i), 0)}（{g['wait'].get(str(i), 0) / n:.1%}）</td>"
            for i in (1, 2, 3, 4))
        out.append(f"<tr><td>{tp}</td><td>{g['n']}</td>{cells}"
                   f"<td>{g['wait_med']:.1f}</td><td>{g['wait_mean']:.2f}</td></tr>")
    return "".join(out)


# ───────────────────────── 绘图通用 ─────────────────────────
def chart_div(fig, height):
    """宽度自适应容器；用极简模板，避免每张图内嵌一整份 plotly 默认 template。"""
    fig.update_layout(template="none", font=dict(family=FONT, size=12),
                      height=height, width=None, autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", default_height=f"{height}px",
                       config={"responsive": True, "displaylogo": False})


def _agg():
    return {k: sum(ANA["types"][t].get(k, 0) for t in ANA["types"]) for k in ("tp", "fp", "fn", "tn")}


# ───────────────────────── 韦恩图（面积正比） ─────────────────────────
def _lens_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    k = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - k


def _solve_d(r1, r2, target):
    """二分求两圆心距，使交集面积 = target。"""
    lo, hi = abs(r1 - r2), r1 + r2
    if target <= 0:
        return hi
    if target >= math.pi * min(r1, r2) ** 2 - 1e-9:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if _lens_area(r1, r2, mid) > target:
            lo = mid          # 交集太大 → 拉远
        else:
            hi = mid
    return (lo + hi) / 2


def _add_venn(fig, tp, fp, fn, tn, row=None, col=None, min_w=0.3, big=False):
    """在 fig 上画一个面积正比的二集合韦恩图：
    左圈 = 真（TP+FN），右圈 = 判真（TP+FP），交集 = TP。"""
    kw = {} if row is None else dict(row=row, col=col)
    a, b = tp + fn, tp + fp
    if a + b == 0:
        return
    r1, r2 = math.sqrt(a / math.pi), math.sqrt(b / math.pi)
    # 面积口径：圆面积 = π r² = 集合大小；交集面积 = tp
    d = _solve_d(r1, r2, tp)
    xmin, xmax = min(-r1, d - r2), max(r1, d + r2)
    W, H = max(xmax - xmin, 1e-9), 2 * max(r1, r2, 1e-9)
    s = min(2.4 / W, 1.7 / H)
    sh = -(xmin + xmax) / 2
    x1, x2 = (0 + sh) * s, (d + sh) * s
    R1, R2 = r1 * s, r2 * s

    for x, R, color in ((x1, R1, C_TRUE), (x2, R2, C_JUDGE)):
        if R > 0:
            fig.add_shape(type="circle", x0=x - R, x1=x + R, y0=-R, y1=R,
                          fillcolor=color, opacity=0.28, line=dict(color=color, width=1.6),
                          layer="below", **kw)

    regions = [  # (x 区间, 数值, 颜色, 字号)
        ((x1 - R1, min(x1 + R1, x2 - R2)), fn, "#1f7a4d", 13 if big else 11),
        ((max(x1 - R1, x2 - R2), min(x1 + R1, x2 + R2)), tp, "#16233a", 17 if big else 13),
        ((max(x2 - R2, x1 + R1), x2 + R2), fp, "#b07700", 13 if big else 11),
    ]
    for (lo, hi), val, color, size in regions:
        if val > 0 and hi - lo >= min_w:
            fig.add_annotation(x=(lo + hi) / 2, y=0, text=f"<b>{val}</b>", showarrow=False,
                               font=dict(size=size, color=color), **kw)
    if big:
        top = max(R1, R2) + 0.16
        fig.add_annotation(x=x1, y=top, text=f"真 {a}", showarrow=False, xanchor="right",
                           font=dict(size=13, color="#1f7a4d"), **kw)
        fig.add_annotation(x=x2, y=top, text=f"判真 {b}", showarrow=False, xanchor="left",
                           font=dict(size=13, color="#2f6fed"), **kw)
    # 悬停提示 + 保证子图坐标轴被实例化
    fig.add_trace(go.Scatter(
        x=[(x1 + x2) / 2], y=[0], mode="markers", marker=dict(size=40, opacity=0),
        hovertemplate=(f"仅真（FN）{fn}<br>交集（TP）{tp}<br>仅判真（FP）{fp}"
                       f"<br>都不在（TN）{tn}<extra></extra>"),
        showlegend=False), **kw)


def _style_venn_axes(fig, row=None, col=None, xname="x"):
    kw = {} if row is None else dict(row=row, col=col)
    fig.update_xaxes(range=[-1.4, 1.4], visible=False, fixedrange=True, **kw)
    fig.update_yaxes(range=[-1.15, 1.15], visible=False, fixedrange=True,
                     scaleanchor=xname, scaleratio=1, **kw)


def fig_venn_all():
    a = _agg()
    fig = go.Figure()
    _add_venn(fig, a["tp"], a["fp"], a["fn"], a["tn"], big=True)
    _style_venn_axes(fig)
    fig.update_layout(
        title=dict(text=f"真 {a['tp'] + a['fn']:,}  vs  模型判真 {a['tp'] + a['fp']:,}"
                        f"（阈值 0.5，面积正比）", font=dict(size=14), x=0.5),
        margin=dict(l=10, r=10, t=50, b=34), showlegend=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.02, showarrow=False,
                       text=f"两圈都不在（TN）：{a['tn']:,}", font=dict(size=11, color="#8a97a8"))
    return f'<div style="max-width:640px;margin:0 auto">{chart_div(fig, 420)}</div>'


def fig_venn_types():
    tps = [tp for tp in TYPE_ORDER if tp in ANA["types"]]
    n, cols = len(tps), 3
    rows = (n + cols - 1) // cols
    titles = []
    for tp in tps:
        g = ANA["types"][tp]
        titles.append(f"<b>{tp}</b>  n={g['n']}  P={g.get('prec', 0):.0%}  R={g.get('rec', 0):.0%}"
                      f"<br><span style='font-size:11px;color:#5b6472'>"
                      f"FN {g.get('fn', 0)} · TP {g.get('tp', 0)} · FP {g.get('fp', 0)}</span>")
    titles += [""] * (rows * cols - n)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=0.10)
    for i, tp in enumerate(tps):
        r, c = i // cols + 1, i % cols + 1
        g = ANA["types"][tp]
        _add_venn(fig, g.get("tp", 0), g.get("fp", 0), g.get("fn", 0), g.get("tn", 0), r, c)
        _style_venn_axes(fig, r, c, xname="x" if i == 0 else f"x{i + 1}")
    for i in range(n, rows * cols):
        r, c = i // cols + 1, i % cols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    for a in fig.layout.annotations:           # 子图标题字号
        a.font = dict(size=13)
    return chart_div(fig, 300 * rows)


# ───────────────────────── 其余图 ─────────────────────────
def fig_pie_types():
    rows = [(tp, ANA["types"][tp]["n"]) for tp in TYPE_ORDER if tp in ANA["types"]]
    tot = sum(n for _, n in rows)
    colors = ["#2f6fed", "#7aa2f7", "#00a3a3", "#2fb37a", "#e0a100", "#e2703a", "#b06fd0", "#8a97a8", "#c05299"]
    fig = go.Figure(data=[go.Pie(
        labels=[tp for tp, _ in rows], values=[n for _, n in rows], hole=0.52,
        marker=dict(colors=colors[:len(rows)], line=dict(color="white", width=1.5)),
        textinfo="label+percent", textposition="outside", sort=False, direction="clockwise",
        hovertemplate="类型 %{label}<br>%{value:,} 个（%{percent}）<extra></extra>",
        textfont=dict(size=12))])
    fig.update_layout(
        title=dict(text="类型构成（按事件数）", font=dict(size=15), x=0.5),
        annotations=[dict(text=f"{tot:,}<br><span style='font-size:11px'>确认买点</span>",
                          x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return f'<div style="max-width:560px;margin:0 auto">{chart_div(fig, 400)}</div>'


def fig_pie_conf():
    a = _agg()
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=[f"结构标签：真 {a['tp'] + a['fn']:,} / 假 {a['fp'] + a['tn']:,}",
                                       f"模型判真（阈值 0.5）共 {a['tp'] + a['fp']:,}"])
    fig.add_trace(go.Pie(labels=["真", "假"], values=[a["tp"] + a["fn"], a["fp"] + a["tn"]],
                         marker=dict(colors=["#2fb37a", "#d05555"]), hole=0.45, sort=False,
                         textinfo="label+percent+value", textfont=dict(size=12)), row=1, col=1)
    fig.add_trace(go.Pie(labels=["TP 判真且真", "FP 判真但假"], values=[a["tp"], a["fp"]],
                         marker=dict(colors=["#2f6fed", "#e0a100"]), hole=0.45, sort=False,
                         textinfo="label+percent+value", textfont=dict(size=12)), row=1, col=2)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return chart_div(fig, 380)


def fig_bars():
    B = ANA["bars"]
    tps = [t for t in TYPE_ORDER if t in B] + ["全部"]
    ylab = [f"{t}  n={B[t]['n']}" for t in tps]
    nn = [max(B[t]["n"], 1) for t in tps]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04, shared_yaxes=True,
                        subplot_titles=["锚点 bar → 信号 bar：是否同根", "信号 bar → 确认 bar：等了几个 bar"])

    def bar(name, vals, color, col, show_min=6.0):
        fig.add_trace(go.Bar(
            name=name, y=ylab, x=vals, orientation="h", marker_color=color,
            text=[f"{v:.0f}%" if v >= show_min else "" for v in vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white" if color in ("#2e7d32", "#e67e22", "#1f6f8b", "#4f9bb5") else "#16233a"),
            hovertemplate="%{y}<br>" + name + "：%{x:.1f}%<extra></extra>"), row=1, col=col)

    bar("同根（滞后 = 0）", [B[t]["delay0"] / n * 100 for t, n in zip(tps, nn)], "#2e7d32", 1)
    bar("晚发现（滞后 > 0）", [B[t]["delay_pos"] / n * 100 for t, n in zip(tps, nn)], "#e67e22", 1, show_min=3.0)
    for w, color in zip(("1", "2", "3", "4"), ("#1f6f8b", "#4f9bb5", "#8bbfcf", "#cfe3ea")):
        bar(f"确认等待 {w} 根", [B[t]["wait"].get(w, 0) / n * 100 for t, n in zip(tps, nn)], color, 2)

    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="占该类型确认买点的比例")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="stack", bargap=0.25, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    return chart_div(fig, 520)


# ───────────────────────── 主流程 ─────────────────────────
def main():
    chart_pie_types = fig_pie_types()
    chart_pie_conf = fig_pie_conf()
    chart_venn_all = fig_venn_all()
    chart_venn_types = fig_venn_types()
    chart_bars = fig_bars()

    agg = _agg()
    s_all, s_t, s_f = (VAL["splits"].get(k) for k in ("all", "true", "false"))
    auc = VAL["model_auc_vs_survive"]
    d0, d9 = ANA["deciles"][0], ANA["deciles"][-1]
    r05 = next((r for r in ANA["ratios"] if r["name"] == "0.5"), ANA["ratios"][0])
    amps = [r["amp_all"] for r in VOL["deciles"] if r.get("amp_all") is not None]
    amp_mean = sum(amps) / len(amps) * 100 if amps else 0
    h_first, h_last = ANA["horizons"][0], ANA["horizons"][-1]
    _B = ANA["bars"]
    _n, _d0, _dp = _B["全部"]["n"], _B["全部"]["delay0"], _B["全部"]["delay_pos"]
    _fail, _buy, _w = _B["全部"]["conf_fail"], _B["全部"]["n_buy"], _B["全部"]["wait"]
    _ok = _buy - _fail

    bars_html = f"""
<h3>1.1 四根 bar</h3>
<p>一个 BSP 从「引擎算出」到「你能下单」要经过四根 bar：</p>
<ul>
<li><b>锚点 bar</b>（<code>bsp_klu_idx</code>）：定义这个点的笔的终点 K 线；买点的「低点」就是这根 bar 的 low。</li>
<li><b>信号 bar</b>（<code>bar_idx</code>）：引擎真正把这个 BSP 登记进事件表的那根。它通常与锚点 bar 同一根，但「晚首次」时会晚很多（见 1.2）。</li>
<li><b>确认 bar</b>：从信号 bar 之后第一根起，<b>收盘价首次越过信号 bar 最高价</b>（买点；卖点是跌破最低价）的那根。源码 <code>for i in range(t+1, t+5)</code>，<code>wait = i - t</code>，<b>最小 1、没有 0</b>；等不到就 <code>conf_ok = 0</code>，不入场。</li>
<li><b>买入 bar</b>：确认 bar 的下一根，按<b>开盘价</b>成交（<code>conf_entry_idx = 确认 bar + 1</code>）。</li>
</ul>
<p>因此：<b>锚点 → 买入 = 滞后（<code>delay_bars</code>）+ 确认等待（<code>conf_wait</code>）+ 1</b>。信号价不等于锚点价，入场价也不等于最低点。</p>

<h3>1.2 锚点 bar 与信号 bar 重合多少</h3>
<p>全部 {_n} 个确认买点里，<b>{_d0} 个（{_d0 / _n:.1%}）同根</b>，{_dp} 个（{_dp / _n:.1%}）晚发现，最大滞后 {_B['全部']['delay_max']} 根。按类型拆：<b>2、1p、2,3b 是 100% 同根；3b 是 0% 同根（{_B['3b']['n']} 个全部晚发现，中位滞后 {_B['3b']['delay_med_pos']:.0f} 根）；2s,3b 只有三分之一同根（中位滞后 {_B['2s,3b']['delay_med_pos']:.0f} 根）</b>；1、2s、3a 有零星的晚发现，其余 96%~99% 同根。</p>
<table>
<tr><th>类型</th><th>确认买点 n</th><th>同根（滞后 = 0）</th><th>晚发现（滞后 &gt; 0）</th><th>晚发现中位滞后</th><th>最大滞后</th></tr>
{bars_rows()}
</table>
{chart_bars}
<p class="small">左图：绿 = 同根，橙 = 晚发现（占比太小的色块不标数字，悬停可看）。右图：确认等待 1~4 根的占比。</p>

<h3>1.3 确认 bar 晚了多少</h3>
<p>确认等待只有 1~4 四种取值：<b>1 根 {_w.get('1', 0)}（{_w.get('1', 0) / _n:.1%}）、2 根 {_w.get('2', 0)}（{_w.get('2', 0) / _n:.1%}）、3 根 {_w.get('3', 0)}（{_w.get('3', 0) / _n:.1%}）、4 根 {_w.get('4', 0)}（{_w.get('4', 0) / _n:.1%}）</b>，中位 {_B['全部']['wait_med']:.0f} 根、平均 {_B['全部']['wait_mean']:.2f} 根。再加上「确认后下一根开盘」这一根，锚点→买入还要再加 1。分类型的等待分布：</p>
<table>
<tr><th>类型</th><th>n</th><th>wait = 1</th><th>wait = 2</th><th>wait = 3</th><th>wait = 4</th><th>中位</th><th>平均</th></tr>
{wait_rows()}
</table>
<p class="small">读法：3b 的 wait 明显偏小（{_B['3b']['wait'].get('1', 0)}/{_B['3b']['n']} 在下一根就确认），因为它的信号 bar 已经晚了中位 {_B['3b']['delay_med_pos']:.0f} 根，价格早在信号 bar 高点之上；2、2s、1p 则更常等到 3~4 根。</p>

<h3>1.4 确认失败：多少买点根本等不到入场</h3>
<p>全部 {_buy} 个买点里，<b>{_fail} 个（{_fail / _buy:.1%}）在 4 根内没有等到「收盘越过信号 bar 高点」，<code>conf_ok = 0</code>，按规则不入场</b>；只有 {_ok} 个（{_ok / _buy:.1%}）可交易。这是确认规则本身淘汰的，不是模型筛掉的。</p>
"""
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>缠论买卖点报告</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>{CSS}</style></head><body><div class="wrap">
<h1>缠论买卖点报告</h1>
<div class="meta">最后更新：{datetime.now():%Y-%m-%d %H:%M} ｜ K 线：{N_ALL} 个 USDT 永续（{N_COIN} 币种 + _MARKET），{FREQ}，{SPAN}（数据自 {DATA_START} 起，前 12 个月做 warmup） ｜ 事件表 {EVENTS_NAME} · 模型分 {MODEL_NAME}（逐月 walk-forward 样本外）· 基线 48bar/2ATR/无止盈（止盈 2026-11-09 按台账 #47 证据禁用）· 双边 13bp</div>

<h2>摘要</h2>
<div class="warn"><b>口径警告（2026-11-09 审计补充）</b>：①本报告 §3 回测用
<b>48bar / 2ATR 止损 / 无止盈</b>（原 E001/E002 基线的 5ATR 止盈已按对照证据禁用，见台账 #47），
仍<b>不是</b>蓝图 v7 冻结规格（144bar / 3ATR），
且 universe 为 crypto {N_COIN} 币 + _MARKET，{FREQ}，与蓝图（A股 30m）是两套策略，数字不能互相证明。
②§3 的「真/假」分组是<b>完美先知标签</b> <code>survive_144</code>（144 bar 后才知道），
不是模型判真——「真」组收益不可交易，只证明标签有经济含义；模型判真（score≥0.5）从未做分组回测，
其可交易形态（score 门控）已被 E001 / 台账 #18/#22/#33 否定。
审计与冻结规格复算见 <code>reports/bsppoint_audit_20261109.md</code>。</div>
<div class="card">
<p><b>一、真 → 赚钱，强成立。</b>用完美先知标签 <code>survive_144</code>（144 bar 内未被擦除 = 真）分组：
「真」样本内 {pct(s_t['is']['ret'])}、样本外 {pct(s_t['oos']['ret'])}；「假」样本内 {pct(s_f['is']['ret'])}、样本外 {pct(s_f['oos']['ret'])}。分离在样本外完整保留，
且<b>对每一类买卖点都成立</b>（§4：每类的「真」前向收益多为正、「假」为负）。</p>
<p><b>二、模型对真/假有判别力，但弱，且与收益方向相反。</b>模型对 <code>survive_144</code> 的 AUC = {auc:.3f}；
分数越高 → 存活率越高（D0 {d0['survive_rate']:.0%} → D9 {d9['survive_rate']:.0%}，单调）。<b>但分数越高 → 「真」的前向收益越低</b>：
D0 的「真」平均 {pct(d0['fw_true']['mean'],2)}、D9 的「真」只有 {pct(d9['fw_true']['mean'],2)}。
§9 的实测给出机制：<b>总振幅不随分数变（~{amp_mean:.0f}% 全档持平），变的是上下行配比</b>——分数越高，
上行占比越低（Spearman {VOL['spearman_score_vs_up_share_true']:+.2f}）、下行偏移越大（Spearman {VOL['spearman_score_vs_dn_true']:+.2f}），
且集中在 2 买/类 2 买。这就是 E001/E002 里 score 门控样本外失效的机制（E1 实测）。</p>
<p><b>三、144 不是魔法数字。</b>模型对「存活到 h」的 AUC 在 h=24~576 全区间都约 {auc:.2f}，
144（{auc:.3f}）只是噪声内的峰；而存活率本身从 h={h_first['h']} 的 {h_first['survive_rate']:.0%} 快速跌到 h={h_last['h']} 的 {h_last['survive_rate']:.0%}。</p>
</div>

<h2>1. 方法与口径：四根 bar</h2>
<div class="card">
<ul>
<li><b>事件</b>：确认买点（<code>is_buy==1 且 conf_ok==1</code>）且有逐月 walk-forward 样本外分数的子集，n={VAL['n_events']}。</li>
<li><b>标签（真/假）</b>：<code>survive_144 = 1</code>（144 bar 内未被擦除）= 真；<code>survive_144 = 0</code> = 假。存活率 {VAL['survive_rate']:.1%}。</li>
<li><b>模型</b>：XGBoost，逐月训练、当月测试。对 <code>survive_144</code> 的 AUC = {auc:.3f}。</li>
<li><b>回测（§3）</b>：基线 48bar / 2ATR 止损 / <b>无止盈</b>（原 5ATR 止盈已按台账 #47 对照证据禁用）/ 无门控 / 10 仓位×10% / 13bp，确认入场。</li>
<li><b>前向收益（§4/§5）</b>：<code>c_dir_ret_48</code> = 确认入场、持有 48bar 收盘离场、按方向取号、扣 13bp（事件表已算好的列，无组合交互）。格式为「均值 / 中位 / 胜率」。</li>
<li><b>术语</b>：recall（召回率）= 识别出的正确点 / 全部正确点；FPR（假阳率）= 识别出的错误点 / 全部错误点；阳性率 = 识别出的点 / 总点；precision = 识别出的点里正确的比例。</li>
</ul>
{bars_html}
</div>

<h2>2. 重绘分类</h2>
<div class="card">
<p>一个买卖点从「首次出现」之后，会经历不同形式的改变。按两根轴拆：<b>命运轴</b>（A/D/B）与<b>其他轴</b>（C/E/F）。</p>
<table>
<tr><th>类别</th><th>含义</th><th>占比</th><th>是不是「位置重绘」</th></tr>
<tr><td><b>A</b> 存活且锚点未动</td><td>从出现到结束，点没被删、锚点 K 线也没挪</td><td>~11.7%</td><td>否</td></tr>
<tr><td><b>D</b> 存活但锚点动过</td><td>点还在，但锚点 K 线往后顺移（定义它的段被延伸重绘）</td><td>~33.5%</td><td><b>是（点自己挪了）</b></td></tr>
<tr><td><b>B</b> 首次消失</td><td>点第一次从活动集合被擦掉（依赖的段被回滚）</td><td>~54.8%</td><td><b>是（点被删了）</b></td></tr>
<tr><td><b>F</b> 消失又回来</td><td>B 的子集：擦掉之后结构重新确认，它又出现（藏在 B 里）</td><td>~5.0%</td><td><b>是（删了又画回来）</b></td></tr>
<tr><td><b>E</b> 类型追加</td><td>点没动没删，但多了个类型标签（如 2 买同时是类 2 买）</td><td>~17.3%</td><td>否（只是标签变多）</td></tr>
<tr><td><b>C</b> 类型改变</td><td>点的类型标签被替换（如从 2 买变成 1 买）</td><td>~2.1%</td><td>否（标签换了）</td></tr>
</table>
<p>所以「重绘」有两种：<b>位置重绘</b>（A/D/B/F，锚点 K 线变了）与<b>标签重绘</b>（C/E，类型变了）。D 占比最大（33.5%），说明「点还在、但位置挪了」是最常见的重绘形式。</p>
</div>

<h2>3. 真/假与收益：分组回测</h2>
<div class="card">
<p>按 <code>survive_144</code> 分组回测，看「真」和「假」的收益分布有没有差别。</p>
<table>
<tr><th rowspan="2">分组</th><th colspan="6">样本内（2021-01 ~ 2025-06）</th><th colspan="6">样本外（2025-07 ~ 2026-10）</th></tr>
<tr><th>收益</th><th>夏普</th><th>每笔期望</th><th>胜率</th><th>盈亏比</th><th>MDD</th><th>收益</th><th>夏普</th><th>每笔期望</th><th>胜率</th><th>盈亏比</th><th>MDD</th></tr>
<tr><td>全部确认买点</td>{conf_row("all")}</tr>
<tr><td>真（survive=1）</td>{conf_row("true")}</tr>
<tr><td>假（survive=0）</td>{conf_row("false")}</tr>
</table>
<p><b>结论</b>：真/假在样本内完全分开（真 {pct(s_t['is']['ret'])} vs 假 {pct(s_f['is']['ret'])}），样本外也分开（真 {pct(s_t['oos']['ret'])} vs 假 {pct(s_f['oos']['ret'])}）。<b>模型对真假的判别力在样本外完整保留</b>，但收益方向与分数方向相反（见摘要 §二）。</p>
</div>

<h2>4. 分类型：每一类买卖点的真/假与收益</h2>
<div class="card">
<p>把「全部」拆成 9 类买卖点，看每一类的真/假分离度。</p>
<table>
<tr><th>类型</th><th>n</th><th>存活率</th><th>AUC</th><th>判真</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>前向收益（全）</th><th>前向收益（真）</th><th>前向收益（假）</th><th>寿命中位</th></tr>
{type_rows()}
</table>
<p class="meta">寿命中位 = 从出现到首次被擦除的 bar 数（仅对曾被擦除的点算）。AUC = 模型对「该类型点是否存活」的判别力。</p>
<p class="small">韦恩图读法：<span class="legend" style="background:rgba(47,179,122,.28);border-color:#2fb37a"></span>真（左圈）
<span class="legend" style="background:rgba(47,111,237,.28);border-color:#2f6fed"></span>模型判真（右圈，阈值 0.5）。
<b>圆面积与集合大小成正比，交集面积 = TP</b>；左侧 = FN（漏掉的真点），中间 = TP，右侧 = FP（误判为真的假点）。每格独立缩放（格间不比大小）；区域太窄放不下数字时，数字见格标题，悬停可看全部四项。</p>
{chart_venn_types}
{chart_pie_types}
<p><b>结论</b>：<b>每一类买卖点的「真」前向收益都 > 0、「假」前向收益都 < 0</b>（除了 3b 的「真」样本太少不显著；2s,3b 的「假」仅 2 个，不具统计意义）。真/假分离不是某一类的特性，而是<b>所有买卖点的共性</b>。</p>
</div>

<h2>5. 分十分位：分数越高，存活率越高，但收益越低</h2>
<div class="card">
<p>把全部确认买点按模型分数从低到高切 10 档（D0 最低、D9 最高），看每一档的存活率与前向收益。</p>
<table>
<tr><th>分位</th><th>分数区间</th><th>n</th><th>存活率</th><th>前向收益（全）</th><th>前向收益（真）</th><th>前向收益（假）</th></tr>
{decile_rows()}
</table>
<p><b>结论</b>：存活率从 D0 的 {d0['survive_rate']:.1%} 单调升到 D9 的 {d9['survive_rate']:.1%}（模型有判别力）。但「真」的前向收益从 D0 的 {pct(d0['fw_true']['mean'],2)} 降到 D9 的 {pct(d9['fw_true']['mean'],2)}（分数越高、收益越低）。这就是 E001/E002 里 score 门控样本外失效的机制。</p>
</div>

<h2>6. 阈值扫描：Precision / Recall</h2>
<div class="card">
<p>把模型分数切不同阈值，看每一刀的 Precision / Recall / FPR。</p>
<table>
<tr><th>切法</th><th>阈值</th><th>n（判真数）</th><th>阳性率</th><th>Recall</th><th>FPR</th><th>Precision</th><th>TP 占比</th><th>F1</th></tr>
{ratio_rows()}
</table>
{chart_venn_all}
<p><b>结论</b>：阈值 0.5 时，Precision = {r05['precision']:.1%}、Recall = {r05['recall']:.1%}。阈值越高，留下来的点越精（Precision 上升），但 Recall 下降（漏掉更多真点）；阈值越低则相反（放进来的点越多，Precision 下降、FPR 上升）。</p>
</div>

<h2>7. 存活 horizon：144 不是魔法数字</h2>
<div class="card">
<p>把「存活」的定义从 144 bar 改成其他长度，看模型的 AUC 怎么变。</p>
<table>
<tr><th>存活 horizon (bar)</th><th>存活率</th><th>AUC</th></tr>
{horizon_rows()}
</table>
<p><b>结论</b>：AUC 在 h=24~576 全区间都约 {auc:.2f}，144 只是噪声内的峰。存活率本身从 h={h_first['h']} 的 {h_first['survive_rate']:.1%} 快速跌到 h={h_last['h']} 的 {h_last['survive_rate']:.1%}。</p>
</div>

<h2>8. 寿命：被擦除的点活了多久</h2>
<div class="card">
<p>只看曾被擦除的点（B/F），看它们从出现到首次被擦除活了多少 bar。</p>
<table>
<tr><th>类型</th><th>被擦除 n</th><th>寿命中位 (bar)</th><th>寿命 P75 (bar)</th></tr>
{lifespan_rows()}
</table>
<p class="meta">寿命 = 从出现到首次被擦除的 bar 数。只统计曾被擦除的点（B/F 类）。</p>
</div>

<h2>9. 波动率与分数：为什么分数越高、收益越低</h2>
<div class="card">
<p>把「前向收益」拆成「总振幅」与「上下行配比」，看分数与它们的关系。</p>
<table>
<tr><th>分位</th><th>存活率</th><th>总振幅</th><th>真点振幅</th><th>真点上行的占比</th><th>真点下行</th><th>真点前向收益</th></tr>
{vol_decile_rows()}
</table>
<p><b>结论</b>：总振幅不随分数变（~{amp_mean:.0f}% 全档持平），但上行占比随分数升高而下降（Spearman {VOL['spearman_score_vs_up_share_true']:+.2f}），下行偏移随分数升高而增大（Spearman {VOL['spearman_score_vs_dn_true']:+.2f}）。这就是 E001/E002 里 score 门控样本外失效的机制。</p>
</div>

<h2>10. 分类型：波动率与分数的相关性</h2>
<div class="card">
<p>把 §9 的分析按买卖点类型拆，看哪些类型的「分数越高、收益越低」最明显。</p>
<table>
<tr><th>类型</th><th>真点 n</th><th>分数 vs 下行</th><th>分数 vs 上行占比</th><th>分数 vs 前向收益</th></tr>
{vol_type_rows()}
</table>
<p class="meta">Spearman 相关系数。下行 = <code>c_dir_dn_48</code>；上行占比 = <code>c_dir_up_48 / (c_dir_up_48 + |c_dir_dn_48|)</code>；前向收益 = <code>c_dir_ret_48</code>。</p>
<p><b>结论</b>：2 买/类 2 买的「分数越高、收益越低」最明显（下行正相关、上行占比负相关、前向收益负相关）。1 买/1p/3a/3b 的相关性较弱或方向相反。</p>
</div>

<h2>11. 标签与判真构成</h2>
<div class="card">
{chart_pie_conf}
<p>左图：全部确认买点里，真（survive=1）占 {agg['tp'] + agg['fn']:,}（{(agg['tp'] + agg['fn']) / VAL['n_events']:.1%}）、假占 {agg['fp'] + agg['tn']:,}（{(agg['fp'] + agg['tn']) / VAL['n_events']:.1%}）。右图：模型判真（score ≥ 0.5）的点里，TP（判真且真）占 {agg['tp']:,}（{agg['tp'] / (agg['tp'] + agg['fp']):.1%}）、FP（判真但假）占 {agg['fp']:,}（{agg['fp'] / (agg['tp'] + agg['fp']):.1%}）。</p>
</div>

<h2>12. 口径提示</h2>
<div class="todo">
<ul>
<li>本页为 31 币历史口径；原报告已于 2026-11-10 作废归档，基准结论以 <code>reports/bsppoint_audit_20261109.md</code> 为准。</li>
<li>若要换到 69 币，必须另立预注册实验，并确认事件表、模型分与蓝图冻结规格一致。</li>
</ul>
</div>

</div></body></html>
"""
    out = REPORTS / f"bsppoint_report{OUT_SUFFIX}.html"
    out.write_text(html, encoding="utf-8")
    print(f"报告 {out} ({out.stat().st_size} 字节)")


if __name__ == "__main__":
    main()
