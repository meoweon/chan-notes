"""生成权益曲线可视化对比报告（自包含 HTML + 内嵌 SVG，无外部依赖）。

两个面板：
A) OOF 全程 2021-02~2026-10：基线 / 存活模型过滤 / frenzy 静态门控 / 滚动门控
B) 真 OOS 2024-01~2026-10：基线 / frenzy 冻结门控（阈值 2023 年底冻结）

输出 reports/equity_compare.html
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "data" / "bt"

VARIANTS_A = [
    ("v7_base_oof", "基线 · 确认买点全做", "#64748b", "",
     "全部确认买点机械入场，144bar/3ATR、无信号离场，10 仓位×10%。研究的对照锚。"),
    ("v7_surv_q20_oof", "ML 存活过滤 · 前20%", "#f59e0b", "",
     "基线 + ML 存活模型分数前 20% 才入场。事件级单调增，但组合级削减狂潮期暴露——反面教材。"),
    ("frenzy_static_gate", "frenzy 静态门控（含泄漏，仅参考）", "#a855f7", "dash",
     "基线 + 狂热指数≥全样本中位数才开机。阈值用了未来数据，只作上限参考，不是可用方案。"),
    ("frenzy_rolling_gate", "frenzy 滚动门控（失败对照）", "#ef4444", "dash",
     "基线 + 滚动 252 日中位数门控（严格 PIT）。度量的是『相对近期热』而非绝对热，反而最差——重要的反面证据。"),
]
VARIANTS_B = [
    ("frozen_always_on", "基线 · 确认买点全做", "#64748b", "",
     "全部确认买点机械入场，144bar/3ATR、无信号离场，10 仓位×10%。2024-01 起真样本外。"),
    ("frozen_frozen_gate", "frenzy 冻结门控 ★ 推荐方案", "#16a34a", "",
     "基线 + 狂热指数（前一交易日值）≥ 冻结阈值 0.00717 才开机，阈值 2023 年底用 2020-2023 数据冻结，OOS 期零泄漏。开机率仅 40%。"),
]

SHADE = [("2024-09-24", "2024-10-08", "924 狂潮")]
COLORS_UP = "#16a34a"


def load(tag):
    e = pd.read_parquet(BT / f"{tag}_equity.parquet")
    s = e["equity"]
    s.index = pd.to_datetime(s.index)
    return s


def metrics(s, n_trades=None):
    norm = s / s.iloc[0]
    total = norm.iloc[-1] - 1
    days = (s.index[-1] - s.index[0]).days
    cagr = (norm.iloc[-1]) ** (365.25 / days) - 1 if days > 0 else np.nan
    d = s.resample("1D").last().dropna().pct_change().dropna()
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    dd = (s / s.cummax() - 1).min()
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=dd, final=s.iloc[-1], n=n_trades)


def downsample(s, max_pts=900):
    if len(s) <= max_pts:
        return s
    idx = np.unique(np.linspace(0, len(s) - 1, max_pts).astype(int))
    return s.iloc[idx]


def svg_panel(series_list, title, note, W=1160, H=320, DH=110):
    """series_list: [(label, color, dash, series_normalized, dd_series)]"""
    ml, mr, mt, mb = 70, 20, 34, 26
    pw, ph = W - ml - mr, H - mt - mb
    dh_top = H + 8
    all_t0 = min(s.index[0] for _, _, _, s, _ in series_list)
    all_t1 = max(s.index[-1] for _, _, _, s, _ in series_list)
    t0, t1 = pd.Timestamp(all_t0), pd.Timestamp(all_t1)
    span = (t1 - t0).total_seconds()
    ymax = max(float(s.max()) for _, _, _, s, _ in series_list) * 1.02
    ymin = min(float(s.min()) for _, _, _, s, _ in series_list) * 0.98
    ddmin = min(float(dd.min()) for _, _, _, _, dd in series_list) * 1.1

    def X(ts):
        return ml + (ts - t0).total_seconds() / span * pw

    def Y(v):
        return mt + (ymax - v) / (ymax - ymin) * ph

    def YD(v):
        return dh_top + (0 - v) / (0 - ddmin) * DH

    out = [f'<svg width="{W}" height="{dh_top + DH + mb + 40}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="Microsoft YaHei, PingFang SC, sans-serif">']
    out.append(f'<text x="{ml}" y="20" font-size="15" font-weight="700" fill="#0f172a">{title}</text>')
    out.append(f'<text x="{ml}" y="{dh_top + DH + mb + 30}" font-size="11" fill="#94a3b8">{note}</text>')

    # 狂潮阴影
    for a, b, lab in SHADE:
        xa, xb = X(pd.Timestamp(a)), X(pd.Timestamp(b))
        if xb > ml and xa < ml + pw:
            xa, xb = max(xa, ml), min(xb, ml + pw)
            out.append(f'<rect x="{xa:.1f}" y="{mt}" width="{xb - xa:.1f}" height="{ph + 8 + DH}" fill="#fef08a" opacity="0.45"/>')
            out.append(f'<text x="{xa + 2:.1f}" y="{mt + 12}" font-size="10" fill="#a16207">{lab}</text>')

    # y 网格（收益）
    import math
    step = 0.2 if ymax < 3 else 0.5
    v = math.floor(ymin / step) * step
    while v <= ymax:
        y = Y(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" font-size="11" fill="#94a3b8" text-anchor="end">{v:.1f}x</text>')
        v += step
    # 回撤网格
    for frac in (0.25, 0.5, 0.75, 1.0):
        v = ddmin * frac
        y = YD(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" font-size="10" fill="#94a3b8" text-anchor="end">{v * 100:.0f}%</text>')
    out.append(f'<text x="{ml - 6}" y="{dh_top - 2}" font-size="10" fill="#94a3b8" text-anchor="end">回撤</text>')

    # x 轴（年度）
    yr = t0.year
    while pd.Timestamp(f"{yr}-01-01") <= t1:
        tt = pd.Timestamp(f"{yr}-01-01")
        if tt >= t0:
            x = X(tt)
            out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{dh_top + DH}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>')
            out.append(f'<text x="{x:.1f}" y="{dh_top + DH + 16}" font-size="11" fill="#94a3b8" text-anchor="middle">{yr}</text>')
        yr += 1

    # 曲线
    for label, color, dash, s, dd in series_list:
        sd = downsample(s)
        pts = " ".join(f"{X(t):.1f},{Y(float(v)):.1f}" for t, v in sd.items())
        dashattr = ' stroke-dasharray="6,4"' if dash else ""
        width = 2.6 if "★" in label else 1.8
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dashattr}/>')
        ddd = downsample(dd)
        pts2 = " ".join(f"{X(t):.1f},{YD(float(v)):.1f}" for t, v in ddd.items())
        out.append(f'<polyline points="{pts2}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.75"{dashattr}/>')

    # 图例
    lx = ml + 10
    for label, color, dash, s, dd in series_list:
        dashattr = ' stroke-dasharray="6,4"' if dash else ""
        out.append(f'<line x1="{lx}" y1="{mt + 10}" x2="{lx + 26}" y2="{mt + 10}" stroke="{color}" stroke-width="2.4"{dashattr}/>')
        out.append(f'<text x="{lx + 32}" y="{mt + 14}" font-size="12" fill="#334155">{label}</text>')
        lx += 32 + len(label) * 13 + 30
    out.append('</svg>')
    return "".join(out)


def table_html(rows):
    th = "".join(f"<th>{c}</th>" for c in ["方案", "交易笔数", "总收益", "CAGR", "夏普", "最大回撤", "期末权益(万)"])
    trs = []
    for r in rows:
        tds = "".join(f"<td>{v}</td>" for v in r)
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def build_panel(variants, title, note):
    series_list, rows, cards = [], [], []
    for tag, label, color, dash, desc in variants:
        s = load(tag)
        tr = pd.read_parquet(BT / f"{tag}_trades.parquet")
        norm = s / s.iloc[0]
        dd = s / s.cummax() - 1
        m = metrics(s)
        series_list.append((label, color, dash, norm, dd))
        star = " style=\"background:#f0fdf4\"" if "★" in label else ""
        rows.append([f"<b{star}>{label}</b>", len(tr), f"{m['total'] * 100:+.1f}%",
                     f"{m['cagr'] * 100:.1f}%", f"{m['sharpe']:.2f}", f"{m['mdd'] * 100:.1f}%",
                     f"{m['final'] / 1e4:.0f}"])
        cards.append(
            f'<div class="card" style="border-left:4px solid {color}">'
            f'<div class="card-title">{label}</div><div class="card-desc">{desc}</div></div>')
    return svg_panel(series_list, title, note), table_html(rows), "".join(cards)


svg_a, tab_a, cards_a = build_panel(
    VARIANTS_A, "A 组 · OOF 全程对比（2021-02 ~ 2026-10，10 仓位 × 10%，144bar/3ATR，无信号离场）",
    "黄色阴影 = 2024-09-24~10-08 狂潮窗口；虚线 = 不可实盘的对照方案（静态门控阈值含未来信息 / 滚动门控为失败对照）")
svg_b, tab_b, cards_b = build_panel(
    VARIANTS_B, "B 组 · 真样本外对比（2024-01 ~ 2026-10，门控阈值 2023 年底冻结，OOS 期零泄漏）",
    "黄色阴影 = 2024-09-24~10-08 狂潮窗口（门控在该窗口全程开机，10 笔交易均值 +1984bp）")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>chan-notes 策略方案权益曲线对比</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #0f172a; background: #fff; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.sub {{ color: #64748b; font-size: 13px; margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0 20px; }}
.card {{ background: #f8fafc; border-radius: 6px; padding: 10px 12px; }}
.card-title {{ font-weight: 700; font-size: 13px; margin-bottom: 4px; }}
.card-desc {{ font-size: 12px; color: #475569; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 30px; font-size: 13px; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: right; }}
th {{ background: #f1f5f9; }}
td:first-child, th:first-child {{ text-align: left; }}
.note {{ font-size: 12px; color: #64748b; line-height: 1.8; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:10px 14px; margin: 18px 0; }}
</style></head><body>
<h1>chan-notes · 策略方案权益曲线对比</h1>
<div class="sub">数据：HS300 289 股 30m · 回测口径：确认买点入场 / 144bar / 3ATR 止损 / 无信号离场 / 10 仓位×10% / 双边 13bp 成本 · 表内夏普按日频权益重采样计算（与 README 引用的 portfolio 口径 0.80/1.21 略有差异，结论一致） · 生成时间 2026-11-01</div>

{svg_a}
<div class="cards">{cards_a}</div>
{tab_a}

{svg_b}
<div class="cards">{cards_b}</div>
{tab_b}

<div class="note"><b>读图要点</b><br>
1. A 组里唯一可用的实盘候选是<b>灰色基线</b>；紫色静态门控只证明「狂热期开机」方向正确（阈值含未来信息），红色滚动门控证明「相对热」不等于「绝对热」。<br>
2. B 组是严格口径：阈值在 2023 年底冻结，2024 年后的数据没有参与任何参数选择。<b>绿色冻结门控 ★</b> 用 40% 的在线时间取得更高收益、近半回撤。<br>
3. 两组结论一致：这个策略的正确用法不是预测个股（5 次证伪），而是 regime 开关——<b>缠论决定买什么，狂热指数决定什么时候买</b>。<br>
4. 诚实边界：夏普 1.21 对应 2.8 年样本（年化 t≈2.0，刚过显著线）；本质仍是尾部游戏（去 top5% 赢家 E=-16bp）；60% 时间空仓的现金管理是实盘前必须回答的问题。</div>
</body></html>"""

out = ROOT / "reports" / "equity_compare.html"
out.write_text(html, encoding="utf-8")
print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
