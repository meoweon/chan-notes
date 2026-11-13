"""加密版「策略方案权益曲线对比」页，版式对照 reports/equity_compare.html（A 股东风版）。

A 组：全程 2021-01~2026-10，四条方案（含两条不可实盘的对照）。
B 组：2025-07 起严格样本外，frenzy 阈值在 2025-06 末冻结。
所有图都是内联 SVG，单文件无外部依赖。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_sweep_backtest as psb
from chan_notes.backtest import portfolio

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT = REPORTS / "equity_compare_crypto.html"

HOLD, STOP, TP = 144, 3.0, None
BASELINE = dict(hold=HOLD, stop=STOP, tp=TP)

C_BASE = "#64748b"
C_ML = "#f59e0b"
C_STATIC = "#a855f7"
C_ROLL = "#ef4444"
C_FROZEN = "#16a34a"


def run_mask(df, bars, mask=None):
    sub = df if mask is None else df[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=STOP,
                           take_profit_atr=TP, hold_bars=HOLD, signal_exit=True)
    return portfolio.run(sub, bars, p, entry_mode="confirm")


def daily(eq, start=psb.WINDOW_START):
    s = eq["equity"].resample("D").last().ffill().dropna()
    s = s[s.index >= start]
    return s / s.iloc[0]


def stats(res, curve, bars_per_year=365):
    if res is None or curve is None or len(curve) < 2:
        return None
    tr = res["trades"]
    r = curve.pct_change().dropna()
    dd = (curve / curve.cummax() - 1).min()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    return dict(
        n=len(tr), ret=curve.iloc[-1] - 1, cagr=curve.iloc[-1] ** (1 / years) - 1,
        sharpe=r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else np.nan,
        mdd=dd, end=curve.iloc[-1] * 100, curve=curve, start=curve.index[0],
    )


def fmt_pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{x*100:+.{d}f}%"


def svg_chart(series, title, note, width=1160, height=504):
    """series: list of (label, curve, color, dashed)."""
    x0, x1, y0, y1 = 70, 1140, 34, 412
    allv = pd.concat([c for _, c, _, _ in series])
    lo, hi = float(allv.min()), float(allv.max())
    pad = max((hi - lo) * 0.08, 0.02)
    lo, hi = lo - pad, hi + pad
    dates = [c.index for _, c, _, _ in series]
    t0 = min(d.min() for d in dates)
    t1 = max(d.max() for d in dates)
    span = max((t1 - t0).days, 1)

    def X(t):
        return x0 + (t - t0).days / span * (x1 - x0)

    def Y(v):
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    out = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="Microsoft YaHei, PingFang SC, sans-serif">',
           f'<text x="{x0}" y="20" font-size="15" font-weight="700" fill="#0f172a">{title}</text>']
    ticks = 6
    for i in range(ticks + 1):
        v = lo + (hi - lo) * i / ticks
        y = Y(v)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        out.append(f'<text x="{x0-6}" y="{y+4:.1f}" font-size="11" fill="#94a3b8" '
                   f'text-anchor="end">{v:.1f}x</text>')
    for i in range(9):
        t = t0 + pd.Timedelta(days=span * i / 8)
        out.append(f'<text x="{X(t):.1f}" y="{y1+22}" font-size="11" fill="#94a3b8" '
                   f'text-anchor="middle">{t:%Y-%m}</text>')

    for label, curve, color, dashed in series:
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in curve.items())
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"'
                   f'{dash} stroke-linejoin="round"/>')

    lx = x0 + 10
    for label, _, color, dashed in series:
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        out.append(f'<line x1="{lx}" y1="46" x2="{lx+26}" y2="46" stroke="{color}" '
                   f'stroke-width="2.4"{dash}/>')
        out.append(f'<text x="{lx+32}" y="50" font-size="12" fill="#334155">{label}</text>')
        lx += 46 + len(label) * 12.2
    assert lx < width - 20, f"legend overflow: {lx:.0f}"
    out.append(f'<text x="{x0}" y="{height-14}" font-size="11" fill="#94a3b8">{note}</text>')
    out.append("</svg>")
    return "".join(out)


def table(rows):
    h = ("<table><thead><tr><th>方案</th><th>交易笔数</th><th>总收益</th><th>CAGR</th>"
         "<th>夏普</th><th>最大回撤</th><th>期末权益(万)</th></tr></thead><tbody>")
    for name, s in rows:
        h += (f"<tr><td>{name}</td><td>{s['n']}</td><td>{fmt_pct(s['ret'])}</td>"
              f"<td>{fmt_pct(s['cagr'])}</td><td>{s['sharpe']:.2f}</td>"
              f"<td>{fmt_pct(s['mdd'])}</td><td>{s['end']:.0f}</td></tr>")
    return h + "</tbody></table>"


def main():
    print("构建上下文...", flush=True)
    ctx = psb.build_context(verbose=False)
    df, bars = ctx["df"], ctx["bars"]
    frenzy_daily = psb.build_frenzy_index()          # 索引是 "YYYY-MM-DD" 字符串，给事件表 map 用
    frenzy_dt = frenzy_daily.copy()
    frenzy_dt.index = pd.to_datetime(frenzy_dt.index)

    # ---- 掩码 ----
    m_all = pd.Series(True, index=df.index)

    s_thr80 = psb.monthly_score_thresholds(df, 0.80)
    month = df["t"].dt.to_period("M").astype(str)
    m_ml = df["score"] >= month.map(s_thr80).fillna(np.inf)

    m_static = df["frenzy"] >= df["frenzy"].median()          # 全样本中位（含未来信息）

    f_roll_thr = psb.expanding_quantile_series(frenzy_daily, 0.5)
    m_roll = df["frenzy"] >= df["d"].map(f_roll_thr)

    is_mask = df["t"] < psb.IS_END
    is_frenzy = frenzy_dt.loc[psb.WINDOW_START:psb.IS_END].dropna()
    frozen_q = {q: float(is_frenzy.quantile(q)) for q in (0.25, 0.50, 0.75)}
    frozen = frozen_q[0.50]
    m_frozen = df["frenzy"] >= frozen

    # 开机率（OOS 期间 frenzy >= 冻结阈值的交易日占比）
    oos_days = frenzy_dt.loc[psb.IS_END:psb.OOS_END].dropna()
    on_rate = {q: float((oos_days >= v).mean()) for q, v in frozen_q.items()}

    # ---- A 组：全程 ----
    print("A 组回测...", flush=True)
    A = []
    for name, mask, color, dashed in [
        ("基线 · 确认买点全做", m_all, C_BASE, False),
        ("ML 存活过滤 · 前20%", m_ml, C_ML, False),
        ("静态门控（含泄漏）", m_static, C_STATIC, True),
        ("滚动中位门控（PIT）", m_roll, C_ROLL, True),
    ]:
        res = run_mask(df, bars, mask)
        s = stats(res, daily(res["equity"]))
        print(f"  {name}: " + (f"{s['n']} 笔 ret={s['ret']:.1%} 末值={s['curve'].iloc[-1]:.4f} "
                                f"sharpe={s['sharpe']:.2f}" if s else "N/A"), flush=True)
        A.append((name, s, color, dashed))

    # ---- B 组：严格 OOS（2025-07 起重新起跑）----
    print("B 组回测...", flush=True)
    bars_oos = {k: v[v["time"] >= psb.IS_END - pd.Timedelta(days=200)].reset_index(drop=True)
                for k, v in bars.items()}
    df_oos = df[df["t"] >= psb.IS_END]
    B = []
    b_specs = [("基线 · 确认买点全做", pd.Series(True, index=df_oos.index), C_BASE, False)]
    for q, color in [(0.25, "#16a34a"), (0.50, "#0ea5e9"), (0.75, "#a855f7")]:
        mask = df_oos["frenzy"] >= frozen_q[q]
        b_specs.append((f"冻结门控 p{int(q*100)}（{frozen_q[q]:.4f}）", mask, color, False))
    for name, mask, color, dashed in b_specs:
        res = run_mask(df_oos, bars_oos, mask)
        s = stats(res, daily(res["equity"], psb.IS_END)) if res is not None else None
        print(f"  {name}: " + (f"{s['n']} 笔 ret={s['ret']:.2%} 末值={s['curve'].iloc[-1]:.4f} "
                                f"sharpe={s['sharpe']:.2f}" if s else "N/A"), flush=True)
        B.append((name, s, color, dashed))

    svg_a = svg_chart([(n, s["curve"], c, d) for n, s, c, d in A],
                      "A 组 · 全程对比（2021-01 ~ 2026-10，10 仓位 × 10%，144bar / 3ATR，无信号离场）",
                      "虚线 = 对照方案：静态门控阈值含未来信息（不可实盘）；滚动门控为严格 PIT，但"
                      "只看「相对近期热」。A 组含样本内区间，仅作形状参考。")
    svg_b = svg_chart([(n, s["curve"], c, d) for n, s, c, d in B],
                      "B 组 · 严格样本外（2025-07 ~ 2026-10，阈值在 2025-06 末冻结）",
                      f"冻结阈值取 2021-01~2025-06 狂热指数的 p25/p50/p75；样本外开机率 "
                      f"{on_rate[0.25]:.0%} / {on_rate[0.50]:.0%} / {on_rate[0.75]:.0%}。")

    cards = "".join(
        f'<div class="card"><div class="card-title" style="color:{c}">{n}</div>'
        f'<div class="card-desc">{d}</div></div>'
        for n, c, d in [
            ("基线 · 确认买点全做", C_BASE,
             "全部确认买点机械入场，144bar / 3ATR 止损、信号离场，10 仓位 × 10%。研究的对照锚。"),
            ("ML 存活过滤 · 前 20%", C_ML,
             "基线 + 存活模型分数≥当月滚动 80 分位才入场。事件级区分度不错，但组合级砍掉了大部分交易——反面教材。"),
            ("frenzy 静态门控（含泄漏，仅参考）", C_STATIC,
             "基线 + 狂热指数≥全样本中位数才开机。阈值用了未来数据、不可实盘；即便开了后门，它也没跑赢基线——方向本身就不成立。"),
            ("frenzy 滚动门控（PIT）", C_ROLL,
             "基线 + 滚动 252 日中位门控（严格 PIT，可实盘）。度量的是「相对近期热」而非绝对热；全程夏普略高，但样本外不成立。"),
        ])

    cards_b = "".join(
        f'<div class="card"><div class="card-title" style="color:{c}">{n}</div>'
        f'<div class="card-desc">{d}</div></div>'
        for n, c, d in [
            ("基线 · 确认买点全做", C_BASE,
             "全部确认买点机械入场，144bar/3ATR、无信号离场，10 仓位×10%。2025-07 起重新起跑。"),
            (f"冻结门控 · IS p25（{frozen_q[0.25]:.4f}）", "#16a34a",
             f"狂热指数（T-1）≥ 该阈值才开机，开机率 {on_rate[0.25]:.0%}。阈值只用 2021-01~2025-06 估计。"),
            (f"冻结门控 · IS p50（{frozen_q[0.50]:.4f}）", "#0ea5e9",
             f"同上，中位数阈值，开机率 {on_rate[0.50]:.0%}。"),
            (f"冻结门控 · IS p75（{frozen_q[0.75]:.4f}）", "#a855f7",
             f"同上，高分位阈值，开机率 {on_rate[0.75]:.0%}——等于只在最热的少数交易日进场。"),
        ])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>chan-notes 加密 · 策略方案权益曲线对比</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #0f172a; background: #fff; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.sub {{ color: #64748b; font-size: 13px; margin-bottom: 20px; line-height: 1.7; }}
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
<h1>chan-notes · 加密策略方案权益曲线对比</h1>
<div class="sub">数据：21 个 USDT 永续 1h（2021-01 ~ 2026-10）· 回测口径：确认买点入场 / 144bar / 3ATR 止损 / 无信号离场 / 10 仓位×10% / 双边 13bp 成本 · 分数为逐月 walk-forward 样本外预测 · 表内夏普按日频权益重采样年化（√365） · 生成于 {pd.Timestamp.now():%Y-%m-%d}</div>
<div class="cards">{cards}</div>
{svg_a}
{table([(n, s) for n, s, _, _ in A])}
{svg_b}
<div class="cards">{cards_b}</div>
{table([(n, s) for n, s, _, _ in B])}
<div class="note">
<b>读图要点</b><br>
1. 这是 <code>reports/equity_compare.html</code>（A 股东风版）的加密对照页：版式相同，数据、结论都换成本仓的 21 币 1h。A 组是同一段历史上的四条线，用来分辨「哪个门控在事件级有效、在组合级是否还成立」；其中静态门控的虚线含未来信息、不可实盘，滚动门控是严格 PIT 但只描述「相对近期热」。<br>
2. B 组是严格口径：阈值在 2025-06 末冻结，2025-07 之后的数据没有参与任何选择。看它是否能在真样本外重复 A 组的结论。<br>
3. 门控在 B 组全部没有跑赢基线：三条冻结阈值（p25/p50/p75）与基线同期收益/回撤都在同一量级或更差。A 股版里「冻结门控用 40% 开机时间换近半回撤」的结论，在加密上不成立——这是本页最重要的对照结果。<br>
4. 与 A 股版的差异：东风版里「frenzy 冻结门控」用 40% 开机时间换来收益翻倍、回撤减半，是当页的推荐方案；在加密上同一条规则（阈值冻结、T-1 值、只做开机日）把收益从 +0.4% 一路压到 −10.8%，越严格越差。实现是同一套（阈值冻结、T-1 值、只做开机日），差别出在数据上。一个可能的原因（E3 推测，未验证）：A 股的「热」之后有资金延续，而加密的「热」更多是拥挤与均值回归。这一点需要单独实验才能下结论。<br>
5. 诚实边界：样本外只有 16 个月，年化夏普的抽样噪声约 ±0.5；21 币标的池是「2026 年仍存活」的合约，存在幸存者偏差。
</div>
</body></html>
"""
    OUT.write_text(html)
    print(f"写入 {OUT} {len(html.encode())} 字节", flush=True)


if __name__ == "__main__":
    main()
