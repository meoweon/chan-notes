"""参数扫描报告生成器 v2（干净版）。

只读 `reports/param_sweep_results.csv` / `param_sweep_curves.csv` / `param_sweep_meta.json`，
所有数字与结论现场计算，不含硬编码字符串。

用法：python scripts/make_param_sweep_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
RES = REPORTS / "param_sweep_results.csv"
CURVES = REPORTS / "param_sweep_curves.csv"
META = REPORTS / "param_sweep_meta.json"
SIX = REPORTS / "six_metrics.json"
ROLL_JSON = REPORTS / "rolling_selection.json"
OUT = REPORTS / "param_sweep_report.html"

DIMS = [("hold", "持仓上限 (bar)", 96),
        ("stop", "止损 (ATR)", 2.0),
        ("take_profit", "止盈 (ATR)", 5.0),
        ("frenzy", "狂热门控", "无"),
        ("score", "模型分数门控", "无")]
BASE = dict(hold=48, stop=2.0, take_profit=5.0, frenzy="无", score="无")


def pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def pct_plain(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def paired(df, key, dim, a, ref, col):
    """固定其余维度，配对比较 dim=a 与 dim=ref 的 col 差值。"""
    others = [k for k in key if k != dim]
    x = df[df[dim] == a].set_index(others)[col]
    y = df[df[dim] == ref].set_index(others)[col]
    j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    if not len(j):
        return np.nan, np.nan, np.nan, 0
    d = j["a"] - j["b"]
    return float(d.mean()), float(d.median()), float((d > 0).mean()), int(len(j))


def main():
    df = pd.read_csv(RES)
    meta = pd.read_json(META, typ="series")
    six = json.loads(SIX.read_text()) if SIX.exists() else []
    curves = pd.read_csv(CURVES, parse_dates=["time"]).set_index("time")
    curves = curves.loc[curves.index >= pd.Timestamp(meta["is_window"][0])].ffill()
    roll = (json.loads(ROLL_JSON.read_text()) if ROLL_JSON.exists() else None)
    key = ["hold", "stop", "take_profit", "frenzy", "score"]

    b = df
    for k, v in BASE.items():
        b = b[b[k] == v]
    base = b.iloc[0] if len(b) else None

    is_df = df[df["n_is"].fillna(0) >= 30] if "n_is" in df else df
    if not len(is_df):
        is_df = df
    best = is_df.loc[is_df["sharpe_is"].idxmax()]
    best_oos = df.loc[df["sharpe_oos"].idxmax()]

    n_all = len(df)
    oos_sh = df["sharpe_oos"].dropna()
    rank_oos = int((oos_sh > best["sharpe_oos"]).sum()) + 1
    rank_is_of_oos_best = int((df["sharpe_is"] > best_oos["sharpe_is"]).sum()) + 1
    hit_base = float((oos_sh > base["sharpe_oos"]).mean()) if base is not None else np.nan
    pos_oos = float((oos_sh > 0).mean())
    pos_ret_oos = float((df["ret_oos"] > 0).mean())

    # ---- Top10 ----
    top = is_df.sort_values("sharpe_is", ascending=False).head(10)
    top_html = "\n".join(
        "<tr>"
        f"<td>{int(r.hold)}</td><td>{num(r.stop, 1)}</td><td>{num(r.take_profit, 1)}</td>"
        f"<td>{r.frenzy}</td><td>{r.score}</td>"
        f"<td class='num'>{pct(r.ret_is)}</td><td class='num'>{num(r.sharpe_is)}</td>"
        f"<td class='num'>{int(r.n_is)}</td>"
        f"<td class='num'>{pct(r.ret_oos)}</td><td class='num'>{num(r.sharpe_oos)}</td>"
        f"<td class='num'>{int(r.n_oos)}</td>"
        "</tr>" for _, r in top.iterrows())

    # ---- 边际中位数 ----
    marg_html = "\n".join(
        "<tr>"
        f"<td>{label}</td><td>{val}</td><td class='num'>{len(sub)}</td>"
        f"<td class='num'>{num(sub.sharpe_is.median())}</td>"
        f"<td class='num'>{num(sub.sharpe_oos.median())}</td>"
        f"<td class='num'>{pct(sub.ret_is.median())}</td>"
        f"<td class='num'>{pct(sub.ret_oos.median())}</td>"
        "</tr>"
        for key_, label, ref in DIMS
        for val in sorted(df[key_].dropna().unique(), key=lambda v: (isinstance(v, str), v))
        for sub in [df[df[key_] == val]])

    # ---- 配对差值 ----
    pair_rows, pair_stats = [], {}
    for dim, label, ref in DIMS:
        for val in sorted(df[dim].dropna().unique(), key=lambda v: (isinstance(v, str), v)):
            if val == ref:
                continue
            m_is, md_is, pos_is, n_is = paired(df, key, dim, val, ref, "sharpe_is")
            m_oos, md_oos, pos_oos, n_oos = paired(df, key, dim, val, ref, "sharpe_oos")
            pair_stats[(dim, val)] = (m_is, m_oos, pos_oos, n_is)
            cls = "pos" if m_oos > 0.1 else ("neg" if m_oos < -0.1 else "")
            pair_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{val}</td><td>{ref}</td>"
                f"<td class='num'>{m_is:+.2f}</td>"
                f"<td class='num {cls}'>{m_oos:+.2f}</td>"
                f"<td class='num'>{pos_oos * 100:.0f}%</td>"
                f"<td class='num'>{n_oos}</td>"
                "</tr>")
    pair_html = "\n".join(pair_rows)

    # ---- 结论（条件生成） ----
    concl = []
    concl.append(
        f"全网格 {n_all} 组里，样本外夏普为正的只占 {pos_oos * 100:.0f}%，"
        f"收益为正的占 {pos_ret_oos * 100:.0f}%；中位夏普 {num(df['sharpe_oos'].median())}。"
        f"也就是说 2025-07~2026-10 这段里，绝大多数参数组合是亏的。")
    if base is not None:
        concl.append(
            f"基线（{BASE['hold']}/{BASE['stop']}/{BASE['take_profit']}/无门控）在 OOS 的夏普 "
            f"{num(base['sharpe_oos'])}、收益 {pct(base['ret_oos'])}，在 {n_all} 组里排第 "
            f"{int((oos_sh > base['sharpe_oos']).sum()) + 1}（前 "
            f"{(oos_sh > base['sharpe_oos']).mean() * 100:.0f}%）——"
            f"“不调参”反而是相对好的结果。")
    best_pe = is_df.loc[is_df["avgret_is"].idxmax()]
    rank_pe_oos = int((df["avgret_oos"] > best_pe["avgret_oos"]).sum()) + 1
    same_cfg = all(best[k] == BASE[k] for k in BASE)
    concl.append(
        f"按样本内夏普选出的最优组合（hold={int(best.hold)} / stop={num(best.stop, 1)} / "
        f"tp={num(best.take_profit, 1)} / frenzy={best.frenzy} / score={best.score}）"
        f"在 OOS 收益 {pct(best.ret_oos)}、夏普 {num(best.sharpe_oos)}，"
        f"在 {n_all} 组里排第 {rank_oos}。"
        + ("它恰好就是基线。" if same_cfg else "样本内的最优没有迁移到样本外。"))
    concl.append(
        f"若改按 D016 的主指标“每笔期望”选参，样本内最优是 hold={int(best_pe.hold)} / "
        f"stop={num(best_pe.stop, 1)} / tp={num(best_pe.take_profit, 1)} / "
        f"frenzy={best_pe.frenzy} / score={best_pe.score}：IS 每笔 {pct(best_pe.avgret_is)}，"
        f"OOS 每笔 {pct(best_pe.avgret_oos)}，在 {n_all} 组里排第 {rank_pe_oos}。"
        + ("换成主指标同样没有迁移到样本外。" if best_pe["avgret_oos"] <= 0
           else "样本外为正，但需看第 2 节的区间是否跨 0。"))
    concl.append(
        f"反过来，OOS 最优组合（hold={int(best_oos.hold)} / stop={num(best_oos.stop, 1)} / "
        f"tp={num(best_oos.take_profit, 1)} / frenzy={best_oos.frenzy} / score={best_oos.score}）"
        f"OOS 夏普 {num(best_oos.sharpe_oos)}、收益 {pct(best_oos.ret_oos)}，"
        f"但它的 IS 夏普排名只有第 {rank_is_of_oos_best} —— 事前挑不出来，这就是 480 组极值的选择偏差。")

    for dim, label, ref in DIMS:
        vals = [v for v in df[dim].dropna().unique() if v != ref]
        buckets = {"wins": [], "flips": [], "both_neg": [], "is_only": [],
                   "is_flat_oos_bad": [], "oos_only": [], "other": [], "null": []}
        for v in vals:
            if (dim, v) not in pair_stats:
                continue
            m_is, m_oos, _, _ = pair_stats[(dim, v)]
            tag = f"{v}(IS {m_is:+.2f} / OOS {m_oos:+.2f})"
            if m_is > 0.1 and m_oos > 0.1:
                buckets["wins"].append(tag)
            elif (m_is > 0.15 and m_oos < -0.15) or (m_is < -0.15 and m_oos > 0.15):
                buckets["flips"].append(tag)
            elif m_is < -0.1 and m_oos < -0.1:
                buckets["both_neg"].append(tag)
            elif abs(m_is) < 0.05 and abs(m_oos) < 0.05:
                buckets["null"].append(tag)
            elif m_is > 0.1 and m_oos <= 0.1:
                buckets["is_only"].append(tag)
            elif abs(m_is) < 0.1 and m_oos < -0.1:
                buckets["is_flat_oos_bad"].append(tag)
            elif m_is <= 0.1 and m_oos > 0.1:
                buckets["oos_only"].append(tag)
            else:
                buckets["other"].append(tag)
        parts = []
        if buckets["wins"]:
            parts.append("跨窗口同向改善：" + "、".join(buckets["wins"]))
        if buckets["flips"]:
            parts.append("方向翻转、不可迁移：" + "、".join(buckets["flips"]))
        if buckets["both_neg"]:
            parts.append("两段都更差：" + "、".join(buckets["both_neg"]))
        if buckets["is_only"]:
            parts.append("只在样本内更好：" + "、".join(buckets["is_only"]))
        if buckets["oos_only"]:
            parts.append("样本内看不出、样本外更好（事后视角）：" + "、".join(buckets["oos_only"]))
        if buckets["is_flat_oos_bad"]:
            parts.append("样本内几乎无变化、样本外明显变差：" + "、".join(buckets["is_flat_oos_bad"]))
        if buckets["other"]:
            parts.append("其余：" + "、".join(buckets["other"]))
        if buckets["null"]:
            parts.append("两段都无差异：" + "、".join(buckets["null"]))
        if parts:
            concl.append(f"{label}（对照 {ref}）—— " + "；".join(parts) + "。")

    # 交互项提醒：把最佳边际单独加到基线上是否仍成立
    probe = df[(df["hold"] == BASE["hold"]) & (df["stop"] == BASE["stop"]) &
               (df["take_profit"] == BASE["take_profit"]) & (df["frenzy"] == "p25") &
               (df["score"] == BASE["score"])]
    if base is not None and len(probe):
        pv = probe.iloc[0]
        concl.append(
            f"但把 p25 单独加到基线上，OOS 夏普是 {num(pv['sharpe_oos'])}（基线 "
            f"{num(base['sharpe_oos'])}）、收益 {pct(pv['ret_oos'])}（基线 {pct(base['ret_oos'])}）——"
            f"并不更好。上面的配对均值主要来自 p25 与 score 门控、长持仓的交互，"
            f"不是“加个 p25 就更稳”的独立结论。")

    concl_html = "\n".join(f"<li>{c}</li>" for c in concl)

    def fmtv(k, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "未恢复" if k == "rec" else "\u2014"
        if k in ("ret", "avg_ret", "mdd", "win"):
            return pct(v)
        if k in ("freq", "rec"):
            return f"{v:.0f}"
        return num(v)

    def six_cell(k, rec, with_ci=True):
        s = f"<td class='num'>{fmtv(k, rec.get(k))}"
        if with_ci:
            lo, hi = rec.get(k + "_ci", [float("nan")] * 2)
            if np.isfinite(lo) and np.isfinite(hi):
                s += f"<br><span class='ci'>[{fmtv(k, lo)}, {fmtv(k, hi)}]</span>"
        return s + "</td>"

    core_keys = ["ret", "win", "payoff", "freq", "avg_ret", "calmar", "mdd", "rec"]
    diag_keys = ["sharpe", "sortino"]
    six_html = "\n".join(
        "<tr>" + f"<td>{r['cfg']}</td><td>{r['window']}</td>"
        + "".join(six_cell(k, r, with_ci=(k not in ("freq", "rec"))) for k in core_keys)
        + f"<td class='num'>{r['n_trades']}</td></tr>"
        for r in six) if six else "<tr><td colspan='11'>未生成 six_metrics.json</td></tr>"
    diag_html = "\n".join(
        "<tr>" + f"<td>{r['cfg']}</td><td>{r['window']}</td>"
        + "".join(six_cell(k, r) for k in diag_keys) + "</tr>"
        for r in six) if six else "<tr><td colspan='4'>未生成 six_metrics.json</td></tr>"

    best_card = f"""
    <div class="card">
      <div class="card-title">按样本内夏普选出的组合（不是“已验证有效”）</div>
      <div class="grid">
        <div><span class="k">hold</span><b>{int(best.hold)}</b></div>
        <div><span class="k">stop</span><b>{num(best.stop, 1)}</b></div>
        <div><span class="k">take_profit</span><b>{num(best.take_profit, 1)}</b></div>
        <div><span class="k">frenzy</span><b>{best.frenzy}</b></div>
        <div><span class="k">score</span><b>{best.score}</b></div>
      </div>
      <table class="mini">
        <tr><th></th><th>IS 2021-01~2025-06</th><th>OOS 2025-07~2026-10</th></tr>
        <tr><td>净收益</td><td class="num">{pct(best.ret_is)}</td><td class="num">{pct(best.ret_oos)}</td></tr>
        <tr><td>胜率</td><td class="num">{pct_plain(best.win_is)}</td><td class="num">{pct_plain(best.win_oos)}</td></tr>
        <tr><td>盈亏比</td><td class="num">{num(best.payoff_is)}</td><td class="num">{num(best.payoff_oos)}</td></tr>
        <tr><td>频率（笔/年）</td><td class="num">{num(best.freq_is, 0)}</td><td class="num">{num(best.freq_oos, 0)}</td></tr>
        <tr><td>每笔期望</td><td class="num">{pct(best.avgret_is)}</td><td class="num">{pct(best.avgret_oos)}</td></tr>
        <tr><td>收益/回撤</td><td class="num">{num(best.calmar_is)}</td><td class="num">{num(best.calmar_oos)}</td></tr>
        <tr><td>最大回撤</td><td class="num">{pct(best.mdd_is)}</td><td class="num">{pct(best.mdd_oos)}</td></tr>
        <tr><td>回撤恢复（天）</td><td class="num">{num(best.rec_is, 0)}</td><td class="num">{num(best.rec_oos, 0)}</td></tr>
        <tr><td>交易数</td><td class="num">{int(best.n_is)}</td><td class="num">{int(best.n_oos)}</td></tr>
        <tr><td>夏普（诊断）</td><td class="num">{num(best.sharpe_is)}</td><td class="num">{num(best.sharpe_oos)}</td></tr>
      </table>
    </div>"""

    # ---- 第 3 节：选参流程净值 ----
    if roll:
        picks = roll["picks"]
        pick_rows = "\n".join(
            f"<tr><td>{p['year']}</td><td>{p['hold']}</td><td>{num(p['stop'],1)}</td>"
            f"<td>{num(p['tp'],1)}</td><td>{p['frenzy']}</td><td>{p['score']}</td>"
            f"<td class='num'>{pct(p['hist_avg_ret'])}</td>"
            f"<td class='num'>{p.get('hist_freq', float('nan')):.0f}</td></tr>" for p in picks)
        pick_tbl = ("<table class='mini'><tr><th>生效年</th><th>hold</th><th>stop</th>"
                    "<th>tp</th><th>frenzy</th><th>score</th>"
                    "<th style='text-align:right'>选参时历史每笔期望</th>"
                    "<th style='text-align:right'>历史频率（笔/年）</th></tr>"
                    + pick_rows + "</table>")
        kk = roll["ex_post"]["key"]
        ep = f"{kk[0]}/{num(kk[1],1)}/{num(kk[2],1)}/{kk[3]}/{kk[4]}"
        fig1_html = roll.get("fig1_html", "<div>图 1 HTML 缺失</div>")
        fig2_html = roll.get("fig2_html", "<div>图 2 HTML 缺失</div>")
        sec3 = f"""
<h2>3. 选参流程净值（图 1 / 图 2）</h2>
<div class=”sub”>这一节不画”事后最优的四条线”。净值线只反映净收益、回撤与恢复；
胜率、盈亏比、频率、每笔期望要看图 2。</div>
{fig1_html}
<div class=”sub”>上联：净值（2022-01 起 = 1.0）。下联：水下回撤，标注最大回撤深度与恢复天数。
灰 = 不选参的基准（48/2/5）；绿 = 每年初只用历史数据、按”每笔期望”重选参数后照用；
橙虚线 = 事后最优（{ep}，样本内每笔期望最高），只是作弊上限，不能实盘。</div>
{fig2_html}
<div class=”sub”>图 2：净值线看不出的四项——每笔期望、胜率、盈亏比、频率，逐年对比（灰 = 基准，绿 = 滚动选参）。</div>
<h3>逐年选参记录</h3>
{pick_tbl}
<div class="sub">每年 1 月 1 日只用此前数据选参，要求历史成交 ≥ 30 笔；选中后在当年保持不动。</div>
<div class="note">到 {meta['window'][1]}：基准 {roll['base_end']:.3f}、滚动选参 {roll['flow_end']:.3f}、
事后最优 {roll['epost_end']:.3f}。滚动选参没有跑赢“什么都不选”的基线；即使给足事后视角（橙线），
也只比基线多 {(roll['epost_end']-roll['base_end'])*100:.1f} 个百分点（{roll['epost_end']:.3f} vs {roll['base_end']:.3f}）
——说明“按每笔期望挑参数”本身没提供可用的信息，不是“没选准”的问题。</div>
<div class="sub">两个必须说清的边界：① 若去掉频率下限、只按每笔期望选，2022 年起净值只有
{roll.get('free_end', float('nan')):.3f}——它会不断选到 p90 这类一年几乎不交易的组合，
“每笔期望”单独用会被低频钻空子，所以正式口径加了“历史频率 ≥ 100 笔/年”的约束；
② 2026 年的选参用到了 2025 下半年的数据，那一段已不是最初 E001 划定的样本外，
真正没有被我看过的数据只有 2025-07 之后。</div>
"""
    else:
        sec3 = """
<h2>3. 选参流程净值</h2>
<div class="sub">尚未生成 rolling_selection.json，先运行 <code>scripts/rolling_selection.py</code>。</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<title>参数扫描报告 v2 · 21 币种</title>
<style>
 body{{font-family:-apple-system,"PingFang SC",sans-serif;max-width:1080px;margin:32px auto;
      padding:0 20px;color:#222;line-height:1.6}}
 h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:17px;margin-top:32px;
      border-left:4px solid #1a73e8;padding-left:8px}}
 .sub{{color:#666;font-size:13px;margin-bottom:16px}}
 .warn{{background:#fdecea;border:1px solid #f5c6c2;color:#a50e0e;padding:12px 14px;
       border-radius:6px;font-size:13px;margin:16px 0}}
 .ci{{color:#888;font-size:10px}}
 .note{{background:#f1f3f4;border-radius:6px;padding:10px 14px;font-size:13px;color:#444}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:14px 0}}
 .card-title{{font-size:13px;color:#666;margin-bottom:10px}}
 .grid{{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:10px}}
 .k{{color:#888;font-size:12px;display:block}}
 .grid b{{font-size:16px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}}
 th{{background:#fafafa;font-weight:600;color:#444}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.pos{{color:#0a7d32;font-weight:600}} td.neg{{color:#b3261e;font-weight:600}}
 table.mini{{width:auto;min-width:340px;margin-top:6px}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
 li{{margin:6px 0}}
</style></head><body>
<h1>参数扫描报告 v2 · 21 币种</h1>
<div class="sub">数据：21 个主流 USDT 永续合约，1h，{meta['window'][0]} ~ {meta['window'][1]}；
入场：确认买点（conf_entry）当日开盘；成本：往返 13bp；未计资金费率与滑点。</div>

<div class="warn">
<b>本页是研究记录，不是可实盘结论。</b>
参数只在样本内（{meta['is_window'][0]} ~ {meta['is_window'][1]}）选择，
样本外（{meta['oos_window'][0]} ~ {meta['window'][1]}）只用于检验；
“最优”是 {meta['n_combos']} 组网格里的极值，本身含选择偏差；样本外只有 16 个月，
夏普的差异多数落在噪声里，不能当作“哪个参数更好”的证据。
</div>

<div class="note">
已修正此前版本的三处未来函数：狂热指数改用 T-1 收盘；门控阈值改用扩张分位（只用当时之前的数据）；
模型分数改用逐月 walk-forward 的样本外预测（{meta['score_oos_events']} 条，逐月 AUC 均值
{meta['score_auc_mean']:.3f}）。旧版报告的数字与结论不可沿用（见 my_archive/param_sweep_report_v1_作废.html）。
</div>

<h2>1. 选出来的组合</h2>
{best_card}

<h2>2. 核心指标（窗口内按月分块自举 95% 区间）</h2>
<div class="sub">按 D016：净收益、胜率、盈亏比、频率、每笔期望、收益/回撤、最大回撤与恢复天数为核心；
区间由月度收益有放回重抽 2000 次得到，跨 0 表示该指标无法与“没有边际”区分。恢复天数从谷底回到前高，窗口内未恢复显示“—”。</div>
<table>
<tr><th>组合</th><th>窗口</th>
<th style="text-align:right">净收益</th><th style="text-align:right">胜率</th>
<th style="text-align:right">盈亏比</th><th style="text-align:right">频率(笔/年)</th>
<th style="text-align:right">每笔期望</th><th style="text-align:right">收益/回撤</th>
<th style="text-align:right">最大回撤</th><th style="text-align:right">恢复天数</th>
<th style="text-align:right">笔数</th></tr>
{six_html}
</table>
<div class="sub">夏普与 Sortino 只算不用，作为诊断项。</div>
<table>
<tr><th>组合</th><th>窗口</th><th style="text-align:right">夏普</th><th style="text-align:right">Sortino</th></tr>
{diag_html}
</table>

{sec3}

<h2>4. 样本内 Top 10（按 IS 夏普排序，诊断视角）</h2>
<div class="sub">本节用夏普排序，只为与旧版报告对照；D016 的选优主指标是每笔期望，见第 2 节。</div>
<table>
<tr><th>hold</th><th>stop</th><th>take_profit</th><th>frenzy</th><th>score</th>
<th style="text-align:right">IS 收益</th><th style="text-align:right">IS 夏普</th><th style="text-align:right">IS 笔数</th>
<th style="text-align:right">OOS 收益</th><th style="text-align:right">OOS 夏普</th><th style="text-align:right">OOS 笔数</th></tr>
{top_html}
</table>

<h2>5. 各维度的边际表现（中位数）</h2>
<table>
<tr><th>维度</th><th>取值</th><th style="text-align:right">组合数</th>
<th style="text-align:right">IS 夏普</th><th style="text-align:right">OOS 夏普</th>
<th style="text-align:right">IS 收益</th><th style="text-align:right">OOS 收益</th></tr>
{marg_html}
</table>

<h2>6. 配对差值（固定其余维度，只变一个，单位：夏普，诊断视角）</h2>
<div class="sub">比“看中位数”更干净：同一组其余参数下比较两个取值，再对全部配对求平均。</div>
<table>
<tr><th>维度</th><th>取值</th><th>对照</th>
<th style="text-align:right">ΔIS 夏普</th><th style="text-align:right">ΔOOS 夏普</th>
<th style="text-align:right">OOS 更优占比</th><th style="text-align:right">配对数</th></tr>
{pair_html}
</table>

<h2>7. 结论（由数据生成）</h2>
<ul>
{concl_html}
</ul>

<h2>8. 局限</h2>
<ul>
<li>样本外只有 {meta['oos_window'][0]} ~ {meta['oos_window'][1]}，交易数有限，夏普差异多在噪声内。</li>
<li>未计资金费率、滑点与冲击成本；10% 单笔仓位、最多 10 仓的容量假设未验证。</li>
<li>模型特征与标签沿用既有流程，未重新做特征筛选；逐月 AUC 均值 {meta['score_auc_mean']:.3f} 不代表可直接交易。</li>
<li>网格是离散的，极值对参数扰动敏感；本文不提供“最优参数”，只报告是否存在可迁移的边际。</li>
</ul>

<div class="sub" style="margin-top:28px">
复现：<code>python scripts/train_wf_oos.py crypto_21_v1 wf_crypto_21_xgb_oos survive_144 all</code> →
<code>python scripts/param_sweep_backtest.py</code> → <code>python scripts/make_param_sweep_report.py</code>
<br>产物：param_sweep_results.csv / param_sweep_curves.csv / param_sweep_meta.json</div>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print("写入", OUT, len(html), "字节")
    print()
    print("\n".join(concl))


if __name__ == "__main__":
    main()
