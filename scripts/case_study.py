"""存活模型案例教学：真实个股 K 线对比（低分被擦除 vs 高分存活）。

输出 reports/survival_case.html —— 两张手绘 K 线图（SVG）+ 大白话讲解。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]

# ── 数据合并：OOF 分数 + 事件元数据（clear_bar / conf / pol144_ret）──
wf = pd.read_parquet(ROOT / "data" / "events" / "wf_surv.parquet")
ev = train.load("events_v7")
cols = ["code", "event_time", "bsp_type", "is_buy", "bar_idx", "clear_bar",
        "conf_ok", "conf_entry_idx", "conf_entry_price", "conf_atr", "pol144_ret"]
ev = ev[cols].copy()
ev["event_time"] = pd.to_datetime(ev["event_time"])
wf["event_time"] = pd.to_datetime(wf["event_time"])
m = wf.merge(ev, on=["code", "event_time", "bsp_type", "is_buy"], how="left")
buy = m[(m.is_buy == 1) & m.conf_ok.notna()].copy()
print(f"merged buy events with OOF score: {len(buy)}")

# ── 选案例：优先同一只股票的一死一活 ──
dead = buy[(buy.survive_144 == 0) & (buy.score <= 0.25) & (buy.pol144_ret <= -0.04)
           & (buy.conf_ok == 1) & (buy.event_time >= "2023-01-01")]
dead["delay"] = dead.clear_bar - dead.bar_idx
dead = dead[(dead.delay >= 8) & (dead.delay <= 80)]
alive = buy[(buy.survive_144 == 1) & (buy.score >= 0.85) & (buy.pol144_ret >= 0.12)
            & (buy.conf_ok == 1) & (buy.event_time >= "2023-01-01")]
common = set(dead.code) & set(alive.code)
print(f"候选：低分被擦除 {len(dead)} / 高分存活 {len(alive)} / 同股兼有 {len(common)}")

case_a = dead[dead.code.isin(common)].sort_values("score").iloc[0] if common else dead.sort_values("score").iloc[0]
same = alive[alive.code == case_a.code]
case_b = same.sort_values("score", ascending=False).iloc[0] if len(same) else alive.sort_values("score", ascending=False).iloc[0]
for tag, c in (("A(死)", case_a), ("B(活)", case_b)):
    print(f"{tag} {c.code} {c.event_time} 类型{c.bsp_type} score={c.score:.3f} "
          f"clear={int(c.clear_bar)} entry={int(c.conf_entry_idx)} ret={c.pol144_ret:+.2%}")

# ── K 线渲染 ──
def load_bars(code):
    b = pd.read_parquet(ROOT / "data" / "bars" / "30m" / f"{code}.parquet")
    b["time"] = pd.to_datetime(b["time"])
    return b.reset_index(drop=True)


def draw(case, title, W=1120, H=380):
    code = case.code
    bars = load_bars(code)
    eb = int(case.bar_idx)
    i0 = max(0, eb - 60)
    i1 = min(len(bars), eb + 150)
    win = bars.iloc[i0:i1].reset_index(drop=True)
    ml, mr, mt, mb = 64, 16, 40, 30
    pw, ph = W - ml - mr, H - mt - mb
    lo, hi = win.low.min(), win.high.max()
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    n = len(win)
    cw = pw / n

    def X(i):
        return ml + (i - i0) * cw + cw / 2

    def Y(p):
        return mt + (hi - p) / (hi - lo) * ph

    s = [f'<svg width="{W}" height="{H + 8}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Microsoft YaHei,PingFang SC,sans-serif">']
    s.append(f'<rect width="{W}" height="{H + 8}" fill="white"/>')
    s.append(f'<text x="{ml}" y="24" font-size="15" font-weight="700" fill="#0f172a">{title}</text>')

    # 网格 + 日期刻度
    for f in np.linspace(0.15, 0.85, 4):
        p = lo + (hi - lo) * f
        s.append(f'<line x1="{ml}" y1="{Y(p):.1f}" x2="{ml + pw}" y2="{Y(p):.1f}" stroke="#f1f5f9"/>')
        s.append(f'<text x="{ml - 6}" y="{Y(p) + 4:.1f}" font-size="10" fill="#94a3b8" text-anchor="end">{p:.2f}</text>')
    step = max(8, n // 12)
    for i in range(i0 - i0, n, step):
        t = win.time.iloc[i]
        if i == 0 or win.time.iloc[i].date() != win.time.iloc[max(0, i - step)].date():
            s.append(f'<text x="{X(i + i0):.1f}" y="{H - 12}" font-size="9" fill="#94a3b8" text-anchor="middle">'
                     f'{t.strftime("%y-%m-%d")}</text>')
            s.append(f'<line x1="{X(i + i0):.1f}" y1="{mt}" x2="{X(i + i0):.1f}" y2="{mt + ph}" stroke="#f8fafc"/>')

    # 持有窗口阴影（入场→+144）
    ei = int(case.conf_entry_idx)
    if i0 <= ei < i1:
        x0 = X(ei)
        x1 = X(min(ei + 144, i1 - 1))
        ret = case.pol144_ret
        col = "#dcfce7" if ret > 0 else "#fee2e2"
        s.append(f'<rect x="{x0:.1f}" y="{mt}" width="{max(2, x1 - x0):.1f}" height="{ph}" fill="{col}" opacity="0.5"/>')

    # K 线（红涨绿跌）
    for i, r in enumerate(win.itertuples()):
        gi = i0 + i
        x = X(gi)
        up = r.close >= r.open
        col = "#dc2626" if up else "#16a34a"
        s.append(f'<line x1="{x:.1f}" y1="{Y(r.high):.1f}" x2="{x:.1f}" y2="{Y(r.low):.1f}" stroke="{col}" stroke-width="1"/>')
        y0, y1 = Y(max(r.open, r.close)), Y(min(r.open, r.close))
        s.append(f'<rect x="{x - cw * 0.32:.1f}" y="{y0:.1f}" width="{cw * 0.64:.1f}" height="{max(1, y1 - y0):.1f}" fill="{col}"/>')

    # 信号出现标记（事件 bar）
    if i0 <= eb < i1:
        ex, ey = X(eb), Y(win.low.iloc[eb - i0]) + 16
        s.append(f'<polygon points="{ex:.1f},{ey - 10:.1f} {ex - 5:.1f},{ey + 2:.1f} {ex + 5:.1f},{ey + 2:.1f}" fill="#2563eb"/>')
        s.append(f'<line x1="{ex:.1f}" y1="{mt}" x2="{ex:.1f}" y2="{mt + ph}" stroke="#2563eb" stroke-dasharray="3,3"/>')
        s.append(f'<text x="{ex + 4:.1f}" y="{mt + 14}" font-size="11" font-weight="700" fill="#2563eb">'
                 f'信号出现（模型分 {case.score:.2f}）</text>')

    # 入场价 / 止损线
    if i0 <= ei < i1:
        ep = case.conf_entry_price
        s.append(f'<line x1="{X(ei):.1f}" y1="{Y(ep):.1f}" x2="{ml + pw}" y2="{Y(ep):.1f}" stroke="#2563eb" stroke-dasharray="6,3" stroke-width="1.2"/>')
        s.append(f'<text x="{ml + pw - 4}" y="{Y(ep) - 4:.1f}" font-size="10" fill="#2563eb" text-anchor="end">确认入场 {ep:.2f}</text>')
        stop = ep - 3 * case.conf_atr
        if lo < stop:
            s.append(f'<line x1="{X(ei):.1f}" y1="{Y(stop):.1f}" x2="{ml + pw}" y2="{Y(stop):.1f}" stroke="#dc2626" stroke-dasharray="4,4"/>')
            s.append(f'<text x="{ml + pw - 4}" y="{Y(stop) - 4:.1f}" font-size="10" fill="#dc2626" text-anchor="end">3ATR 止损 {stop:.2f}</text>')

    # 撤销标记
    cb = int(case.clear_bar)
    if cb > 0 and i0 <= cb < i1:
        cx, cy = X(cb), Y(win.high.iloc[cb - i0]) - 14
        s.append(f'<line x1="{cx:.1f}" y1="{mt}" x2="{cx:.1f}" y2="{mt + ph}" stroke="#dc2626" stroke-width="1.5"/>')
        s.append(f'<text x="{cx - 4:.1f}" y="{cy:.1f}" font-size="14" font-weight="700" fill="#dc2626" text-anchor="end">✖</text>')
        s.append(f'<text x="{cx - 4:.1f}" y="{cy - 6:.1f}" font-size="11" font-weight="700" fill="#dc2626" text-anchor="end">'
                 f'第 {cb - eb} 根：信号被擦除</text>')
    s.append('</svg>')
    return "".join(s)


svg_a = draw(case_a, f'案例 A · {case_a.code} · {case_a.event_time:%Y-%m-%d} · 类型{case_a.bsp_type}买点 · 模型分 {case_a.score:.2f}（底部 {int((wf.score <= case_a.score).mean() * 100)}%）→ 被擦除')
svg_b = draw(case_b, f'案例 B · {case_b.code} · {case_b.event_time:%Y-%m-%d} · 类型{case_b.bsp_type}买点 · 模型分 {case_b.score:.2f}（顶部 {int((wf.score >= case_b.score).mean() * 100)}%）→ 存活')

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>存活模型案例：这个买点会不会被擦掉？</title>
<style>
body {{ font-family:"Microsoft YaHei","PingFang SC",sans-serif; max-width:1160px; margin:24px auto; padding:0 16px; color:#0f172a; }}
h1 {{ font-size:22px; margin-bottom:4px; }}
.sub {{ color:#64748b; font-size:13px; margin-bottom:18px; }}
.big {{ background:#0f172a; color:#fff; border-radius:10px; padding:16px 20px; font-size:15px; line-height:1.9; margin-bottom:22px; }}
.big b {{ color:#fbbf24; }}
.panel {{ border:1px solid #e2e8f0; border-radius:10px; padding:14px; margin-bottom:22px; }}
.verdict {{ display:flex; gap:12px; margin-top:10px; }}
.vc {{ flex:1; border-radius:8px; padding:10px 14px; font-size:13px; line-height:1.7; }}
.bad {{ background:#fef2f2; border:1px solid #fecaca; }}
.good {{ background:#f0fdf4; border:1px solid #bbf7d0; }}
.mean {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:14px 18px; font-size:13.5px; line-height:2; }}
.mean b {{ color:#b45309; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; margin-top:10px; }}
th,td {{ border:1px solid #e2e8f0; padding:6px 10px; text-align:left; }}
th {{ background:#f1f5f9; }}
</style></head><body>
<h1>存活模型的交易意义：这个买点会不会被"擦掉"？</h1>
<div class="sub">真实个股案例 · 30m K 线（红涨绿跌）· 蓝色虚线=信号与入场 · 红色竖线=信号被框架擦除 · 底纹=持有窗口 · 2026-11-01</div>

<div class="big">
<b>一句大白话：</b>你在实盘中看到的缠论买卖点，<b>58% 会在之后被行情"擦掉"重画</b>——今天图上的"二类买"，下个月再看就不存在了。
存活模型做的事：信号出现的那一瞬间，告诉你<b>"这个信号有多大概率会被擦掉"</b>（0~1 分）。
分越低越像"将死的信号"，分越高越像"能进历史的真结构"。
</div>

<div class="panel">
{svg_a}
<div class="verdict">
<div class="vc bad"><b>当时你看到的：</b>{case_a.event_time:%Y-%m-%d} 出现一个类型{case_a.bsp_type}买点，图很标准。模型打分 <b>{case_a.score:.2f}</b>——全场最低的一档（底部 {int((wf.score <= case_a.score).mean() * 100)}%）。<br>
<b>后来发生的：</b>第 {int(case_a.clear_bar - case_a.bar_idx)} 根 bar，框架重算结构，<b>这个买点从图上被擦掉了</b>——它从来不是真结构。如果按信号入场并持有 144 根：<b>{case_a.pol144_ret:+.1%}</b>。</div>
</div>
</div>

<div class="panel">
{svg_b}
<div class="verdict">
<div class="vc good"><b>当时你看到的：</b>{case_b.event_time:%Y-%m-%d} 出现类型{case_b.bsp_type}买点。模型打分 <b>{case_b.score:.2f}</b>——全场最高的一档（顶部 {int((wf.score >= case_b.score).mean() * 100)}%）。<br>
<b>后来发生的：</b>之后 144 根 bar 这个买点<b>一直挂在图上</b>，结构被时间确认。按信号入场并持有 144 根：<b>{case_b.pol144_ret:+.1%}</b>。</div>
</div>
</div>

<div class="mean">
<b>所以它在交易上的意义是什么？</b><br>
① <b>它是"假信号探测器"，不是"赚钱筛选器"。</b>低分信号（&lt;0.3）意味着七成概率这个买点会被擦掉——别当真、别补仓、别给它编故事；高分信号才值得按流程走完止损和持有期。<br>
② <b>它不能用来提高收益（已实测）。</b>只做高分信号的组合反而少赚（+13.4% vs +46.8%）——因为狂潮期好信号扎堆，过滤恰好削掉了妖股暴露。赚钱靠的是"什么时候开机"（frenzy 闸门），不是"挑哪个信号"。<br>
③ <b>它的正确岗位：信号可信度标签。</b>实盘中每个新买点旁边实时显示一个分数——你看到的不再是"一个买点"，而是"一个 75% 概率能活下来的买点"。决策仍按机械规则走，但你对信号的<b>心理权重</b>有了依据。<br>
④ <b>它证明了缠论不是事后看图。</b>结构的可存活性在出现当下就有可识别的特征（笔振幅、在线段中的位置、力度比），同一套结构规则预测结构自身——AUC 0.66，67 个月 100% 稳定。
</div>

<table>
<tr><th>指标</th><th>案例 A（低分→死）</th><th>案例 B（高分→活）</th><th>全样本（4.1 万个买点）</th></tr>
<tr><td>模型分</td><td>{case_a.score:.2f}</td><td>{case_b.score:.2f}</td><td>买侧 AUC 0.658</td></tr>
<tr><td>结局</td><td>第 {int(case_a.clear_bar - case_a.bar_idx)} 根被擦除</td><td>144 根内存活</td><td>58% 最终被擦除</td></tr>
<tr><td>144 根交易结果</td><td><b style="color:#dc2626">{case_a.pol144_ret:+.1%}</b></td><td><b style="color:#16a34a">{case_b.pol144_ret:+.1%}</b></td><td>存活组 E=+271.7bp vs 死亡组 -190.1bp</td></tr>
</table>

<div class="sub" style="margin-top:16px">注：模型分为 walk-forward 样本外打分（只用该月之前 12 个月数据训练），无未来函数。案例选自 2023 年后数据。</div>
</body></html>"""

out = ROOT / "reports" / "survival_case.html"
out.write_text(html, encoding="utf-8")
print(f"saved {out} ({out.stat().st_size // 1024} KB)")
