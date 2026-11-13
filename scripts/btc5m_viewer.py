"""BTC 5 分钟 K 线查看器：多窗口蜡烛图 + BSP 买卖点标注 → 单文件 HTML。
窗口：最近一段 / 2024-11 大选突破 / 2021-05-19 崩盘 / 2025 区间震荡。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

bars = pd.read_parquet(ROOT / "data" / "bars_crypto" / "5m" / "BTCUSDT.parquet")
bars["time"] = pd.to_datetime(bars["time"])
ev = pd.read_parquet(ROOT / "data" / "events" / "events_b5.parquet",
                     columns=["bar_idx", "event_time", "bsp_type", "is_buy"])

def pick(center=None, n=480, tail=False):
    if tail:
        return bars.iloc[-n:].reset_index(drop=True)
    c = pd.Timestamp(center)
    i = bars["time"].searchsorted(c)
    return bars.iloc[max(0, i - n // 2): i + n // 2].reset_index(drop=True)

UP, DN = "#c0392b", "#1e8449"  # 红涨绿跌（A股习惯）

def render(win, title, w=1180, h=340):
    n = len(win)
    lo, hi = win["low"].min(), win["high"].max()
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    cw = w / n
    bw = max(1.0, cw * 0.62)
    def y(p):
        return h - 24 - (p - lo) / (hi - lo) * (h - 44)
    t0 = bars["time"].searchsorted(win["time"].iloc[0])
    marks = ev[(ev.bar_idx >= t0) & (ev.bar_idx < t0 + n)]
    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{w}" height="{h}" fill="#fafafa"/>']
    # 网格 + 价格刻度
    for i in range(5):
        yy = 20 + i * (h - 44) / 4
        pv = hi - i * (hi - lo) / 4
        parts.append(f'<line x1="0" y1="{yy:.0f}" x2="{w}" y2="{yy:.0f}" stroke="#e3e3e3"/>')
        parts.append(f'<text x="{w-4}" y="{yy+10:.0f}" font-size="10" fill="#888" text-anchor="end">{pv:,.0f}</text>')
    # 日期刻度
    prev_d = None
    for i, t in enumerate(win["time"]):
        d = t.strftime("%m-%d")
        if d != prev_d:
            prev_d = d
            parts.append(f'<line x1="{i*cw:.1f}" y1="0" x2="{i*cw:.1f}" y2="{h-24}" stroke="#efefef"/>')
            parts.append(f'<text x="{i*cw+3:.1f}" y="{h-10}" font-size="10" fill="#999">{d}</text>')
    # 蜡烛
    for i, r in win.iterrows():
        x = i * cw
        col = UP if r["close"] >= r["open"] else DN
        parts.append(f'<line x1="{x+cw/2:.1f}" y1="{y(r["high"]):.1f}" x2="{x+cw/2:.1f}" y2="{y(r["low"]):.1f}" stroke="{col}" stroke-width="0.8"/>')
        yo, yc = y(r["open"]), y(r["close"])
        top, hh = min(yo, yc), max(abs(yc - yo), 0.8)
        fill = "#ffffff" if r["close"] >= r["open"] else col
        parts.append(f'<rect x="{x+(cw-bw)/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{fill}" stroke="{col}" stroke-width="0.8"/>')
    # BSP 标注
    for _, m in marks.iterrows():
        x = (m.bar_idx - t0) * cw + cw / 2
        p = bars["low"].iloc[m.bar_idx] if m.is_buy else bars["high"].iloc[m.bar_idx]
        yy = y(p) + (14 if m.is_buy else -14)
        tri = f'M {x:.1f} {yy-6 if m.is_buy else yy+6:.1f} l -4 {8 if m.is_buy else -8} l 8 0 z'
        col = "#1a5fb4" if m.is_buy else "#b03a2e"
        parts.append(f'<path d="{tri}" fill="{col}" opacity="0.9"><title>{m.event_time} {m.bsp_type}</title></path>')
    parts.append("</svg>")
    nb = (marks.is_buy == 1).sum() if len(marks) else 0
    ns = (marks.is_buy == 0).sum() if len(marks) else 0
    return (f'<div class="win"><h3>{title}</h3>'
            f'<div class="meta">{win["time"].iloc[0]} ~ {win["time"].iloc[-1]} ｜ {n} 根 ｜ '
            f'<span style="color:#1a5fb4">▲ 买点 {nb}</span> ｜ <span style="color:#b03a2e">▼ 卖点 {ns}</span></div>'
            + "".join(parts) + "</div>")

windows = [
    (pick(tail=True), "① 最近一段（数据末尾）"),
    (pick("2024-11-06 12:00"), "② 2024-11-06 美国大选突破段"),
    (pick("2021-05-19 12:00"), "③ 2021-05-19 崩盘段（单日 -30% 级别）"),
    (pick("2025-06-15 00:00"), "④ 2025-06 区间震荡段"),
]

r = bars["close"].pct_change().dropna()
atr_pct = (bars["high"] - bars["low"]).mean() / bars["close"].mean() * 1e4
html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>BTC 5分钟K线</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;margin:24px;background:#fff;color:#222;max-width:1240px}}
h1{{font-size:20px}} .win{{margin:18px 0;padding:12px;border:1px solid #ddd;border-radius:8px}}
h3{{margin:4px 0}} .meta{{font-size:12px;color:#666;margin-bottom:6px}}
.stats{{display:flex;gap:14px;flex-wrap:wrap}} .card{{border:1px solid #ddd;border-radius:8px;padding:10px 16px;font-size:13px}}
.card b{{font-size:17px;display:block}}</style></head><body>
<h1>BTC/USDT 永续 · 5 分钟 K 线（币安，2020-01 ~ 2026-10，共 {len(bars):,} 根）</h1>
<div class="stats">
<div class="card"><b>{len(bars):,}</b>总 bar 数（6.8 年）</div>
<div class="card"><b>{bars["close"].iloc[0]:,.0f} → {bars["close"].iloc[-1]:,.0f}</b>首末收盘（USDT）</div>
<div class="card"><b>{atr_pct:.0f}bp</b>单 bar 平均振幅（高低差/价）</div>
<div class="card"><b>{r.std()*1e4:.0f}bp</b>单 bar 收益波动</div>
</div>
<p style="font-size:13px;color:#555">▲ 蓝三角 = 缠论买点（events_b5，10,279 个事件中的窗口内部分）；▼ 红三角 = 卖点。红涨绿跌。</p>
{''.join(render(w_, t) for w_, t in windows)}
</body></html>"""

out = ROOT / "reports" / "btc_5m_kline.html"
out.write_text(html, encoding="utf-8")
print(f"saved {out}  {out.stat().st_size/1024:.0f}KB")
