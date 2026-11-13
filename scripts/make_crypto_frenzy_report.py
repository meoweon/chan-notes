"""生成加密市场 frenzy 门控可视化报告（HTML）。

对比：基线（全部交易） vs frenzy 门控（只在狂热时开机）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.config import DATA
from chan_notes.eval.bars import load_frames
from chan_notes.models import train

FREQ = "1h"
HOLD, STOP = 48, 2.0
FRENZY_THR = 0.06  # p75 阈值

ROOT = Path(__file__).resolve().parents[1]
BAR_DIR = ROOT / "data" / "bars" / FREQ

# --- 构建狂热指数 ---
daily = {}
for f in sorted(BAR_DIR.glob("*.parquet")):
    if f.stem == "_MARKET":
        continue
    b = pd.read_parquet(f, columns=["time", "close"])
    b["d"] = b["time"].astype(str).str[:10]
    g = b.groupby("d")["close"].last()
    daily[f.stem] = g
px = pd.DataFrame(daily).sort_index()
rets = px.pct_change()
frenzy = (rets > 0.08).sum(axis=1) / rets.notna().sum(axis=1)
frenzy5 = frenzy.rolling(5, min_periods=1).mean()

# --- 加载事件 ---
NAME = "crypto_10_v1"
df = train.load(NAME)
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
bars = load_frames(df.code.unique(), FREQ)

# --- 基线回测 ---
p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=STOP, hold_bars=HOLD, signal_exit=True)
res_base = portfolio.run(df, bars, p, entry_mode="confirm")

# --- frenzy 门控回测 ---
df["d"] = pd.to_datetime(df["event_time"]).dt.strftime("%Y-%m-%d")
df["frenzy"] = df["d"].map(frenzy5)
df_gate = df[df["frenzy"] >= FRENZY_THR].copy()
res_gate = portfolio.run(df_gate, bars, p, entry_mode="confirm")

# --- 生成 HTML ---


def fmt_pct(v):
    return f"{v*100:+.1f}%" if v >= 0 else f"{v*100:.1f}%"


def fmt_bp(v):
    return f"{v*1e4:+.1f}"


base_m = res_base["metrics"]
gate_m = res_gate["metrics"]

# 合并权益曲线（同一坐标系）
eq_base = res_base["equity"]["equity"]
eq_gate = res_gate["equity"]["equity"]
all_eq = pd.concat([eq_base.rename("base"), eq_gate.rename("gate")], axis=1)
all_eq = all_eq.ffill().fillna(1.0)
vmin = min(all_eq["base"].min(), all_eq["gate"].min())
vmax = max(all_eq["base"].max(), all_eq["gate"].max())
if vmax == vmin:
    vmax = vmin + 1

W, H = 900, 240
dx = W / (len(all_eq) - 1)
dy = H / (vmax - vmin)

def to_path(s: pd.Series, color: str) -> str:
    pts = []
    for i, v in enumerate(s.values):
        x = i * dx
        y = H - (v - vmin) * dy
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>'

path_base = to_path(all_eq["base"], "#94a3b8")
path_gate = to_path(all_eq["gate"], "#10b981")

# 时间轴标签
t0 = all_eq.index.min()
t1 = all_eq.index.max()
label_left = str(t0)[:10] if hasattr(t0, 'strftime') else str(t0)[:10]
label_right = str(t1)[:10] if hasattr(t1, 'strftime') else str(t1)[:10]

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>加密市场 frenzy 门控报告</title>
<style>
body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #0f172a; background: #fff; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.sub {{ color: #64748b; font-size: 13px; margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0 24px; }}
.card {{ background: #f8fafc; border-radius: 8px; padding: 14px 16px; border-left: 4px solid #94a3b8; }}
.card.gate {{ border-left-color: #10b981; }}
.card-title {{ font-weight: 700; font-size: 14px; margin-bottom: 6px; }}
.card-desc {{ font-size: 12px; color: #475569; line-height: 1.8; }}
.metric {{ font-size: 20px; font-weight: 700; }}
.metric.pos {{ color: #10b981; }}
.metric.neg {{ color: #ef4444; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: right; }}
th {{ background: #f1f5f9; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
.note {{ font-size: 12px; color: #64748b; line-height: 1.8; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 12px 16px; margin: 18px 0; }}
.chart {{ background: #fafafa; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.legend {{ display: flex; gap: 24px; font-size: 12px; margin-top: 8px; }}
.legend span {{ display: inline-block; width: 12px; height: 3px; margin-right: 6px; vertical-align: middle; }}
</style></head><body>

<h1>加密市场 · frenzy 门控回测报告</h1>
<div class="sub">数据：10 主流合约（BTC/ETH/BNB/XRP/ADA/DOGE/LINK/SOL/AVAX/DOT）1h · 2020-01 ~ 2026-10 · 确认买点入场 / {HOLD}bar / {STOP}ATR 止损 / 10 仓位×10% · 狂热指数 = 日涨幅>8% 币种占比 5 日平滑</div>

<div class="cards">
  <div class="card">
    <div class="card-title">基线（全部交易）</div>
    <div class="card-desc">
      累计 <span class="metric {'pos' if base_m.get('total_ret',0)>=0 else 'neg'}">{fmt_pct(base_m.get('total_ret',0))}</span><br>
      年化 {fmt_pct(base_m.get('cagr',0))} · 夏普 {base_m.get('sharpe',0):.2f} · MDD {fmt_pct(base_m.get('mdd',0))}<br>
      交易 {base_m.get('n_trades',0)} 笔 · 胜率 {base_m.get('win_rate',0)*100:.1f}% · 期望 {fmt_bp(base_m.get('avg_ret',0))}bp
    </div>
  </div>
  <div class="card gate">
    <div class="card-title">frenzy 门控（≥p75 时开机）</div>
    <div class="card-desc">
      累计 <span class="metric {'pos' if gate_m.get('total_ret',0)>=0 else 'neg'}">{fmt_pct(gate_m.get('total_ret',0))}</span><br>
      年化 {fmt_pct(gate_m.get('cagr',0))} · 夏普 {gate_m.get('sharpe',0):.2f} · MDD {fmt_pct(gate_m.get('mdd',0))}<br>
      交易 {gate_m.get('n_trades',0)} 笔 · 胜率 {gate_m.get('win_rate',0)*100:.1f}% · 期望 {fmt_bp(gate_m.get('avg_ret',0))}bp
    </div>
  </div>
</div>

<div class="chart">
  <svg width="100%" viewBox="0 0 {W} {H+30}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="{W}" height="{H}" fill="#fafafa"/>
    <line x1="0" y1="{H - (1.0 - vmin) * dy}" x2="{W}" y2="{H - (1.0 - vmin) * dy}" stroke="#e2e8f0" stroke-dasharray="4"/>
    {path_base}
    {path_gate}
    <text x="4" y="{H+16}" font-size="11" fill="#94a3b8">{label_left}</text>
    <text x="{W-60}" y="{H+16}" font-size="11" fill="#94a3b8">{label_right}</text>
  </svg>
  <div class="legend">
    <div><span style="background:#94a3b8"></span>基线（全部交易）</div>
    <div><span style="background:#10b981"></span>frenzy 门控（开机时交易）</div>
    <div style="color:#94a3b8">虚线 = 初始净值 1.0</div>
  </div>
</div>

<table>
<tr><th>指标</th><th>基线</th><th>frenzy 门控</th><th>变化</th></tr>
<tr><td>累计收益</td><td>{fmt_pct(base_m.get('total_ret',0))}</td><td>{fmt_pct(gate_m.get('total_ret',0))}</td><td>{fmt_pct(gate_m.get('total_ret',0) - base_m.get('total_ret',0))}</td></tr>
<tr><td>年化收益</td><td>{fmt_pct(base_m.get('cagr',0))}</td><td>{fmt_pct(gate_m.get('cagr',0))}</td><td>{fmt_pct(gate_m.get('cagr',0) - base_m.get('cagr',0))}</td></tr>
<tr><td>夏普比率</td><td>{base_m.get('sharpe',0):.2f}</td><td>{gate_m.get('sharpe',0):.2f}</td><td>{gate_m.get('sharpe',0) - base_m.get('sharpe',0):+.2f}</td></tr>
<tr><td>最大回撤</td><td>{fmt_pct(base_m.get('mdd',0))}</td><td>{fmt_pct(gate_m.get('mdd',0))}</td><td>—</td></tr>
<tr><td>交易笔数</td><td>{base_m.get('n_trades',0)}</td><td>{gate_m.get('n_trades',0)}</td><td>{gate_m.get('n_trades',0) - base_m.get('n_trades',0):+d}</td></tr>
<tr><td>胜率</td><td>{base_m.get('win_rate',0)*100:.1f}%</td><td>{gate_m.get('win_rate',0)*100:.1f}%</td><td>{(gate_m.get('win_rate',0) - base_m.get('win_rate',0))*100:+.1f}%</td></tr>
<tr><td>期望值 (bp)</td><td>{fmt_bp(base_m.get('avg_ret',0))}</td><td>{fmt_bp(gate_m.get('avg_ret',0))}</td><td>{fmt_bp(gate_m.get('avg_ret',0) - base_m.get('avg_ret',0))}</td></tr>
<tr><td>开机率</td><td>100%</td><td>{gate_m.get('n_trades',0)/base_m.get('n_trades',1)*100:.0f}%</td><td>—</td></tr>
</table>

<div class="note">
<strong>结论</strong>：frenzy 门控在加密市场同样成立——只在狂热期（涨幅>8% 币种占比 ≥ p75）开机，其余时间空仓。
<br>与 A 股结论一致：缠论结构本身不含可复制 alpha，"什么时候开机"才是唯一有效的增强。
<br>本质仍是尾部游戏：收益集中在少数狂潮窗口，门控的作用是把暴露集中在这些窗口。
</div>

</body></html>"""

out_path = ROOT / "reports" / "crypto_frenzy_report.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"saved {out_path}")
print(f"\n基线: {fmt_pct(base_m.get('total_ret',0))} / 夏普 {base_m.get('sharpe',0):.2f}")
print(f"门控: {fmt_pct(gate_m.get('total_ret',0))} / 夏普 {gate_m.get('sharpe',0):.2f}")
