"""日线 regime 压力测试：events_d1 确认买点 144bar/3ATR 分年度表现。

对照 30m 的逐年结论（2024 慢涨失效），检验日线口径在 2010-2026 各 regime
（2015 股灾、2016 熔断后、2018 熊市、2019-2021 结构牛、2022-2024 熊、2024-09 狂潮）下的期望。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.models import train

NAME = sys.argv[1] if len(sys.argv) > 1 else "events_d1"

df = train.load(NAME)
ok = df.dropna(subset=["pol144_ret"]).copy()
ok["t"] = pd.to_datetime(ok["conf_entry_time"])
ok["year"] = ok["t"].dt.year

# 市场年度收益（日线 _MARKET）
from chan_notes.data import store
mkt = store.load("_MARKET", "d")
mkt["t"] = pd.to_datetime(mkt["time"])
mkt["year"] = mkt["t"].dt.year
mret = mkt.groupby("year")["close"].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)

rows = []
for y, g in ok.groupby("year"):
    r = g["pol144_ret"]
    rows.append({
        "year": y, "n": len(g), "codes": g.code.nunique(),
        "E_bp": r.mean() * 1e4, "med_bp": r.median() * 1e4,
        "win%": (r > 0).mean() * 100,
        "W_bp": r[r > 0].mean() * 1e4 if (r > 0).any() else 0,
        "L_bp": -r[r <= 0].mean() * 1e4 if (r <= 0).any() else 0,
        "top5%share": r[r >= r.quantile(0.95)].sum() / r[r > 0].sum() * 100 if (r > 0).any() and r[r > 0].sum() != 0 else np.nan,
        "mkt%": mret.get(y, np.nan) * 100,
    })
a = pd.DataFrame(rows)
a["R"] = a["W_bp"] / a["L_bp"].replace(0, np.nan)
print("=== events_d1 conf_buy 144bar/3ATR 分年度 ===")
print(a.round(1).to_string(index=False))
print(f"\n全样本: n={len(ok)} E={ok.pol144_ret.mean()*1e4:.1f}bp win={(ok.pol144_ret>0).mean()*100:.1f}%")

# 去尾（对照 30m Phase A）
r = ok["pol144_ret"]
for q in (0.99, 0.95, 0.90):
    kept = r[r <= r.quantile(q)]
    print(f"去 top{(1-q)*100:.0f}%: E={kept.mean()*1e4:.1f}bp (n={len(kept)})")

out = Path(__file__).resolve().parents[1] / "reports" / "daily_regime.csv"
a.to_csv(out, index=False)
print(f"saved {out}")
