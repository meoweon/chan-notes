"""bg_ 背景特征单变量有效性检验（stop gate：先证据后模型）。

判定标准（预先承诺，防事后挑数）：
- 五分位均值对分位序号的 Spearman 相关 |ρ| ≥ 0.8（单调）；
- 且 顶-底 分位收益差 ≥ 20bp（效应量有意义）；
- 两个口径都看：全部确认买点（2020-2026）、冻结门控内子集（2024-2026 OOS）。
通过的才允许进 WF；不通过的进墓地。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2024-01-01")

ev = pd.read_parquet(ROOT / "data" / "events" / "events_v8.parquet")
ev["event_time"] = pd.to_datetime(ev["event_time"])
sub = ev[(ev.is_buy == 1) & (ev.conf_ok == 1) & ev.pol144_ret.notna()].copy()

fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"])
fz["prev5"] = fz.frenzy5.shift(1)
thr = float(fz.loc[fz.d < CUTOFF, "frenzy5"].median())
sub["_d"] = sub.event_time.dt.strftime("%Y-%m-%d")
sub["fz"] = sub._d.map(dict(zip(fz["d"].dt.strftime("%Y-%m-%d"), fz["prev5"])))
gated = sub[(sub.event_time >= CUTOFF) & (sub.fz >= thr)]

bg_cols = [c for c in ev.columns if c.startswith("bg_")]
print(f"conf_buy={len(sub)}  gated_oos={len(gated)}  bg_features={len(bg_cols)}")


def spearman(x, y):
    x, y = pd.Series(x).rank(), pd.Series(y).rank()
    return float(x.corr(y))


def quintile_test(df, col, label="pol144_ret"):
    d = df[[col, label]].dropna()
    if len(d) < 200 or d[col].nunique() < 5:
        return None
    d = d.assign(q=pd.qcut(d[col].rank(method="first"), 5, labels=False))
    g = d.groupby("q")[label].agg(["mean", "count"])
    rho = spearman(g.index, g["mean"])
    spread = (g["mean"].iloc[-1] - g["mean"].iloc[0]) * 1e4
    return rho, spread, g["mean"] * 1e4


rows = []
for c in bg_cols:
    r1 = quintile_test(sub, c)
    r2 = quintile_test(gated, c)
    row = {"feature": c}
    if r1:
        row.update(rho_all=round(r1[0], 2), spread_all=round(r1[1], 1),
                   means_all=" ".join(f"{v:+.0f}" for v in r1[2]))
    if r2:
        row.update(rho_gated=round(r2[0], 2), spread_gated=round(r2[1], 1),
                   means_gated=" ".join(f"{v:+.0f}" for v in r2[2]))
    ok = (r1 and abs(r1[0]) >= 0.8 and abs(r1[1]) >= 20) or \
         (r2 and abs(r2[0]) >= 0.8 and abs(r2[1]) >= 20)
    row["PASS"] = bool(ok)
    rows.append(row)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(ROOT / "reports" / "bg_check.csv", index=False)
passed = out.loc[out.PASS, "feature"].tolist()
print(f"\nPASS: {len(passed)}/{len(bg_cols)}: {passed}")
print("saved reports/bg_check.csv")
