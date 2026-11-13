"""月度 Walk-Forward：扩展窗口训练，逐月 OOF 预测。

用法:
  python train_wf.py <dataset> <out> [label_col] [subset]
  subset: all | buy | conf_buy | conf_all
"""
import json
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.config import MODELS
from chan_notes.features import schema
from chan_notes.models import train

H = 24


def _subset(df, name):
    if name == "buy":
        return df[df.is_buy == 1]
    if name == "sell":
        return df[df.is_buy == 0]
    if name == "conf_buy":
        return df[(df.is_buy == 1) & (df.conf_ok == 1)]
    if name == "conf_all":
        return df[df.conf_ok == 1]
    return df


def run(name="events_v1", out="wf_v1", label="pol_dir_ret", subset="all",
        min_train=1500, matures_days=28, min_pos=0.05, thr=0.0, window_months=0):
    # matures_days：训练截点与测试月的隔离天数。144bar 标签（30m≈18 交易日≈26 日历日）
    # 需 ≥28 天才保证训练标签在预测时点前已成熟（旧默认 10 天不够，台账 #37）。
    df = _subset(train.load(name), subset).copy()
    df["t"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("t").reset_index(drop=True)
    df = df[pd.notna(df[label])]          # NaN 标签剔除（旧版被当负类）
    y = (df[label] > thr).astype(int).to_numpy()
    cols = [c for c in schema.load() if c in df.columns]
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    print(f"events={len(df)} pos={y.mean():.3f} features={len(cols)}")

    months = pd.period_range(df["t"].min().to_period("M") + 12, df["t"].max().to_period("M"), freq="M")
    preds, last, imp_rows = [], None, []
    for m in months:
        start = m.start_time
        cut = start - pd.Timedelta(days=matures_days)
        if window_months > 0:
            tr = (df["t"] <= cut) & (df["t"] > cut - pd.DateOffset(months=window_months))
        else:
            tr = df["t"] <= cut
        nxt = (m + 1).start_time
        te = (df["t"] >= start) & (df["t"] < nxt)   # 只含本月（旧版 end_time+1day 会重复计入次月 1 日）
        if tr.sum() < min_train or te.sum() < 5:
            continue
        pos_all = np.flatnonzero(tr.to_numpy())
        if y[pos_all].mean() < min_pos or y[pos_all].mean() > 1 - min_pos:
            continue
        # 独立时间验证集：训练窗末段 15%（旧版把训练集同时当验证集 → 训练集上早停）
        vsplit = int(len(pos_all) * 0.85)
        ptr, pva = pos_all[:vsplit], pos_all[vsplit:]
        if len(pva) < 10:
            ptr, pva = pos_all, pos_all[:max(1, len(pos_all) // 10)]
        keep = [c for c in cols if X.iloc[ptr][c].nunique(dropna=True) > 1]
        bst = train.fit_xgb(X.iloc[ptr][keep], y[ptr], X.iloc[pva][keep], y[pva], rounds=120)
        for fn_, g_ in bst.get_score(importance_type="gain").items():
            imp_rows.append({"month": str(m), "feature": fn_, "gain": g_})
        bi = getattr(bst, "best_iteration", None)
        it_rng = (0, bi + 1) if bi is not None else None
        te_pos = np.flatnonzero(te.to_numpy())
        dm = __import__("xgboost").DMatrix(X.iloc[te_pos][keep])
        p = bst.predict(dm, iteration_range=it_rng) if it_rng else bst.predict(dm)
        sub = df.loc[te, ["code", "event_time", "bsp_type", "is_buy", "delay_bars",
                          label, "dir_ret_24", "ret_24"]].copy()
        sub["score"] = p
        sub["month"] = str(m)
        preds.append(sub)
        last = (bst, keep)

    if not preds:
        print("no predictions")
        return None
    out_df = pd.concat(preds, ignore_index=True)
    p = str(MODELS.parent / "events" / f"{out}.parquet")
    out_df.to_parquet(p, index=False)
    print(f"saved {p} rows={len(out_df)}")
    if imp_rows:
        ip = str(MODELS.parent / "events" / f"{out}_importance.parquet")
        pd.DataFrame(imp_rows).to_parquet(ip, index=False)
        print(f"saved importance -> {ip}")
    if last:
        bst, keep = last
        bst.save_model(str(MODELS / "latest.json"))
        json.dump(keep, open(MODELS / "latest_features.json", "w"))
    return out_df


def evaluate(df, label="pol_dir_ret", thr=0.0):
    from chan_notes.models.train import auc
    print("\n## OOF 评估")
    rows = []
    for side, sd in ((1, "BUY"), (0, "SELL")):
        s = df[df.is_buy == side]
        if len(s) < 50:
            continue
        y = (s[label] > thr).astype(int)
        hs = s[s.score >= s.score.quantile(0.8)][label]
        ls = s[s.score <= s.score.quantile(0.2)][label]
        rows.append({"group": sd, "n": len(s), "auc": auc(y, s.score),
                     "base": y.mean(), "top20_bp": hs.mean() * 1e4,
                     "bot20_bp": ls.mean() * 1e4,
                     "top20_win": (hs > 0).mean()})
        for t in ("1", "1p", "2", "2s", "3a"):
            st = s[s.bsp_type.str.split(",").apply(lambda xs: t in xs)]
            if len(st) >= 100:
                yt = (st[label] > thr).astype(int)
                rows.append({"group": f"{sd}_{t}", "n": len(st),
                             "auc": auc(yt, st.score), "base": yt.mean(),
                             "top20_bp": st[st.score >= st.score.quantile(0.8)][label].mean() * 1e4,
                             "bot20_bp": st[st.score <= st.score.quantile(0.2)][label].mean() * 1e4,
                             "top20_win": (st[st.score >= st.score.quantile(0.8)][label] > 0).mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    monthly_auc(df, label, thr)


def monthly_auc(df, label, thr):
    from chan_notes.models.train import auc
    rows = []
    for m, g in df.groupby("month"):
        y = (g[label] > thr).astype(int)
        if y.nunique() > 1 and len(g) >= 10:
            rows.append({"month": m, "n": len(g), "auc": auc(y, g["score"])})
    if not rows:
        return
    a = pd.DataFrame(rows)
    print(f"\n逐月 AUC: mean={a.auc.mean():.4f}  std={a.auc.std():.4f}  min={a.auc.min():.4f}  "
          f">=0.55 占比 {(a.auc >= 0.55).mean()*100:.0f}%  ({len(a)} 个月)")
    a.to_csv(str(MODELS.parent / "events" / "monthly_auc.csv"), index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    name = a[0] if len(a) > 0 else "events_v1"
    out = a[1] if len(a) > 1 else "wf_v1"
    label = a[2] if len(a) > 2 else "pol_dir_ret"
    subset = a[3] if len(a) > 3 else "all"
    mt = int(a[4]) if len(a) > 4 else 1500
    thr = float(a[5]) if len(a) > 5 else 0.0
    win = int(a[6]) if len(a) > 6 else 0
    d = run(name, out, label, subset, min_train=mt, thr=thr, window_months=win)
    if d is not None:
        evaluate(d, label, thr)
