"""门控实验室：每个门控单独开关，OOS 组合级对比。

流程对齐 scripts/frenzy_frozen.py（台账 #19 范式）：
IS 期（< split）中位数冻结阈值 → PIT shift(1) 事件过滤 → portfolio.run。
输出列对齐 reports/frenzy_frozen_oos.csv，可横向对比。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.data import store
from chan_notes.eval.bars import load_frames
from chan_notes.eval.metrics import expectancy
from chan_notes.gates import (MARKET_DEFAULTS, daily_close_wide,
                              filter_events, freeze_threshold, get_gate,
                              is_percentile)
from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", required=True, choices=list(MARKET_DEFAULTS))
    ap.add_argument("--events", default=None, help="事件集名（默认取市场默认）")
    ap.add_argument("--freq", default=None)
    ap.add_argument("--hold", type=int, default=None)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--split", default="2024-01-01", help="IS/OOS 分界")
    ap.add_argument("--p-on", type=float, default=0.5,
                    help="IS 期目标开机率，阈值取 IS 的 (1-p_on) 分位；默认 0.5=中位数")
    ap.add_argument("--dump-series", action="store_true",
                    help="把门控日频值及其 IS 历史分位写入 reports/")
    ap.add_argument("--gates", default="frenzy",
                    help="门控，逗号分隔；'+' 表示 AND 组合。breadth 已停用，需显式指定")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = MARKET_DEFAULTS[args.market]
    events_name = args.events or d["events"]
    if events_name is None:
        sys.exit(f"market={args.market} 暂无默认事件集，请用 --events 指定")
    freq = args.freq or d["freq"]
    hold = args.hold if args.hold is not None else d["hold"]
    stop = args.stop if args.stop is not None else d["stop"]

    df = train.load(events_name)
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    df = df[df["event_time"] >= args.split]
    print(f"事件集 {events_name}: OOS(>= {args.split}) conf_buy {len(df)} 笔, "
          f"{df.code.nunique()} 个标的")

    close_wide = daily_close_wide(store.BARS / freq, args.market)
    print(f"门控宇宙: {close_wide.shape[1]} 个标的, "
          f"{close_wide.index[0]}..{close_wide.index[-1]}")

    bars = load_frames(df.code.unique(), freq)
    bars = {c: b[b["time"] >= args.split] for c, b in bars.items()}
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=stop,
                           hold_bars=hold, signal_exit=False)

    variants = [("always_on", df, {})]
    for spec in [s.strip() for s in args.gates.split(",") if s.strip()]:
        # "frenzy+breadth" = 两门控同时开机（AND）；"frenzy" = 单门控
        names = [n.strip() for n in spec.split("+") if n.strip()]
        sub = df
        meta = {}
        for gname in names:
            gate = get_gate(gname)
            series = gate.compute(close_wide, args.market)
            thr = freeze_threshold(series, args.split, 1.0 - args.p_on)
            is_pct = is_percentile(series, args.split, thr)
            oos_med = float(series[series.index >= args.split].median())
            oos_pct = is_percentile(series, args.split, oos_med)
            sub = filter_events(sub, series, thr)
            rate = len(sub) / len(df) * 100 if len(df) else 0.0
            print(f"门控 {gname}（{spec}）: p_on={args.p_on:.0%} → IS 阈值 {thr:.5f}"
                  f"（IS 第 {is_pct:.0f} 分位）, OOS 中位 {oos_med:.5f}"
                  f"（IS 第 {oos_pct:.0f} 分位）, 开机 {len(sub)}/{len(df)}（{rate:.0f}%）")
            if len(names) == 1:
                meta = {"thr": round(thr, 6), "is_pct": round(is_pct, 1),
                        "oos_med_pct": round(oos_pct, 1)}
                if args.dump_series:
                    is_part = series[series.index < args.split].dropna()
                    pct = series.map(lambda v: float((is_part <= v).mean() * 100)
                                     if pd.notna(v) else float("nan"))
                    path = ROOT / "reports" / f"gate_series_{args.market}_{gname}.csv"
                    pd.DataFrame({"value": series, "is_pct": pct}).to_csv(path)
                    print(f"  序列分位 saved {path}")
        variants.append((spec, sub, meta))

    rows = []
    for tag, sub, meta in variants:
        res = portfolio.run(sub, bars, p, entry_mode="confirm")
        m = res["metrics"]
        e = expectancy(res["trades"]["ret"])
        rows.append({
            "variant": tag, "trades": m["n_trades"],
            "total%": m["total_ret"] * 100, "cagr%": m["cagr"] * 100,
            "sharpe": m["sharpe"], "mdd%": m["mdd"] * 100,
            "E_bp": e["expectancy_bp"], "win%": e["win_rate"] * 100,
            "R": e["payoff_R"], "kelly%": e["kelly"] * 100,
            "thr": meta.get("thr"), "is_pct": meta.get("is_pct"),
            "oos_med_pct": meta.get("oos_med_pct"),
        })
        res["trades"].to_parquet(
            ROOT / "data" / "bt" / f"gate_{args.tag}_{tag}_trades.parquet",
            index=False)
        res["equity"].to_parquet(
            ROOT / "data" / "bt" / f"gate_{args.tag}_{tag}_equity.parquet")

    a = pd.DataFrame(rows)
    print()
    print(a.round(2).to_string(index=False))
    out = Path(args.out) if args.out else (
        ROOT / "reports" / f"gate_lab_{args.market}_{args.tag}.csv")
    a.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
