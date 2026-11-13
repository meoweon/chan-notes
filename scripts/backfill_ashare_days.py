"""按「本地缺哪些 (币,周期,日)」补 A 股 bars：baostock 拉日区间，只追加不重写。

与 download_ashare.py 的区别：后者整表覆盖且对空返回静默跳过（季度丢块时不报错）；
本脚本先读本地 parquet 自己在 [start,end] 里找缺的时间戳，只补缺的，合并走
backfill_binance_daily._atomic_merge 的 6 条断言（旧行不丢、列序、单调、有量必有额）。

追加前会核对重叠时间戳的上游读数与本地逐位相同；任何偏差说明复权基准变了，
此时不允许只追加（会拼出两条复权基准），脚本直接失败并要求整表重下。

用法:
    python scripts/backfill_ashare_days.py --start 2026-11-09 --end 2026-11-11 --dry-run
    python scripts/backfill_ashare_days.py --start 2026-11-09 --end 2026-11-11
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backfill_binance_daily as bfd  # noqa: E402
from chan_notes.data import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FREQ_MAP = {"15m": "15", "30m": "30", "1h": "60"}
BARS_PER_DAY = {"15m": 16, "30m": 8, "1h": 4}
FIELDS = "date,time,code,open,high,low,close,volume,amount"


def fetch(code: str, baostock_freq: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        code, FIELDS, start_date=start, end_date=end,
        frequency=baostock_freq, adjustflag="2",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"{code} {baostock_freq}: {rs.error_msg}")
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "amount"])
    df = pd.DataFrame(rows, columns=FIELDS.split(","))
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["date", "code"]).sort_values("time").reset_index(drop=True)


def to_store(code: str, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "time": df["time"], "code": code,
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "volume": df["volume"], "amount": df["amount"], "adjustflag": "2",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", default="15m,30m,1h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tfs = [x.strip() for x in args.timeframes.split(",") if x.strip()]
    t0 = time.time()

    # 计划阶段：只读本地，列出缺的时间戳
    plan = []
    for tf in tfs:
        per = BARS_PER_DAY[tf]
        for p in sorted((ROOT / "data" / "bars" / tf).glob("*.parquet")):
            if not p.stem.startswith(("sh.", "sz.")):
                continue
            t = pd.read_parquet(p, columns=["time"]).time
            lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(days=1)
            local = set(t[(t >= lo) & (t < hi)])
            plan.append({"code": p.stem, "tf": tf, "local_in_range": len(local), "per_day": per})

    print("计划:", len(plan), "个 (代码,周期)；区间", args.start, "~", args.end)
    by_tf = pd.Series([x["tf"] for x in plan]).value_counts().to_dict()
    print("  按周期:", by_tf)
    if args.dry_run:
        print("  (dry-run，未联网)")
        return

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"baostock 登录失败: {lg.error_msg}")

    details, no_data, upstream_short = [], [], []
    try:
        for tf in tfs:
            bs_freq = FREQ_MAP[tf]
            per = BARS_PER_DAY[tf]
            codes = [x["code"] for x in plan if x["tf"] == tf]
            for code in codes:
                new = fetch(code, bs_freq, args.start, args.end)
                if new.empty:
                    no_data.append({"code": code, "tf": tf})
                    continue
                old = pd.read_parquet(store._path(code, tf), columns=["time"] + FIELDS.split(",")[2:])
                lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(days=1)
                old_in = old[(old.time >= lo) & (old.time < hi)]
                both = old_in.merge(new, on="time", suffixes=("_l", "_n"))
                dev = 0.0
                if len(both):
                    dev = max(float(np.abs(both[k + "_l"] - both[k + "_n"]).max())
                              for k in ["open", "high", "low", "close", "volume", "amount"])
                if dev > 0:
                    raise AssertionError(
                        f"{code} {tf}: 重叠时间戳上游与本地读数不一致（max_abs_dev={dev:g}）"
                        "——复权基准已变，只追加会拼出两条基准；须整表重下，本脚本拒绝继续")

                add = new[~new.time.isin(old.time)].copy()
                # 上游自己缺的整日：该日不足整日根数
                cnt = new.time.dt.floor("D").value_counts()
                upstream_short += [{"code": code, "tf": tf, "day": str(d)[:10], "bars": int(n)}
                                   for d, n in cnt.items() if n < per]
                if add.empty:
                    continue
                before, after, overlap = bfd._atomic_merge(code, tf, to_store(code, add))
                details.append({"code": code, "tf": tf, "bars_added": after - before,
                                "bars_before": before, "bars_after": after,
                                "overlap": overlap, "days_added": int(cnt.sum() // per),
                                "new_max": str(pd.read_parquet(store._path(code, tf),
                                                               columns=["time"]).time.max())})
                print(f"  {tf:<4} {code:<10} +{after - before:>3} 根  {before} → {after}")
                time.sleep(0.3)
    finally:
        bs.logout()

    print(f"\n补入 {sum(d['bars_added'] for d in details)} 根，涉及 {len(details)} 个"
          f"(代码,周期)；无数据 {len(no_data)}；上游不足整日 {len(upstream_short)}")
    meta = {
        "source": "baostock_gapfill",
        "purpose": "按本地缺口补 A 股 bars；只追加，合并走 _atomic_merge 断言",
        "download_time": datetime.now().isoformat(),
        "range": [args.start, args.end],
        "timeframes": tfs,
        "details": details,
        "no_data": no_data,
        "upstream_short_days": upstream_short,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out = ROOT / "data" / "meta" / f"baostock_gapfill_{'-'.join(tfs)}_{datetime.now():%Y%m%dT%H%M}.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("元数据:", out)


if __name__ == "__main__":
    main()
