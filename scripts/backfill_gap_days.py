"""按「内部空洞 + 末端尾巴」补 crypto bars：逐日拉 daily 归档，只追加不重写。

与 backfill_binance_daily.py 的区别：后者按给定日期区间对全部标的补齐，不知道哪里缺；
本脚本先从本地 parquet 自己找出缺口（两个时刻之间不足整日 bar 的日历日、以及末端之后的日子），
只对缺口发请求。合并仍走 backfill_binance_daily._atomic_merge 的 6 条断言。

用法:
    python scripts/backfill_gap_days.py --timeframes 15m,1h,4h --tail-end 2026-11-10 --dry-run
    python scripts/backfill_gap_days.py --timeframes 15m,1h,4h --tail-end 2026-11-10
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import backfill_binance_daily as bfd  # noqa: E402

BARS_PER_DAY = {"1d": 1, "15m": 96, "1h": 24, "4h": 6}
STEP = {"1d": pd.Timedelta(days=1), "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}

MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
_MONTHLY_CACHE: dict[tuple[str, str, str], pd.DataFrame | None] = {}


def _download_zip(url: str) -> bytes | None:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.urlopen(req, timeout=45).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _read_zip(payload: bytes) -> pd.DataFrame | None:
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        frames = []
        for name in zf.namelist():
            raw = zf.read(name)
            if not raw.strip():
                continue
            df = pd.read_csv(io.BytesIO(raw), header=None)
            if str(df.iloc[0, 0]) == "open_time":
                df = df.iloc[1:]
            frames.append(df)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw[0] = pd.to_numeric(raw[0], errors="coerce")  # 带表头的块会留成 object，_convert 只认数值
    return raw.dropna(subset=[0]).sort_values(0).drop_duplicates(subset=[0]).reset_index(drop=True)


def _monthly_day(symbol: str, tf: str, day: str) -> pd.DataFrame | None:
    """日包 404 时回退到月包，切出该日。月包按 (sym, tf, 月份) 缓存，一个月只下一次。"""
    key = (symbol, tf, day[:7])
    if key not in _MONTHLY_CACHE:
        stamp = day[:7].replace("-", "-")
        payload = _download_zip(f"{MONTHLY_BASE}/{symbol}/{tf}/{symbol}-{tf}-{stamp}.zip")
        _MONTHLY_CACHE[key] = None if payload is None else _read_zip(payload)
    month = _MONTHLY_CACHE[key]
    if month is None:
        return None
    t = pd.to_datetime(pd.to_numeric(month[0], errors="coerce"), unit="ms")
    lo, hi = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    mask = (t >= lo) & (t < hi)
    if not mask.any():
        return None
    return month.loc[mask.values].reset_index(drop=True)


def plan_targets(tf: str, tail_end: str | None) -> list[dict]:
    """[(symbol, tf, day)]：首末根之间不足整日的日历日，加末端之后到 tail_end 的日子。"""
    root = ROOT / "data" / "bars" / tf
    per_day = BARS_PER_DAY[tf]
    step = STEP[tf]
    out, listing_edges = [], []
    for p in sorted(root.glob("*USDT.parquet")):
        t = pd.read_parquet(p, columns=["time"]).time
        if not len(t):
            continue
        first, last = t.min(), t.max()
        t0 = t.to_numpy()
        # 内部空洞：严格在首末根之间、且该日不足整日的日子
        for day in pd.date_range(first.normalize(), last.normalize(), freq="D")[1:-1]:
            lo, hi = day, day + pd.Timedelta(days=1)
            n = int(((t0 >= lo.to_datetime64()) & (t0 < hi.to_datetime64())).sum())
            if n < per_day:
                out.append({"symbol": p.stem, "timeframe": tf, "day": day.strftime("%Y-%m-%d"), "kind": "hole"})
        # 末端：last 之后的整根到 tail_end
        if tail_end:
            nxt = last + step
            while nxt.normalize() <= pd.Timestamp(tail_end):
                out.append({"symbol": p.stem, "timeframe": tf, "day": nxt.normalize().strftime("%Y-%m-%d"), "kind": "tail"})
                nxt += pd.Timedelta(days=1)
        listing_edges.append((p.stem, str(first), str(last)))
    return out, listing_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", default="15m,1h,4h")
    ap.add_argument("--tail-end", default="", help="补到这一天（含）；留空只补内部空洞")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tfs = [x.strip() for x in args.timeframes.split(",") if x.strip()]
    tail_end = args.tail_end.strip() or None
    t0 = time.time()

    targets, edges = [], []
    for tf in tfs:
        got, ed = plan_targets(tf, tail_end)
        targets += got
        edges += [{"timeframe": tf, "symbol": s, "first": f, "last": l} for s, f, l in ed]
        print(f"{tf}: 计划 {len(got)} 个 (币,日) —— 空洞 {sum(1 for x in got if x['kind']=='hole')} / 尾巴 {sum(1 for x in got if x['kind']=='tail')}")

    if args.dry_run:
        by_day = pd.Series([x["day"] for x in targets]).value_counts().sort_index()
        print("\n按日期汇总:")
        print(by_day.to_string())
        print(f"\n合计 {len(targets)} 次请求（dry-run，未联网）")
        return

    def fetch(item):
        import urllib.error
        last = ""
        for attempt in range(5):
            try:
                df = bfd._fetch_day(item["symbol"], item["timeframe"], item["day"])
                used = "daily"
                if df is None:  # 日包 404：月包常常是齐的，回退切日
                    df = _monthly_day(item["symbol"], item["timeframe"], item["day"])
                    used = "monthly"
                return {**item, "rows": 0 if df is None else len(df), "ok": df is not None,
                        "df": df, "source": used}
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {**item, "rows": 0, "ok": False, "df": None, "source": "daily"}
                last = f"HTTP {e.code}"
            except Exception as e:  # 超时/连接重置：重试，别学 _download_month 静默吞掉
                last = type(e).__name__
            time.sleep(1.5 * (attempt + 1))
        return {**item, "rows": 0, "ok": False, "df": None, "error": last, "source": "daily"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = list(pool.map(fetch, targets))

    missing = [{"symbol": d["symbol"], "timeframe": d["timeframe"], "day": d["day"],
                "kind": d["kind"], "error": d.get("error", "日包与月包都没有该日")}
               for d in done if not d["ok"]]
    via_month = [{"symbol": d["symbol"], "timeframe": d["timeframe"], "day": d["day"]}
                 for d in done if d["ok"] and d.get("source") == "monthly"]
    print(f"\n取回 {sum(1 for d in done if d['ok'])} / {len(done)}，失败 {len(missing)}"
          f"（其中重试后仍报错 {sum(1 for m in missing if 'HTTP' in m['error'] or m['error'].endswith('Error'))}）"
          f"，月包回退 {len(via_month)} 个")

    # 每个 (symbol, tf) 一次合并
    groups: dict[tuple[str, str], list] = {}
    for d in done:
        if d["ok"] and d["rows"]:
            groups.setdefault((d["symbol"], d["timeframe"]), []).append(d["df"])

    details = []
    for (sym, tf), frames in sorted(groups.items()):
        new = bfd._convert(pd.concat(frames, ignore_index=True), sym)
        before, after, overlap = bfd._atomic_merge(sym, tf, new)
        last = pd.read_parquet(bfd.store._path(sym, tf), columns=["time"])["time"].max()
        details.append({"symbol": sym, "timeframe": tf, "days_filled": len(frames),
                        "bars_before": before, "bars_after": after, "bars_added": after - before,
                        "bars_overlap": overlap, "new_max_time": str(last)})
        print(f"  {tf:<4} {sym:<14} 补 {len(frames):>3} 日  {before:>7} → {after:>7} (+{after-before})  末条 {last}")

    meta = {
        "source": "binance_futures_daily_gapfill",
        "purpose": "按本地缺口逐日拉 daily 归档补齐；合并走 _atomic_merge 断言，只追加",
        "download_time": datetime.now().isoformat(),
        "timeframes": tfs,
        "tail_end": tail_end,
        "planned_requests": len(targets),
        "upstream_404": missing,
        "monthly_fallback": via_month,
        "details": details,
        "listing_edges": edges,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    stamp = datetime.now().strftime("%Y%m%dT%H%M")
    out = ROOT / "data" / "meta" / f"binance_gapfill_{'-'.join(tfs)}_{stamp}.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n新增 {sum(d['bars_added'] for d in details)} 根，涉及 {len(details)} 个(币,周期)；"
          f"耗时 {meta['elapsed_seconds']}s\n元数据: {out}")


if __name__ == "__main__":
    main()
