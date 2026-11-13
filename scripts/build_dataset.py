"""建数据集：遍历本地 30m 数据，抽事件 + 特征 + 标签，存 parquet。"""
import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from chan_notes.data import store
from chan_notes.features import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v1")
    ap.add_argument("--freq", default="30m")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", default="", help="逗号分隔；默认取该周期全部已缓存代码")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or store.list_codes(args.freq)
    if args.limit:
        codes = codes[: args.limit]
    print(f"codes={len(codes)}", flush=True)
    df = build.build(codes, args.freq, warmup=args.warmup)
    if df.empty:
        print("no events")
        return
    p = build.save(df, args.name)
    print(f"rows={len(df)} cols={df.shape[1]} -> {p}", flush=True)


if __name__ == "__main__":
    main()
