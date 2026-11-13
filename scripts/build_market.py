"""构建等权市场指数序列（篮子内全部标的收盘价归一化后取中位数），用于市场环境特征。"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.data import store


def build_market(freq: str = "30m", min_stocks: int = 30, out_code: str = "_MARKET",
                 codes: list[str] | None = None, min_frac: float | None = None,
                 dry_run: bool = False) -> pd.DataFrame:
    # 指数的标的池必须和用它算 mkt_* 特征的事件集同源，否则中位数会混进另一个市场
    if codes is None:
        codes = [c for c in store.list_codes(freq) if not c.startswith("_")]
    if not codes:
        raise ValueError(f"{freq} 篮子为空，无法构建指数")
    frames = []
    for c in codes:
        df = store.load(c, freq)[["time", "close"]].copy()
        df = df.rename(columns={"close": c})
        frames.append(df.set_index("time"))
    px = pd.concat(frames, axis=1).sort_index()
    px = px.ffill(limit=8)
    norm = px / px.bfill().iloc[0]
    level = norm.median(axis=1)
    cnt = px.notna().sum(axis=1)
    if min_frac is not None:
        if not 0 < min_frac <= 1:
            raise ValueError(f"min_frac 须在 (0, 1]，收到 {min_frac}")
        # 篮子随时间增长（新币陆续上市）时按已上市成员的比例设阈，绝对值会抹掉早期整段
        listed = px.notna().cummax().sum(axis=1)
        keep = cnt >= np.ceil(listed * min_frac)
        thresh_desc = f"min_frac={min_frac}"
    else:
        if min_stocks > len(codes):
            raise ValueError(
                f"min_stocks={min_stocks} 大于篮子大小 {len(codes)}，会筛掉全部时刻；"
                f"请调小 --min-stocks，或改用 --min-frac 按已上市成员比例设阈"
            )
        keep = cnt >= min_stocks
        thresh_desc = f"min_stocks={min_stocks}"
    level = level[keep]
    if level.empty:
        raise ValueError(
            f"{freq} 篮子 {len(codes)} 个标的、{thresh_desc}：筛掉了全部 {len(cnt)} 个时刻"
            f"（最多同时在场 {int(cnt.max())} 个），拒绝写空指数"
        )
    ret = level.pct_change().fillna(0)
    close = 1000 * (1 + ret).cumprod()
    op = close.shift(1).fillna(close.iloc[0])
    hi = close * (1 + ret.abs().rolling(3).mean().fillna(0).values * 0.3)
    lo = close * (1 - ret.abs().rolling(3).mean().fillna(0).values * 0.3)
    # 合成序列必须满足 high>=max(open,close)、low<=min(open,close)，否则过不了框架校验
    hi = np.maximum(hi, np.maximum(op, close))
    lo = np.minimum(lo, np.minimum(op, close))
    df = pd.DataFrame({
        "time": close.index,
        "open": op.values,
        "high": hi.values,
        "low": lo.values,
        "close": close.values,
        "volume": 0.0,
    })
    if not dry_run:
        store.save(out_code, df, freq)
    print(f"[{freq}] 篮子={len(codes)} {thresh_desc} 保留={len(df)} "
          f"丢弃={int((~keep).sum())} 起点={df['time'].min()} 终点={df['time'].max()} "
          f"{'dry-run 未写入' if dry_run else '已写入 ' + out_code}", flush=True)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="30m")
    ap.add_argument("--min-stocks", type=int, default=30,
                    help="绝对在场数阈值（默认 30，A 股时代口径）；篮子小于它时会硬失败")
    ap.add_argument("--min-frac", type=float, default=None,
                    help="改用相对阈值：每个时刻至少 ceil(已上市成员数 × 该比例) 在场；给了它就忽略 --min-stocks")
    ap.add_argument("--codes", default="", help="逗号分隔；默认取该周期全部非 _ 代码")
    ap.add_argument("--dry-run", action="store_true", help="只算不写，用于核对口径而不覆盖现有指数")
    args = ap.parse_args()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    build_market(args.freq, min_stocks=args.min_stocks, codes=codes,
                 min_frac=args.min_frac, dry_run=args.dry_run)
