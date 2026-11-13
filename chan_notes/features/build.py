"""把事件回放 + 标签组装成训练/回测用的数据集。"""
import pandas as pd

from chan_notes.config import EVENTS
from chan_notes.core import replay
from chan_notes.data import store
from chan_notes.features import indicators, schema
from chan_notes.features.market import MarketContext
from chan_notes.core.policy import POLICY_KEYS, policy_labels
from chan_notes.labels.label import FORWARD_KEYS, forward_labels
from chan_notes.labels.confirm import confirmation_entry


def build(codes, freq: str = "30m", warmup: int = 200, verbose: bool = True,
          with_market: bool = True) -> pd.DataFrame:
    mkt = None
    if with_market and store.has("_MARKET", freq):
        mkt = MarketContext(store.load("_MARKET", freq))
    rows, feats_cols, stats = [], set(), []
    failed = []
    for code in codes:
        if code.startswith("_") or not store.has(code, freq):
            continue
        try:
            events, elapsed = replay.replay(code, freq, warmup=warmup)
        except Exception as e:
            failed.append((code, f"{type(e).__name__}: {e}"))
            print(f"  {code}: FAIL {type(e).__name__}: {e}", flush=True)
            continue
        if not events:
            continue
        df = store.load(code, freq)
        ind = indicators.precompute(df)
        for e in events:
            r = {
                "code": code, "freq": freq, "bar_idx": e.bar_idx, "event_time": e.event_time,
                "bsp_bi_idx": e.bsp_bi_idx, "bsp_type": e.bsp_type, "is_buy": e.is_buy,
                "bsp_klu_idx": e.bsp_klu_idx, "delay_bars": e.delay_bars,
                "entry_idx": e.entry_idx, "entry_time": e.entry_time, "entry_price": e.entry_price,
                "clear_bar": e.clear_bar,
                "atr": float(ind["atr14"][e.bar_idx]) if ind["atr14"][e.bar_idx] == ind["atr14"][e.bar_idx] else 0.0,
            }
            r.update(e.features)
            if mkt is not None:
                mk = mkt.features_at(e.event_time)
                r.update(mk)
                feats_cols.update(mk)
            r.update(forward_labels(ind, e.entry_idx, e.is_buy, r["atr"]))
            r.update(policy_labels(ind, e.entry_idx, e.is_buy, r["atr"]))
            ok, cidx, wait = confirmation_entry(ind, e.bar_idx, e.is_buy, max_wait=4)
            r["conf_ok"] = 1.0 if ok else 0.0
            r["conf_wait"] = float(wait)
            r["conf_entry_idx"] = float(cidx)
            if ok:
                r["conf_entry_time"] = str(df["time"].iloc[cidx])
                r["conf_entry_price"] = float(ind["open"][cidx])
                r["conf_atr"] = r["atr"]
                r.update({f"c_{k}": v for k, v in policy_labels(ind, cidx, e.is_buy, r["atr"]).items()})
                r.update({f"c_{k}": v for k, v in forward_labels(ind, cidx, e.is_buy, r["atr"]).items()})
            else:
                r["conf_entry_time"] = ""
                r["conf_entry_price"] = float("nan")
                r["conf_atr"] = float("nan")
                r.update({f"c_{k}": float("nan") for k in POLICY_KEYS})
                r.update({f"c_{k}": float("nan") for k in FORWARD_KEYS})
            rows.append(r)
            feats_cols.update(e.features)
        stats.append((code, len(events), elapsed))
        if verbose:
            print(f"  {code}: events={len(events)} {elapsed:.1f}s", flush=True)

    if not rows:
        return pd.DataFrame()

    cols = schema.ensure(feats_cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df[cols] = df[cols].fillna(0.0)
    return df


def save(df: pd.DataFrame, name: str):
    p = EVENTS / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p
