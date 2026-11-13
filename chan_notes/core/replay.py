"""单只股票的事件回放：逐 bar 喂入，首次出现 BSP 即记录并抽特征。"""
import time
from dataclasses import dataclass, asdict

import numpy as np

from chan_notes.config import chan_config
from chan_notes.data import store
from chan_notes.features import indicators, pa, chan_ctx, patterns, brooks, dynamics, context


@dataclass
class Event:
    code: str
    freq: str
    bar_idx: int          # 发现该 BSP 的 bar（决策 bar）
    event_time: str
    bsp_bi_idx: int
    bsp_type: str
    is_buy: int
    bsp_klu_idx: int      # BSP 所锚定的 K 线
    delay_bars: int       # 决策 bar 与锚定 K 线的距离
    entry_idx: int        # 入场 bar（决策 bar 的下一根）
    entry_time: str
    entry_price: float    # 下一根 bar 开盘价
    features: dict
    clear_bar: int = -1   # BSP 从 bs_point_lst 消失的 bar；-1=回放结束仍存活


def _log(msg: str):
    print(msg, flush=True)


def replay(code: str, freq: str = "30m", max_bsp_lookback: int = 8,
           warmup: int = 200, verbose: bool = False, df=None):
    from Chan import CChan
    from Common.CEnum import AUTYPE, KL_TYPE

    if df is None:
        df = store.load(code, freq)
    ind = indicators.precompute(df)
    times = df["time"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    opens = ind["open"]

    lv_map = {"30m": KL_TYPE.K_30M, "d": KL_TYPE.K_DAY, "day": KL_TYPE.K_DAY,
              "5m": KL_TYPE.K_5M, "15m": KL_TYPE.K_15M,
              "60m": KL_TYPE.K_60M, "1h": KL_TYPE.K_60M}
    chan = CChan(code=code, data_src="custom:LocalStore.CLocalStore",
                 lv_list=[lv_map[freq]], config=chan_config(trigger_step=True),
                 autype=AUTYPE.QFQ)

    seen = set()
    events = []
    alive: dict[int, list[int]] = {}  # bi_idx -> 该笔上已记录事件的索引（存活追踪）
    n_bar = 0
    skipped = 0
    t0 = time.time()

    for snap in chan.step_load():
        lv = snap[0]
        klu = lv[-1][-1]
        t = klu.idx
        n_bar += 1
        if t < warmup:
            continue

        # 存活追踪：bi 级 BSP 从 bsp_store_flat_dict 消失 = 失效（clear_store_end 只删未确认段）
        if alive:
            try:
                current = {b.bi.idx for b in lv.bs_point_lst.bsp_store_flat_dict.values()}
            except Exception:
                current = None
            if current is not None:
                for bi_idx in [k for k in alive if k not in current]:
                    for ei in alive.pop(bi_idx):
                        events[ei].clear_bar = t

        try:
            latest = lv.bs_point_lst.get_latest_bsp(max_bsp_lookback)
        except Exception:
            continue
        if not latest:
            continue

        for bsp in latest:
            key = (bsp.bi.idx, bsp.type2str())
            if key in seen:
                continue
            seen.add(key)
            if t + 1 >= len(df):
                continue
            feats = {}
            try:
                feats.update(pa.price_action_features(ind, t))
                feats.update(patterns.pattern_features(ind, t))
                feats.update(brooks.swing_features(ind, t))
                feats.update(brooks.regime_features(ind, t))
                feats.update(brooks.bar_count_features(ind, t))
                feats.update(brooks.breakout_strength_features(ind, t))
                feats.update(brooks.pattern2_features(ind, t))
                feats.update(chan_ctx.structure_features(lv, klu, bsp))
                feats.update(dynamics.dynamics_features(lv, bsp))
                feats.update(context.context_features(ind, t))
            except Exception as e:
                skipped += 1
                if verbose:
                    _log(f"  [{code}] bar={t} feature skip: {type(e).__name__}: {e}")
                continue
            ev = Event(
                code=code, freq=freq, bar_idx=t, event_time=times[t],
                bsp_bi_idx=bsp.bi.idx, bsp_type=bsp.type2str(), is_buy=1 if bsp.is_buy else 0,
                bsp_klu_idx=bsp.klu.idx, delay_bars=t - bsp.klu.idx,
                entry_idx=t + 1, entry_time=times[t + 1], entry_price=float(opens[t + 1]),
                features=feats,
            )
            events.append(ev)
            alive.setdefault(bsp.bi.idx, []).append(len(events) - 1)

    elapsed = time.time() - t0
    if verbose:
        _log(f"[{code}] bars={n_bar} events={len(events)} skipped={skipped} elapsed={elapsed:.1f}s")
    return events, elapsed
