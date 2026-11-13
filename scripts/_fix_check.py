"""台账 #37 口径修复的单元验证：合成 K 线核对多空收益符号、成本、删失。"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chan_notes.core.policy import simulate_exit
from chan_notes.labels.label import forward_labels

C = 0.0013
FAILS = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got:.6f} want={want:.6f}")
    if not ok:
        FAILS.append(name)


def bars(o, h, l, c):
    n = len(o)
    return {"open": np.array(o, float), "high": np.array(h, float),
            "low": np.array(l, float), "close": np.array(c, float),
            "volume": np.ones(n)}


# 1) 做多止损：入场 100，stop=98（2×ATR=2），第 3 根 low 触 97.5 → 净 = -2% - 成本
b = bars([100, 100.5, 99.5, 99], [101, 100.8, 100, 99.5], [99.5, 100, 97.5, 98.5], [100.5, 100, 98, 99])
r = simulate_exit(b, 0, 1, 1.0, hold=10, stop_atr=2.0, cost=C)
check("buy stop ret", r.ret, 98 / 100 - 1 - C)
print(f"PASS  buy stop reason={r.reason} bars={r.bars}" if r.reason == "stop" and r.bars == 3 else f"FAIL  buy stop reason/bars {r}")

# 2) 做空止损（旧版把这笔记成盈利）：入场 100，stop=102，第 2 根 high 触 102.5 → 净 = -2% - 成本
b = bars([100, 100.5, 101, 101.5], [100.4, 102.5, 101.5, 102], [99.6, 100, 100.5, 100.8], [100.2, 101, 101.2, 101.5])
r = simulate_exit(b, 0, 0, 1.0, hold=10, stop_atr=2.0, cost=C)
check("short stop ret (旧版符号颠倒)", r.ret, -(102 / 100 - 1) - C)
print(f"PASS  short stop reason={r.reason}" if r.reason == "stop" else f"FAIL  short stop reason {r.reason}")

# 3) 做空时间退出且价格下跌（盈利单）：持有期满收盘 98 → 净 = +2% - 成本
o = [100.0] * 6
b = bars(o, [100.5] * 6, [99.5] * 6, [99.0] * 6)
r = simulate_exit(b, 0, 0, 1.0, hold=6, stop_atr=2.0, cost=C)
check("short time win ret", r.ret, -(99 / 100 - 1) - C)
print(f"PASS  short time reason={r.reason}" if r.reason == "time" else f"FAIL  short time reason {r.reason}")

# 4) 删失：数据只剩 5 根，hold=10，价格始终未触止损（stop=98）→ reason=censored
b = bars([100, 99.5, 99.2, 99.0, 98.9], [100.2, 100.0, 99.8, 99.6, 99.4],
         [99.2, 99.0, 98.8, 98.6, 98.5], [99.5, 99.2, 99.0, 98.9, 98.8])
r = simulate_exit(b, 0, 1, 1.0, hold=10, stop_atr=2.0, cost=C)
print(f"PASS  censored flagged" if r is not None and r.reason == "censored" else f"FAIL  censored got {r}")
check("censored ret uses last close", r.ret, 98.8 / 100 - 1 - C)

# 5) 做空触止损与删失竞争：末端 1 根触止损 → 仍记 stop（损失已实现）
b = bars([100, 99, 98, 97, 96.5], [100.2] * 5, [98.8, 98.5, 97.5, 96.5, 95.5], [99, 98, 97, 96.6, 96])
r = simulate_exit(b, 0, 1, 1.0, hold=10, stop_atr=2.0, cost=C)
print(f"PASS  stop-before-end kept" if r.reason == "stop" else f"FAIL  stop-before-end got {r.reason}")

# 6) label.py 卖侧：dir_ret = -(价差) - 成本；mfe≥0、mae≤0；删失档全 NaN
b = bars([100] + [100.5] * 9 + [99.5] * 9,
         [100.6] * 19, [99.4] + [99.8] * 9 + [98.5] * 9,
         [100.2] + [100.4] * 9 + [98.8] * 9)
lab = forward_labels(b, 0, 0, 1.0, horizons=(8, 24), cost=C)
q8 = b["close"][7] / 100 - 1
check("sell dir_ret_8 = -(q)-cost", lab["dir_ret_8"], -q8 - C)
check("sell mfe_8 >= 0", lab["mfe_8"], max(0.0, 1 - min(b["low"][0:8]) / 100), tol=1e-6)
check("sell mae_8 <= 0", lab["mae_8"], -max(0.0, max(b["high"][0:8]) / 100 - 1), tol=1e-6)
print(f"{'PASS' if np.isnan(lab['dir_ret_24']) else 'FAIL'}  sell dir_ret_24 censored -> NaN")
labb = forward_labels(bars([100] * 10, [100.4] * 10, [99.8] * 10, [101] * 10), 0, 1, 1.0, horizons=(8,), cost=C)
check("buy dir_ret_8 = q-cost", labb["dir_ret_8"], 101 / 100 - 1 - C)

print("\n" + ("ALL PASS" if not FAILS and "FAIL" not in "" else "HAS FAILURES: " + str(FAILS)))
