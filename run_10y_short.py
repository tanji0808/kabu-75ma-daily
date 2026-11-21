"""過去10年の空売り版バックテスト → 資金シミュレーション → 買い（採用ルール）との組み合わせ。結果は backtest10s/ に保存。"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

os.environ.update(K_ADJUST="1", BT_PERIOD="11y", BT_OUT="backtest10s", BT_VARIANTS="short_trail_calm,short_trail")
import backtest  # noqa: E402

backtest.main()
runs = []
for v in ("short_trail_calm", "short_trail"):
    for lo, hi, tag in ((0, 10, "all"), (0, 5, "rank0-5"), (6, 10, "rank6")):
        out = f"backtest10s/portfolio_{v}_{tag}.json"
        env = dict(os.environ, P2_SIDE="short", P2_TRADES=f"backtest10s/trades_{v}.csv", P2_PERIOD="11y",
                   P2_MIN_RANK=str(lo), P2_MAX_RANK=str(hi), P2_SPLIT="2021-01-01", P2_OUT=out)
        subprocess.run([sys.executable, "portfolio2.py"], env=env, check=True)
        runs.append(out)
# 買い（採用ルール・ランク6以上）の毎日の資産（配当調整なしの株価で、これまでと同じ条件）
env = {k_: v_ for k_, v_ in os.environ.items() if k_ != "K_ADJUST"}
env.update(P2_TRADES="backtest10/trades_trail_calm5_rsi60.csv", P2_PERIOD="11y", P2_MIN_RANK="6",
           P2_SPLIT="2021-01-01", P2_OUT="backtest10s/portfolio_long_rank6.json")
subprocess.run([sys.executable, "portfolio2.py"], env=env, check=True)

# 組み合わせ：買いに1,000万円＋空売りに1,000万円（それぞれ別に運用した損益を足す）
long = json.load(open("backtest10s/portfolio_long_rank6.json", encoding="utf-8"))
L = {r["name"]: r for r in long["results"]}
long_eq = pd.Series(next(r for n, r in L.items() if n.startswith("⑥＋③ TOPIX"))["daily"])
combos = {}
for path in runs:
    sh = json.load(open(path, encoding="utf-8"))
    for r in sh["results"]:
        short_eq = pd.Series(r["daily"])
        idx = sorted(set(long_eq.index) & set(short_eq.index))
        le, se = long_eq[idx] - 10_000_000, short_eq[idx] - 10_000_000
        comb = 20_000_000 + le + se
        dd = (comb.cummax() - comb)
        yl = le.groupby(pd.to_datetime(le.index).year).last().diff().fillna(le.groupby(pd.to_datetime(le.index).year).last())
        ys = se.groupby(pd.to_datetime(se.index).year).last().diff().fillna(se.groupby(pd.to_datetime(se.index).year).last())
        dl, ds = le.diff().dropna(), se.diff().dropna()
        combos[f"{os.path.basename(path)}|{r['name']}"] = {
            "combined_pnl": round(float(le.iloc[-1] + se.iloc[-1])), "combined_max_dd": round(float(dd.max())),
            "combined_max_dd_pct": round(float((dd / comb.cummax()).max() * 100), 1),
            "long_pnl": round(float(le.iloc[-1])), "short_pnl": round(float(se.iloc[-1])),
            "daily_corr": round(float(np.corrcoef(dl, ds)[0, 1]), 2),
            "years": {str(y): {"long": round(float(yl[y])), "short": round(float(ys[y])), "both": round(float(yl[y] + ys[y]))} for y in yl.index},
        }
with open("backtest10s/combined.json", "w", encoding="utf-8") as f:
    json.dump(combos, f, ensure_ascii=False, indent=1)
print("組み合わせ", len(combos))
