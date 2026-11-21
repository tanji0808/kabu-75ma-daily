"""合図の日（75日線を終値で上抜けた日）の終値と比べて、その後の株価がどう動いたかを集計する。"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402

t = pd.read_csv("backtest/trades_trail.csv", dtype={"code": str})
t = t[t["frank"] >= 6].copy()
prices = k.download_prices(sorted(t["code"].unique()), period="6y")
rows = []
for _, r in t.iterrows():
    df = prices.get(r["code"])
    if df is None:
        continue
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    d = pd.Timestamp(r["signal"])
    if d not in df.index:
        continue
    i = df.index.get_loc(d)
    c, o = df["Close"].to_numpy(float), df["Open"].to_numpy(float)
    m25 = df["Close"].rolling(25).mean()
    d1 = (m25 - df["Close"].rolling(25).std(ddof=0)).to_numpy()
    if i + 21 >= len(df):
        continue
    sc = c[i]
    width = o[i + 1] - d1[i]
    x = {"gap": (o[i + 1] / sc - 1) * 100}
    for n in (1, 5, 10, 20):
        x[f"below_{n}"] = c[i + n] < sc
        x[f"ret_{n}"] = (c[i + n] / sc - 1) * 100
    nxt = c[i + 1:i + 21]
    x["dip20"] = bool((nxt < sc).any())
    x["max_dd20"] = (nxt.min() / sc - 1) * 100
    # 下回ったあとに、利益確保ライン（翌日始値＋幅）まで戻したか（120営業日以内）
    after = c[i + 1:i + 121]
    first_dip = np.argmax(after < sc) if (after < sc).any() else None
    reach = after >= o[i + 1] + width
    x["reach_lock"] = bool(reach.any())
    x["dip_then_reach"] = bool(first_dip is not None and reach[first_dip:].any())
    x["reason"] = r["reason"]
    rows.append(x)
d = pd.DataFrame(rows)
pc = lambda s: round(float(s.mean() * 100), 1)
out = {
    "n": int(len(d)),
    "gap_up": pc(d["gap"] > 0), "gap_down": pc(d["gap"] < 0), "gap_avg": round(float(d["gap"].mean()), 2),
    "gap_median": round(float(d["gap"].median()), 2),
    "below": {n: pc(d[f"below_{n}"]) for n in (1, 5, 10, 20)},
    "ret_avg": {n: round(float(d[f"ret_{n}"].mean()), 2) for n in (1, 5, 10, 20)},
    "ret_median": {n: round(float(d[f"ret_{n}"].median()), 2) for n in (1, 5, 10, 20)},
    "dip20": pc(d["dip20"]),
    "max_dd20_median": round(float(d["max_dd20"].median()), 2),
    "reach_lock": pc(d["reach_lock"]),
    "dip_then_reach_of_dipped": pc(d.loc[d["dip20"], "dip_then_reach"]),
    "no_dip_reach": pc(d.loc[~d["dip20"], "reach_lock"]),
}
os.makedirs("backtest", exist_ok=True)
with open("backtest/signal_stats.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
