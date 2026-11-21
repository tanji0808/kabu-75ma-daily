#!/usr/bin/env python3
"""
買い（採用ルール：急騰除外・ランク6以上・引き上げ式ロスカット）を、その時点の決算でランクを付け直して確かめる。

材料 : backtest/trades_trail_calm5_rsi60.csv（ランクで絞る前の全取引。frank = 今の財務データでのランク）
やること : J-Quants の決算データがそろっている期間の取引について、合図の日までに公表されていた決算で
          ランクを計算し直し（frank_pit）、「今のデータでランク6以上」と「その時点のデータでランク6以上」を比べる。
          1回100万円で全部の取引を受けたとして、損益も出す（資金の制約なし）。
ランクの作り方・期間は short_rel_pit.py と同じ（負債比率の0.5点は使わない）。
出力 : backtest/buy_pit.json, backtest/buy_pit_trades.csv
"""

from __future__ import annotations

import bisect
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_rel_pit as sp  # noqa: E402

TRADES = os.environ.get("BUY_TRADES", "backtest/trades_trail_calm5_rsi60.csv")
MIN_RANK = 6
LOT = 1_000_000


def group(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"trades": 0}
    yen = d["ret"] / 100 * LOT
    return {"trades": int(len(d)), "win_rate": round(float((d["ret"] > 0).mean() * 100), 1),
            "avg_ret_pct": round(float(d["ret"].mean()), 2), "pnl_yen_100man_each": int(yen.sum()),
            "avg_excess_pct": round(float(d["excess"].mean()), 2) if "excess" in d else None}


def main() -> None:
    fins = sp.load_fins()
    disc = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    master = k.load_prime_list()
    t = pd.read_csv(TRADES, dtype={"code": str})
    all_d = sorted({r["DiscDate"] for v in fins.values() for r in v})
    last_disc = all_d[-1]
    codes = master["コード"].tolist()
    w0 = None
    for d in all_d[::5]:
        have = sum(1 for c in codes if any(r["CurPerType"] == "FY" and r["DiscDate"] <= d for r in fins.get(c, [])))
        if have >= 0.8 * len(codes):
            w0 = d
            break
    t = t[(t["signal"] >= w0) & (t["signal"] <= last_disc)].copy()
    print(f"期間 {w0} 〜 {last_disc}：買いの取引 {len(t)}件")
    prices = k.download_prices(codes, period="3y")
    clean = {c: sb.clean(df) for c, df in prices.items()}
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items() if len(d)})
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    pit = {}
    for sd in sorted(t["signal"].unique()):
        ts = pd.Timestamp(sd)
        if ts not in closes.index:
            continue
        fund = {}
        for c in clean:
            n = bisect.bisect_right(disc.get(c, []), sd)
            if n:
                fund[c] = sp.pit_fund(fins[c][:n])
        cm = {c: float(x) for c, x in closes.loc[ts].items() if x == x}
        pit[sd] = k.fund_ranks(master, clean, fund, close_map=cm)
    t["frank_pit"] = [pit.get(s, {}).get(c) for s, c in zip(t["signal"], t["code"])]
    now6 = t["frank"].fillna(-1) >= MIN_RANK
    pit6 = t["frank_pit"].fillna(-1) >= MIN_RANK
    res = {"all": group(t), "now_rank6": group(t[now6]), "pit_rank6": group(t[pit6]),
           "both": group(t[now6 & pit6]), "now_only": group(t[now6 & ~pit6]), "pit_only": group(t[~now6 & pit6]),
           "pit_rank_missing": int(t["frank_pit"].isna().sum())}
    for kname, v in res.items():
        print(kname, v)
    t.to_csv("backtest/buy_pit_trades.csv", index=False, encoding="utf-8-sig")
    with open("backtest/buy_pit.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "window": [w0, last_disc],
                   "min_rank": MIN_RANK, "results": res}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
