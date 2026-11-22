#!/usr/bin/env python3
"""
空売りの条件の組み合わせを、候補の表（backtest/short_candidates_11y.csv.gz）を使ってまとめて試す。

前半（2017〜2021年に売った取引）で条件を選び、後半（2022〜2026年）で確かめる。
同時に持つのは最大10銘柄・建玉1,000万円まで（short_backtest.portfolio と同じ）。往復0.1%の売買コストを引く。
出力 : backtest/short_grid2.json（全部の組み合わせの前半・後半・全体の成績）
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_backtest as sb  # noqa: E402

COST = 0.001
SPLIT = "2022-01-01"
GRID = {
    "walk": [5, 6, 7],
    "vol": [200_000, 500_000],
    "rel10": [None, 0, -3, -5],
    "rank": [4, 5, 6, None],
    "lsh": [None, 0.5, 1.0, 1.5],
    "lchg": [None, 0.2, 0.5],
    "rev": ["any", "no_up", "down"],
    "exit": ["m2", "lk25c"],
}


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"trades": 0, "after_cost": 0, "max_dd": 0, "win": None, "years": {}}
    d = pd.DataFrame(tr).sort_values("exit_date")
    net = d["pnl"] - d["notional"] * COST
    eq = net.cumsum()
    yrs = d.assign(net=net, y=d["exit_date"].str[:4]).groupby("y")["net"].sum()
    return {"trades": int(len(d)), "after_cost": int(net.sum()), "max_dd": int((eq - eq.cummax()).min()),
            "win": round(float((d["pnl"] > 0).mean() * 100), 1), "avg_ret": round(float(d["ret_pct"].mean()), 2),
            "years": {y: int(v) for y, v in yrs.items()}}


def main() -> None:
    c = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str})
    c = c[c["signal"] >= "2017-01-01"]
    res = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        m = (c["walk"] >= p["walk"]) & (c["vol"] >= p["vol"])
        if p["rel10"] is not None:
            m &= c["rel10"] <= p["rel10"]
        if p["rank"] is not None:
            m &= c["rank"].notna() & (c["rank"] <= p["rank"])
        if p["lsh"] is not None:
            m &= c["lsh"] >= p["lsh"]
        if p["lchg"] is not None:
            m &= c["lchg"] < p["lchg"]
        if p["rev"] == "no_up":
            m &= c["rev"] != 1
        elif p["rev"] == "down":
            m &= c["rev"] == -1
        s = c[m]
        ex = p["exit"]
        tr = [{"code": r.code, "entry_date": getattr(r, f"{ex}_entry_date"), "exit_date": getattr(r, f"{ex}_exit_date"),
               "notional": getattr(r, f"{ex}_notional"), "pnl": getattr(r, f"{ex}_pnl"), "ret_pct": getattr(r, f"{ex}_ret_pct"),
               "signal": r.signal} for r in s.itertuples()]
        taken = sb.portfolio(tr)
        a = [t for t in taken if t["signal"] < SPLIT]
        b = [t for t in taken if t["signal"] >= SPLIT]
        res.append({"params": p, "first": stats(a), "second": stats(b), "all": stats(taken)})
    json.dump(res, open("backtest/short_grid2.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)}通りを保存しました")


if __name__ == "__main__":
    main()
