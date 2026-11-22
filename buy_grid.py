#!/usr/bin/env python3
"""
買いの条件・売り方の組み合わせを、候補の表（backtest/buy_candidates_11y.csv.gz）でまとめて試す。
前半（2017〜2021年に合図）で選び、後半（2022〜2026年）で確かめる。
同時10銘柄・1,000万円まで。同じ日に重なったら増益率の高い順。金額はランクで変える（6:50万・7:100万・8:150万・9〜10:200万）。
往復0.1%の売買コストを引く。
出力 : backtest/buy_grid.json
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

COST = 0.001
SPLIT = "2022-01-01"
GRID = {
    "calm": ["cur", "none"],                 # cur = 5日+3%以下＆RSI60以下（今の急騰除外）
    "rank": [6, 7],
    "tpx": [None, "ma75", "ma200"],          # TOPIXが75日線・200日線より上のときだけ
    "rel": [None, 0, 3],                     # 20日でTOPIXより何%以上強いか
    "vratio": [None, 1.5],                   # 抜けた日の出来高が20日平均の何倍以上か
    "earn": [None, 30],                      # 直近の決算発表から何日以内か
    "exit": ["cur", "t5", "t10", "half", "ma25", "low10", "t10half"],
}
LOT = {6: 0.5, 7: 1.0, 8: 1.5}


def portfolio(rows: list[dict]) -> list[dict]:
    taken, open_ = [], []
    for t in sorted(rows, key=lambda x: (x["entry_date"], -x["eg"])):
        open_ = [x for x in open_ if x["exit_date"] > t["entry_date"]]
        if len(open_) >= 10 or sum(x["notional"] for x in open_) + t["notional"] > 1e7 or any(x["code"] == t["code"] for x in open_):
            continue
        open_.append(t)
        taken.append(t)
    return taken


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"trades": 0, "net": 0, "dd": 0, "years": {}}
    d = pd.DataFrame(tr).sort_values("exit_date")
    net = d["pnl"] - d["notional"] * COST
    eq = net.cumsum()
    yrs = d.assign(net=net).groupby(d["exit_date"].str[:4])["net"].sum()
    return {"trades": int(len(d)), "net": int(net.sum()), "dd": int((eq - eq.cummax()).min()),
            "win": round(float((d["pnl"] > 0).mean() * 100), 1), "years": {y: int(v) for y, v in yrs.items()}}


def main() -> None:
    c = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str})
    c = c[(c["signal"] >= "2017-01-01") & c["rank"].notna()].copy()
    c["eg"] = c["eg"].fillna(-9)
    res = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        m = c["rank"] >= p["rank"]
        if p["calm"] == "cur":
            m &= (c["up5"] <= 3) & (c["rsi"] <= 60)
        if p["tpx"]:
            m &= c[f"tpx_{p['tpx']}"].astype(bool)
        if p["rel"] is not None:
            m &= c["rel20"] >= p["rel"]
        if p["vratio"]:
            m &= c["vratio"] >= p["vratio"]
        if p["earn"]:
            m &= c["dsince"] <= p["earn"]
        ex = p["exit"]
        s = c[m & c[f"{ex}_ret"].notna()]
        rows = []
        for x in s.itertuples():
            w = LOT.get(int(x.rank), 2.0)
            r = getattr(x, f"{ex}_ret")
            rows.append({"code": x.code, "entry_date": x.entry, "exit_date": getattr(x, f"{ex}_exit"),
                         "notional": 1e6 * w, "pnl": 1e4 * r * w, "eg": x.eg, "signal": x.signal})
        taken = portfolio(rows)
        res.append({"params": p, "first": stats([t for t in taken if t["signal"] < SPLIT]),
                    "second": stats([t for t in taken if t["signal"] >= SPLIT]), "all": stats(taken)})
    json.dump(res, open("backtest/buy_grid.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)}通りを保存しました")


if __name__ == "__main__":
    main()
