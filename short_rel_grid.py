#!/usr/bin/env python3
"""
空売り「TOPIXより弱い銘柄だけ」の条件の数字を変えても勝てるかを確かめる（過去約5年・東証プライム）

共通 : バンドウォーク崩れ（10日のうち7日以上+1σより上 → 終値で+1σ割れ）の日に、翌営業日の始値で空売り。出来高50万株以上。
変える数字
  L   : 何営業日の値動きで比べるか（5・10・20日）
  G   : TOPIXより何％以上弱いか（3・5・7％）
  mk  : TOPIXがその期間に下げていないことを条件にするか（あり／なし）
  f4  : ファンダメンタルランク4以下を条件にするか（あり／なし）
買い戻し : m2（−2σ割れで利確）と lk25c（25日線逆指値＋−2σ割れで利確）。損切りはどちらも+1σ超え。
資金・費用・注意点は short_bw_backtest.py と同じ。往復0.1%の売買コストを引いた収支も出す。
出力 : backtest/short_rel_grid.json, backtest/short_rel_trades_<設定>.csv（SAVE の設定だけ）
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_bw_backtest as bw  # noqa: E402
import short_bw2_backtest as b2  # noqa: E402

PERIOD = os.environ.get("BT_PERIOD", "6y")
OUT = "backtest"
COST = 0.001
LS, GS, MKS, FS, EXS = (5, 10, 20), (3, 5, 7), (True, False), (True, False), ("m2", "lk25c")
SAVE = {"L10_G5_f4__m2", "L10_G5_f4__lk25c"}   # スクリーナーの「★ バンドウォーク崩れ×TOPIXより弱い」と同じ設定


def key(L, G, mk, f4, ex):
    return f"L{L}_G{G}{'_mk' if mk else ''}{'_f4' if f4 else ''}__{ex}"


def summarize(taken: list[dict]) -> dict:
    st = sb.stats(taken)
    if not taken:
        return st
    d = pd.DataFrame(taken)
    d["net"] = d["pnl"] - d["notional"] * COST
    d["y"] = d["exit_date"].str[:4]
    by = {y: int(g["net"].sum()) for y, g in d.groupby("y")}
    return {"trades": st["trades"], "total_pnl": st["total_pnl"], "after_cost": int(d["net"].sum()),
            "win_rate": st["win_rate"], "profit_factor": st["profit_factor"], "max_drawdown": st["max_drawdown"],
            "avg_ret_pct": round(float(d["ret_pct"].mean()), 2), "by_year_after_cost": by,
            "plus_years": sum(v > 0 for v in by.values()), "years": len(by),
            "after_cost_excl_top10": int(d["net"].sum() - d["pnl"].nlargest(10).sum())}


def main() -> None:
    import yfinance as yf
    topix = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    fpath = "data/fundamentals.json"
    fund = json.load(open(fpath, encoding="utf-8")) if os.path.exists(fpath) else {}
    os.makedirs(OUT, exist_ok=True)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) >= bw.WARMUP + 20:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()})
    sig = {}
    for code, df in clean.items():
        v = df["Volume"].to_numpy(float)
        sig[code] = [i for i in bw.find_signals(df)["BW"] if v[i] >= bw.MIN_VOLUME]
    # 合図の日ごとに、各期間の「銘柄の騰落率 − TOPIXの騰落率」とTOPIXの騰落率を計算しておく
    info = {}
    for code, df in clean.items():
        c = df["Close"].to_numpy(float)
        t = topix.reindex(df.index).ffill().to_numpy()
        rows = []
        for i in sig[code]:
            r = {}
            for L in LS:
                if i >= L and t[i - L] == t[i - L] and t[i] == t[i]:
                    mr = t[i] / t[i - L] - 1
                    r[L] = (c[i] / c[i - L] - 1 - mr, mr)
            rows.append((i, r))
        info[code] = rows
    need = sorted({clean[cd].index[i] for cd, rows in info.items() for i, r in rows
                   if any(r.get(L, (0, 0))[0] <= -0.03 for L in LS)})
    rank_on = {}
    for d in need:
        if d in closes.index:
            cm = {c: float(v) for c, v in closes.loc[d].items() if v == v}
            rank_on[d] = k.fund_ranks(master, clean, fund, close_map=cm)
    print(f"ファンダメンタルランクを計算した日数: {len(rank_on)}")
    raw = {key(*x): [] for x in product(LS, GS, MKS, FS, EXS)}
    for code, df in clean.items():
        cache: dict = {}
        sim = {}
        for L, G, mk, f4, ex in product(LS, GS, MKS, FS, EXS):
            busy = ""
            out = raw[key(L, G, mk, f4, ex)]
            for i, r in info[code]:
                if L not in r:
                    continue
                rel, mr = r[L]
                if rel > -G / 100 or (mk and mr < 0):
                    continue
                if f4:
                    rk = rank_on.get(df.index[i], {}).get(code)
                    if rk is None or rk > 4:
                        continue
                sd = df.index[i].strftime("%Y-%m-%d")
                if sd <= busy:
                    continue
                if (i, ex) not in sim:
                    sim[(i, ex)] = b2.simulate(code, df, i, ex, cache)
                t = sim[(i, ex)]
                out.append(t)
                busy = t["exit_date"]
    res = {}
    for kname, tr in raw.items():
        taken = sb.portfolio(tr)
        res[kname] = summarize(taken)
        if kname in SAVE:   # サイトのプリセットと同じ設定は、取引の一覧も保存する
            pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_rel_trades_{kname}.csv"), index=False, encoding="utf-8-sig")
        s = res[kname]
        print(f"{kname:24s} {s.get('trades', 0):5d}回 費用後 {s.get('after_cost', 0):>11,}円 "
              f"プラスの年 {s.get('plus_years')}/{s.get('years')} 落ち込み {s.get('max_drawdown', 0):>10,}円")
    with open(os.path.join(OUT, "short_rel_grid.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "cost_pct": COST * 100, "results": res},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
