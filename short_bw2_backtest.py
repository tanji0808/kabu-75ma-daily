#!/usr/bin/env python3
"""
空売り バンドウォーク崩れ＋ファンダメンタル4以下 の改良案をまとめて比較する（過去約5年・東証プライム）

土台 : short_bw_backtest.py の「BW_f4」（バンドウォーク崩れ・+1σ割れで空売り・ファンダメンタル4以下）

利確のしかた（損切りはどれも「終値が+1σを上回ったら翌日の始値で買い戻し」）
  m1   : 終値が −1σ を下回ったら翌日の始値で買い戻す（今のルール）
  m2   : 終値が −2σ を下回ったら翌日の始値で買い戻す
  ma25 : いったん終値が25日線を下回ったあと、終値が25日線を上回ったら翌日の始値で買い戻す
  lock : 終値が −1σ を下回ったら、翌日から −1σ に買い戻しの逆指値、−2σ に買い戻しの指値を置く
         （注文の値段は前日の終値で計算した −1σ・−2σ。寄り付きで飛び越えたら始値で約定。
           同じ日に両方に届いたら −1σ の逆指値が先に約定したとみなす＝厳しめ）
  lockt: lock と同じだが、安値が −1σ に届いた時点で注文を置く
  どれも最長60営業日。

絞り込み（合図の日に判定）
  hot   : 直近10日のうちに RSI(14)が70以上 または 25日線から+10%以上離れた日がある（上げすぎ）
  mk25  : TOPIX連動ETF(1306)の終値が25日線より下
  mk75  : TOPIX連動ETF(1306)の終値が75日線より下
  vol   : +1σを割った日の出来高が20日平均の1.5倍以上

資金・費用の前提は short_bw_backtest.py と同じ（1回約100万円、同時10銘柄、建玉1,000万円、貸株料年1.1%）。
各パターンについて、往復0.1%の売買コストを引いた場合の収支も出す。
出力 : backtest/short_bw2_summary.json, backtest/short_bw2_trades_<パターン>.csv
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_bw_backtest as bw  # noqa: E402

PERIOD = os.environ.get("BT_PERIOD", "6y")
OUT = "backtest"
FRANK_MAX = 4
COST = 0.001

EXITS = {"m1": "−1σ割れで利確", "m2": "−2σ割れで利確", "ma25": "25日線割れ後、25日線超えで買い戻し",
         "lock": "終値が−1σを割ったら、−1σに逆指値（戻れば利益確保）・−2σに指値で利確",
         "lockt": "安値が−1σに届いたら、−1σに逆指値（戻れば利益確保）・−2σに指値で利確"}
FILTERS = {"": "絞り込みなし", "hot": "上げすぎ（RSI70以上か25日線+10%以上）", "mk25": "TOPIXが25日線より下",
           "mk75": "TOPIXが75日線より下", "vol": "割れた日の出来高1.5倍以上",
           "hot+mk25": "上げすぎ＋TOPIXが25日線より下", "hot+vol": "上げすぎ＋出来高1.5倍以上",
           "mk25+vol": "TOPIXが25日線より下＋出来高1.5倍以上"}
PATTERNS = {f"{e}{('_' + f) if f else ''}": f"{en}／{fn}" for e, en in EXITS.items() for f, fn in FILTERS.items()}


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    o = np.full(len(c), np.nan)
    if len(c) <= n:
        return o
    d = np.diff(c)
    g, l = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), l[:n].mean()
    o[n] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(c)):
        ag = (ag * (n - 1) + g[i - 1]) / n
        al = (al * (n - 1) + l[i - 1]) / n
        o[i] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return o


def stock_flags(df: pd.DataFrame, i: int, cache: dict) -> dict:
    if "r" not in cache:
        c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
        ma25 = pd.Series(c).rolling(25).mean().to_numpy()
        cache.update(r=rsi(c), dev=c / ma25 - 1, v=v, vavg=pd.Series(v).rolling(20).mean().shift(1).to_numpy())
    r, dev, v, vavg = cache["r"], cache["dev"], cache["v"], cache["vavg"]
    w = slice(max(0, i - 9), i + 1)
    hot = bool(np.nanmax(r[w]) >= 70 or np.nanmax(dev[w]) >= 0.10)
    vol = bool(vavg[i] > 0 and v[i] >= vavg[i] * 1.5)
    return {"hot": hot, "vol": vol}


def simulate(code: str, df: pd.DataFrame, i: int, exit_rule: str, cache: dict) -> dict:
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    if "u1" not in cache:
        s = pd.Series(c)
        m = s.rolling(25).mean()
        sd = s.rolling(25).std(ddof=0)
        cache.update(u1=(m + sd).to_numpy(), d1=(m - sd).to_numpy(), d2=(m - 2 * sd).to_numpy(), m25=m.to_numpy())
    u1, d1, d2, m25 = cache["u1"], cache["d1"], cache["d2"], cache["m25"]
    e = i + 1
    entry = o[e]
    shares = max(100, int(bw.LOT_YEN // entry // 100 * 100))
    n = len(c)
    exit_i, why, below25, armed, px = None, None, False, False, None
    for j in range(e, min(n - 1, e + bw.MAX_HOLD)):
        if armed:
            s1, s2 = d1[j - 1], d2[j - 1]
            if o[j] <= s2:
                px, why = o[j], "利確（−2σ・寄りで約定）"
            elif o[j] >= s1:
                px, why = o[j], "利益確保（−1σ戻り・寄りで約定）"
            elif h[j] >= s1:
                px, why = s1, "利益確保（−1σ戻り）"
            elif l[j] <= s2:
                px, why = s2, "利確（−2σ）"
            if px is not None:
                exit_i = j
                break
            continue
        if exit_rule == "lock" and c[j] < d1[j]:
            armed = True
            continue
        if exit_rule == "lockt" and j > e and l[j] <= d1[j - 1]:
            armed = True
            continue
        if c[j] > u1[j]:
            exit_i, why = j + 1, "損切り（+1σ超え）"
            break
        if exit_rule == "m1" and c[j] < d1[j]:
            exit_i, why = j + 1, "利確（−1σ割れ）"
            break
        if exit_rule == "m2" and c[j] < d2[j]:
            exit_i, why = j + 1, "利確（−2σ割れ）"
            break
        if exit_rule == "ma25":
            if below25 and c[j] > m25[j]:
                exit_i, why = j + 1, "買い戻し（25日線超え）"
                break
            if c[j] < m25[j]:
                below25 = True
    if exit_i is None:
        if e + bw.MAX_HOLD <= n - 1:
            exit_i, why, px = e + bw.MAX_HOLD, "期限", o[e + bw.MAX_HOLD]
        else:
            exit_i, why, px = n - 1, "保有中（最終日の終値で評価）", c[n - 1]
    elif px is None:
        px = o[exit_i]
    days = (df.index[exit_i] - df.index[e]).days
    fee = entry * shares * bw.LEND_FEE * max(days, 1) / 365
    return {"code": code, "signal": df.index[i].strftime("%Y-%m-%d"),
            "entry_date": df.index[e].strftime("%Y-%m-%d"), "exit_date": df.index[exit_i].strftime("%Y-%m-%d"),
            "entry": round(entry, 1), "exit": round(px, 1), "shares": shares, "notional": round(entry * shares),
            "days": days, "why": why, "pnl": round((entry - px) * shares - fee), "fee": round(fee),
            "ret_pct": round((entry - px) / entry * 100, 2)}


def market_sets() -> dict:
    import yfinance as yf
    b = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    f = lambda m: {d.strftime("%Y-%m-%d") for d, ok in m.items() if ok}
    return {"mk25": f(b < b.rolling(25).mean()), "mk75": f(b < b.rolling(75).mean())}


def main() -> None:
    mk = market_sets()
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
    sigs = {code: bw.find_signals(df)["BW"] for code, df in clean.items()}
    need = sorted({clean[cd].index[i] for cd, s in sigs.items() for i in s})
    rank_on = {}
    for d in need:
        if d in closes.index:
            cm = {c: float(v) for c, v in closes.loc[d].items() if v == v}
            rank_on[d] = k.fund_ranks(master, clean, fund, close_map=cm)
    raw = {p: [] for p in PATTERNS}
    for code, df in clean.items():
        cache: dict = {}
        cand = []
        for i in sigs[code]:
            r = rank_on.get(df.index[i], {}).get(code)
            if r is None or r > FRANK_MAX:
                continue
            fl = stock_flags(df, i, cache)
            sd = df.index[i].strftime("%Y-%m-%d")
            fl["mk25"], fl["mk75"] = sd in mk["mk25"], sd in mk["mk75"]
            cand.append((i, sd, fl))
        for p in PATTERNS:
            ex, _, flt = p.partition("_")
            conds = [x for x in flt.split("+") if x]
            busy = ""
            for i, sd, fl in cand:
                if sd <= busy or not all(fl[x] for x in conds):
                    continue
                t = simulate(code, df, i, ex, cache)
                raw[p].append(t)
                busy = t["exit_date"]
    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(clean), "cost_pct": COST * 100,
               "patterns": {}}
    for p, name in PATTERNS.items():
        taken = sb.portfolio(raw[p])
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_bw2_trades_{p}.csv"), index=False, encoding="utf-8-sig")
        st = sb.stats(taken)
        if taken:
            d = pd.DataFrame(taken)
            d["net"] = d["pnl"] - d["notional"] * COST
            d["y"] = d["exit_date"].str[:4]
            st["avg_ret_pct"] = round(float(d["ret_pct"].mean()), 2)
            st["total_after_cost"] = int(d["net"].sum())
            st["by_year_after_cost"] = {y: int(g["net"].sum()) for y, g in d.groupby("y")}
            st["top10_pnl"] = int(d["pnl"].nlargest(10).sum())
        summary["patterns"][p] = {"name": name, "portfolio": st, "all_signals_trades": len(raw[p])}
        print(f"{p:14s} {st.get('trades', 0):5d}回 損益 {st.get('total_pnl', 0):>11,}円 "
              f"費用後 {st.get('total_after_cost', 0):>11,}円 勝率 {st.get('win_rate')}%")
    with open(os.path.join(OUT, "short_bw2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
