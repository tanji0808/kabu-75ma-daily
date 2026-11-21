#!/usr/bin/env python3
"""
空売り「バンドウォーク崩れ×TOPIXより弱い×ファンダメンタル4以下」の追加の工夫を比べる。

土台（base）: サイトのプリセットと同じ。バンドウォーク崩れ、10営業日でTOPIX(1306)より5%以上弱い、
              ファンダメンタルランク4以下（今の財務データ）、出来高50万株以上。翌営業日の始値で空売り。
              買い戻しは m2（−2σ割れで利確／+1σ超えで損切り）。1回100万円。
比べる工夫
  sec     : TOPIXの代わりに「同じ業種の平均（自分を除く）」より10営業日で5%以上弱い
  sec+tpx : TOPIXより弱く、業種の平均よりも弱い
  bull    : TOPIXが200日線より上のときだけ（上げ相場のときだけ）
  bear    : TOPIXが200日線より下のときだけ（参考）
  vsize   : 値動きの荒さで株数を変える。1回の金額 = 100万円 × 2% ÷ その銘柄の20日の値動きの大きさ
            （30万〜200万円の範囲）。荒い銘柄は少なく、穏やかな銘柄は多く建てる
  組み合わせ : sec+bull, sec+tpx+bull, bull+vsize, sec+vsize, sec+bull+vsize
期間は BT_PERIOD（6y=約5年、11y=約10年）。資金・費用・注意点は short_bw_backtest.py と同じ。
出力 : backtest/short_ext<_期間>.json, backtest/short_ext_trades_<パターン><_期間>.csv
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
import short_bw2_backtest as b2  # noqa: E402

PERIOD = os.environ.get("BT_PERIOD", "6y")
SUF = "" if PERIOD == "6y" else f"_{PERIOD}"
OUT = "backtest"
COST = 0.001
L, G, FMAX = 10, 5, 4
PATTERNS = ["base", "sec", "sec+tpx", "bull", "bear", "vsize",
            "sec+bull", "sec+tpx+bull", "bull+vsize", "sec+vsize", "sec+bull+vsize"]


def summarize(taken: list[dict]) -> dict:
    st = sb.stats(taken)
    if not taken:
        return st
    d = pd.DataFrame(taken)
    d["net"] = d["pnl"] - d["notional"] * COST
    d["y"] = d["exit_date"].str[:4]
    by = {y: {"trades": int(len(g)), "win_rate": round(float((g["pnl"] > 0).mean() * 100), 1),
              "pnl": int(g["pnl"].sum()), "after_cost": int(g["net"].sum())} for y, g in d.groupby("y")}
    st.update(after_cost=int(d["net"].sum()), avg_ret_pct=round(float(d["ret_pct"].mean()), 2),
              avg_notional=int(d["notional"].mean()), by_year_detail=by,
              plus_years=sum(v["after_cost"] > 0 for v in by.values()), years=len(by),
              after_cost_excl_top10=int(d["net"].sum() - d["pnl"].nlargest(10).sum()))
    return st


def main() -> None:
    import yfinance as yf
    topix = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    tma200 = topix.rolling(200).mean()
    master = k.load_prime_list()
    sector = dict(zip(master["コード"], master["業種"]))
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    fund = json.load(open("data/fundamentals.json", encoding="utf-8"))
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) >= bw.WARMUP + 20:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()}).sort_index()
    # 業種の平均（自分を除く）の10営業日の騰落率
    ret = closes / closes.shift(L) - 1
    ret = ret.where(ret.abs() < 1.5)
    sec_sum, sec_cnt = {}, {}
    for sec in set(sector.get(c) for c in closes.columns):
        cols = [c for c in closes.columns if sector.get(c) == sec]
        sec_sum[sec] = ret[cols].sum(axis=1, min_count=1)
        sec_cnt[sec] = ret[cols].notna().sum(axis=1)
    tpx = topix.reindex(closes.index).ffill()
    tret = tpx / tpx.shift(L) - 1
    bull = (tpx > tma200.reindex(closes.index).ffill())

    raw_sig = []
    for code, df in clean.items():
        v = df["Volume"].to_numpy(float)
        for i in bw.find_signals(df)["BW"]:
            if v[i] >= bw.MIN_VOLUME:
                raw_sig.append((code, i))
    need = sorted({clean[c].index[i] for c, i in raw_sig})
    rank_on = {}
    for d in need:
        cm = {c: float(x) for c, x in closes.loc[d].items() if x == x}
        rank_on[d] = k.fund_ranks(master, clean, fund, close_map=cm)
    print(f"ファンダメンタルランクを計算した日数: {len(rank_on)}")

    by_code: dict[str, list] = {}
    for code, i in raw_sig:
        df = clean[code]
        d = df.index[i]
        rk = rank_on.get(d, {}).get(code)
        if rk is None or rk > FMAX:
            continue
        r = ret.at[d, code] if d in ret.index else np.nan
        if r != r:
            continue
        sec = sector.get(code)
        n = sec_cnt[sec].get(d, 0)
        sec_avg = (sec_sum[sec].get(d, np.nan) - r) / (n - 1) if n > 1 else np.nan
        tr = tret.get(d, np.nan)
        c = df["Close"].to_numpy(float)
        vol20 = float(pd.Series(c[max(0, i - 20):i + 1]).pct_change().std())
        by_code.setdefault(code, []).append({
            "i": i, "sd": d.strftime("%Y-%m-%d"),
            "tpx": tr == tr and r - tr <= -G / 100,
            "sec": sec_avg == sec_avg and r - sec_avg <= -G / 100,
            "bull": bool(bull.get(d, False)), "vol20": vol20})
    raw = {p: [] for p in PATTERNS}
    for code, sigs in by_code.items():
        df = clean[code]
        cache: dict = {}
        for p in PATTERNS:
            parts = set(p.split("+"))
            busy = ""
            for s in sigs:
                if s["sd"] <= busy:
                    continue
                ok_rel = (s["sec"] if "sec" in parts else True) and (s["tpx"] if ("tpx" in parts or "sec" not in parts) else True)
                if not ok_rel:
                    continue
                if "bull" in parts and not s["bull"]:
                    continue
                if "bear" in parts and s["bull"]:
                    continue
                t = dict(b2.simulate(code, df, s["i"], "m2", cache))
                if "vsize" in parts and s["vol20"] > 0:
                    lot = min(2_000_000, max(300_000, 1_000_000 * 0.02 / s["vol20"]))
                    sh = max(100, int(lot // t["entry"] // 100 * 100))
                    f = sh / t["shares"]
                    t.update(shares=sh, notional=round(t["entry"] * sh), pnl=round(t["pnl"] * f), fee=round(t["fee"] * f))
                raw[p].append(t)
                busy = t["exit_date"]
    res = {}
    for p in PATTERNS:
        taken = sb.portfolio(raw[p])
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_ext_trades_{p}{SUF}.csv"), index=False, encoding="utf-8-sig")
        res[p] = summarize(taken)
        s = res[p]
        print(f"{p:16s} {s.get('trades', 0):4d}回 費用後 {s.get('after_cost', 0):>11,}円 プラスの年 {s.get('plus_years')}/{s.get('years')} "
              f"落ち込み {s.get('max_drawdown', 0):>10,}円 勝率 {s.get('win_rate')}%")
    with open(os.path.join(OUT, f"short_ext{SUF}.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "period": PERIOD, "results": res},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
