#!/usr/bin/env python3
"""
空売り バンドウォーク崩れ：入り口（売るタイミング・銘柄の選び方）の比較（過去約5年・東証プライム）

共通 : 直前にバンドウォーク（10日のうち7日以上、終値が+1σより上）で上昇していた銘柄が、終値で+1σを割った（= 合図）。
       売る日のファンダメンタルランク4以下、出来高50万株以上。売る日の翌営業日の始値で空売り。

入り口
  base : 合図の日に売る（今のやり方）
  reb  : 戻り売り。合図のあと20営業日以内に、いったん3%以上戻ったが高値は合図前の山を超えず（高値の切り下げ）、
         そのあと終値が戻る前の安値を割った日に売る
  ma25 : 合図のあと15営業日以内に、終値が25日線を上から下に割った日に売る
  big  : 合図の日が大陰線（始値から2%以上下げ、終値が日中の値幅の下から25%以内）か、窓を開けて下げた陰線
  rel  : 合図の日までの10営業日で、その銘柄がTOPIX（1306）より5%以上弱く、TOPIXは10日で下げていない
  cap  : 売る日の時価総額が1,000億円以上
  _vol : さらに、売る日の出来高が20日平均の1.5倍以上

買い戻し（short_bw2_backtest.py と同じ）
  m2    : 終値が−2σを割ったら翌日の始値で利確／終値が+1σを超えたら翌日の始値で損切り
  lk25c : 終値が−1σを割ったら25日線に逆指値（戻れば利益確保）、終値が−2σを割ったら翌日の始値で利確／+1σ超えで損切り

資金・費用・注意点は short_bw_backtest.py と同じ。往復0.1%の売買コストを引いた収支も出す。
出力 : backtest/short_entry_summary.json, backtest/short_entry_trades_<パターン>.csv
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
OUT = "backtest"
FRANK_MAX = 4
COST = 0.001
CAP_MIN = 100_000_000_000

ENTRIES = {"base": "合図の日に売る", "reb": "戻り売り（高値の切り下げを確認）", "ma25": "25日線割れまで待つ",
           "big": "大陰線・窓あけで割れた日だけ", "rel": "TOPIXより弱い銘柄だけ", "cap": "時価総額1,000億円以上だけ"}
EXITS = {"m2": "−2σ割れで利確", "lk25c": "25日線逆指値＋−2σ割れで利確"}
PATTERNS = {}
for en, enn in ENTRIES.items():
    for v in ("", "_vol"):
        for ex, exn in EXITS.items():
            PATTERNS[f"{en}{v}__{ex}"] = f"{enn}{'＋出来高1.5倍' if v else ''}／{exn}"


def triggers(df: pd.DataFrame, sig: list[int], topix: pd.Series | None) -> dict[str, list[int]]:
    """合図（+1σ割れの日）から、入り口ごとの売る日を求める。"""
    o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    n = len(c)
    ma25 = pd.Series(c).rolling(25).mean().to_numpy()
    out = {e: [] for e in ENTRIES}
    tpx = topix.reindex(df.index).ffill().to_numpy() if topix is not None else None
    for b in sig:
        out["base"].append(b)
        out["cap"].append(b)
        rng = h[b] - l[b]
        big = (o[b] - c[b]) / o[b] >= 0.02 and rng > 0 and (c[b] - l[b]) / rng <= 0.25
        gap = o[b] < c[b - 1] * 0.99 and c[b] < o[b]
        if big or gap:
            out["big"].append(b)
        if tpx is not None and b >= 10 and tpx[b - 10] == tpx[b - 10] and tpx[b] == tpx[b]:
            mr = tpx[b] / tpx[b - 10] - 1
            sr = c[b] / c[b - 10] - 1
            if mr >= 0 and sr - mr <= -0.05:
                out["rel"].append(b)
        for j in range(b + 1, min(n - 1, b + 16)):
            if c[j - 1] >= ma25[j - 1] and c[j] < ma25[j]:
                out["ma25"].append(j)
                break
        top = h[max(0, b - 10):b + 1].max()
        low_i = b
        for j in range(b + 1, min(n - 1, b + 21)):
            if h[j] >= top:
                break                       # 前の山を超えた → 戻り売りの形ではない
            seg = h[low_i + 1:j]            # 戻る前の安値（low_i）のあとの戻り高値
            if len(seg) and seg.max() >= l[low_i] * 1.03 and c[j] < l[low_i]:
                out["reb"].append(j)
                break
            if l[j] < l[low_i] and not (len(seg) and seg.max() >= l[low_i] * 1.03):
                low_i = j                   # まだ戻っていないうちは、安値を更新していく
    for e in out:
        out[e] = sorted(set(out[e]))
    return out


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
    trig = {code: triggers(df, bw.find_signals(df)["BW"], topix) for code, df in clean.items()}
    need = sorted({clean[cd].index[i] for cd, t in trig.items() for e in t for i in t[e]})
    rank_on = {}
    for d in need:
        if d in closes.index:
            cm = {c: float(v) for c, v in closes.loc[d].items() if v == v}
            rank_on[d] = k.fund_ranks(master, clean, fund, close_map=cm)
    print(f"ファンダメンタルランクを計算した日数: {len(rank_on)}")
    raw = {p: [] for p in PATTERNS}
    counts = {e: 0 for e in ENTRIES}
    for code, df in clean.items():
        cache: dict = {}
        c = df["Close"].to_numpy(float)
        sh = (fund.get(code) or {}).get("sh")
        for e in ENTRIES:
            ok = []
            for i in trig[code][e]:
                r = rank_on.get(df.index[i], {}).get(code)
                if r is None or r > FRANK_MAX or df["Volume"].iloc[i] < bw.MIN_VOLUME:
                    continue
                if e == "cap" and not (sh and sh * c[i] >= CAP_MIN):
                    continue
                ok.append((i, b2.stock_flags(df, i, cache)["vol"]))
            counts[e] += len(ok)
            for v in ("", "_vol"):
                for ex in EXITS:
                    p = f"{e}{v}__{ex}"
                    busy = ""
                    for i, vol in ok:
                        sd = df.index[i].strftime("%Y-%m-%d")
                        if sd <= busy or (v and not vol):
                            continue
                        t = b2.simulate(code, df, i, ex, cache)
                        raw[p].append(t)
                        busy = t["exit_date"]
    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(clean), "cost_pct": COST * 100,
               "entry_counts": counts, "patterns": {}}
    for p, name in PATTERNS.items():
        taken = sb.portfolio(raw[p])
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_entry_trades_{p}.csv"), index=False, encoding="utf-8-sig")
        st = sb.stats(taken)
        if taken:
            d = pd.DataFrame(taken)
            d["net"] = d["pnl"] - d["notional"] * COST
            d["y"] = d["exit_date"].str[:4]
            st["avg_ret_pct"] = round(float(d["ret_pct"].mean()), 2)
            st["total_after_cost"] = int(d["net"].sum())
            st["by_year_after_cost"] = {y: int(g["net"].sum()) for y, g in d.groupby("y")}
            st["after_cost_excl_top10"] = int(d["net"].sum() - d["pnl"].nlargest(10).sum())
        summary["patterns"][p] = {"name": name, "portfolio": st, "all_signals_trades": len(raw[p])}
        print(f"{p:16s} {st.get('trades', 0):5d}回 損益 {st.get('total_pnl', 0):>11,}円 "
              f"費用後 {st.get('total_after_cost', 0):>11,}円 勝率 {st.get('win_rate')}%")
    with open(os.path.join(OUT, "short_entry_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
