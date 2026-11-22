#!/usr/bin/env python3
"""
空売りの「条件をゆるめた候補」をすべて洗い出し、売った結果と判断材料を1つの表にまとめる（10年・上場廃止銘柄を含む）。
この表を使って、条件の組み合わせを手元でまとめて試す（2017〜2021年で選び、2022〜2026年で確かめる）。

候補 : 終値が+1σを上から下に割った日で、その前の10日のうち5日以上は終値が+1σより上（今のルールは7日以上）。
       出来高20万株以上（今のルールは50万株以上）。翌営業日の始値で空売り。
判断材料（すべて合図の日の時点で分かっていたもの）
  walk   : 前の10日のうち+1σより上だった日数
  rel10, rel20 : 10日・20日の騰落率 − TOPIX(1306)の騰落率（％）
  tpx10  : TOPIXの10日の騰落率（％）。tpx_bull : TOPIXが200日線より上か
  vol, vratio : 出来高、出来高 ÷ 20日平均
  dev25, rsi : 25日線からの乖離（％）、RSI(14)
  rank   : その時点の決算でのファンダメンタルランク（J-Quants・負債比率を除く）
  eg, rg, roe : その時点の増益率・増収率・ROE
  rev    : 直近90日の業績予想の修正（-1=予想EPSを下げた、1=上げた、0=なし）
  lsh, lchg, mratio : 買い残 ÷ 発行済株式（％）、買い残の4週間の増減、買い残 ÷ 売り残（J-Quants の信用残。公表は金曜の5日後とみなす）
  cap    : 時価総額（億円）
売った結果 : m2（−2σ割れで利確）と lk25c（25日線逆指値＋−2σ割れで利確）。損切りはどちらも+1σ超え。1回約100万円・貸株料あり。
出力 : backtest/short_candidates_11y.csv.gz
"""

from __future__ import annotations

import bisect
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_bw2_backtest as b2  # noqa: E402
import short_rel_pit as sp  # noqa: E402

PERIOD = os.environ.get("BT_PERIOD", "11y")
WARMUP = 90
MIN_VOL = 200_000


def margin_table():
    m = pd.read_csv("data/jq/margin_interest.csv.gz", dtype=str)
    for c in ("ShrtVol", "LongVol"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["code"] = m["Code"].str[:4]
    m["avail"] = (pd.to_datetime(m["Date"]) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d")
    m = m.sort_values(["code", "Date"])
    return {c: (list(g["avail"]), g["LongVol"].to_numpy(float), g["ShrtVol"].to_numpy(float)) for c, g in m.groupby("code")}


def rev_flag(recs: list[dict], sd: str) -> int:
    """直近90日に業績予想（今期の予想EPS）を変えたか。前の予想より下げたら-1、上げたら1。"""
    lo = (pd.Timestamp(sd) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    prev, flag = None, 0
    for r in recs:
        fy = r.get("CurFYEn")
        fe = r["NxFEPS"] if (r["CurPerType"] == "FY" and "FinancialStatements" in str(r["DocType"])) else r["FEPS"]
        if fe != fe or fe is None:
            continue
        is_fy = r["CurPerType"] == "FY" and "FinancialStatements" in str(r["DocType"])
        if fy != fy or not fy:
            continue
        # 通期決算の「来期の予想」は、1年後に終わる期の予想として扱う
        key = (pd.Timestamp(fy) + pd.DateOffset(years=1)).strftime("%Y-%m") if is_fy else pd.Timestamp(fy).strftime("%Y-%m")
        if prev is not None and prev[0] == key and r["DiscDate"] >= lo and fe != prev[1]:
            flag = -1 if fe < prev[1] else 1
        prev = (key, fe)
    return flag


def main() -> None:
    import yfinance as yf
    topix = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    fins = sp.load_fins()
    disc = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    mg = margin_table()
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        if len(df) >= WARMUP + 20:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()}).sort_index()
    tpx = topix.copy()
    tpx.index = pd.to_datetime(tpx.index).tz_localize(None).normalize()
    tma200 = tpx.rolling(200).mean()

    # 1) 候補を洗い出す
    cand = []
    for code, df in clean.items():
        c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
        s = pd.Series(c)
        m25 = s.rolling(25).mean()
        u1 = (m25 + s.rolling(25).std(ddof=0)).to_numpy()
        walk = pd.Series((c > u1).astype(int)).rolling(10).sum().shift(1).to_numpy()
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        rsi = b2.rsi(c)
        t = tpx.reindex(df.index).ffill().to_numpy()
        tb = (tpx > tma200).reindex(df.index).ffill().to_numpy()
        for i in range(WARMUP, len(c) - 1):
            if np.isnan(u1[i]) or np.isnan(u1[i - 1]) or v[i] < MIN_VOL:
                continue
            if not (c[i - 1] > u1[i - 1] and c[i] < u1[i]) or walk[i] < 5:
                continue
            row = {"code": code, "i": i, "signal": df.index[i].strftime("%Y-%m-%d"), "walk": int(walk[i]),
                   "vol": v[i], "vratio": v[i] / vavg[i] if vavg[i] > 0 else np.nan,
                   "dev25": (c[i] / m25[i] - 1) * 100, "rsi": rsi[i], "tpx_bull": bool(tb[i]) if tb[i] == tb[i] else None}
            for L in (10, 20):
                if i >= L and t[i - L] == t[i - L] and t[i] == t[i]:
                    mr = t[i] / t[i - L] - 1
                    row[f"rel{L}"] = (c[i] / c[i - L] - 1 - mr) * 100
                    if L == 10:
                        row["tpx10"] = mr * 100
            cand.append(row)
    print(f"候補: {len(cand)}件")

    # 2) その時点の決算・信用残
    pit_cache: dict = {}

    def pit(code, n):
        kk = (code, n)
        if kk not in pit_cache:
            pit_cache[kk] = sp.pit_fund(fins[code][:n])
        return pit_cache[kk]

    by_date: dict[str, list] = {}
    for r in cand:
        by_date.setdefault(r["signal"], []).append(r)
    for n, (sd, rows) in enumerate(sorted(by_date.items())):
        ts = pd.Timestamp(sd)
        fund = {}
        for c in clean:
            nn = bisect.bisect_right(disc.get(c, []), sd)
            if nn:
                fund[c] = pit(c, nn)
        cm = {c: float(x) for c, x in closes.loc[ts].items() if x == x} if ts in closes.index else {}
        ranks = k.fund_ranks(master, clean, fund, close_map=cm)
        for r in rows:
            c = r["code"]
            r["rank"] = ranks.get(c)
            f = fund.get(c, {})
            for x in ("eg", "rg", "roe"):
                r[x] = f.get(x)
            nn = bisect.bisect_right(disc.get(c, []), sd)
            r["rev"] = rev_flag(fins[c][:nn], sd) if nn else 0
            sh = None
            for rec in reversed(fins.get(c, [])[:nn]):
                if rec["ShOutFY"] == rec["ShOutFY"] and rec["ShOutFY"]:
                    sh = rec["ShOutFY"] - (rec["TrShFY"] if rec["TrShFY"] == rec["TrShFY"] else 0)
                    break
            if sh and c in cm:
                r["cap"] = sh * cm[c] / 1e8
            g = mg.get(c)
            if g:
                j = bisect.bisect_right(g[0], sd) - 1
                if j >= 0:
                    lv, sv = g[1][j], g[2][j]
                    r["mratio"] = lv / sv if sv > 0 else np.inf
                    if sh:
                        r["lsh"] = lv / sh * 100
                    if j >= 4 and g[1][j - 4] > 0:
                        r["lchg"] = lv / g[1][j - 4] - 1
        if n % 200 == 0:
            print(f"  {sd}（{n + 1}/{len(by_date)}日）")

    # 3) 売った結果
    caches: dict = {}
    for r in cand:
        df = clean[r["code"]]
        cache = caches.setdefault(r["code"], {})
        for ex in ("m2", "lk25c"):
            t = b2.simulate(r["code"], df, r["i"], ex, cache)
            for x in ("entry_date", "exit_date", "notional", "pnl", "ret_pct", "why"):
                r[f"{ex}_{x}"] = t[x]
    out = pd.DataFrame(cand).drop(columns=["i"])
    os.makedirs("backtest", exist_ok=True)
    out.to_csv("backtest/short_candidates_11y.csv.gz", index=False, compression="gzip")
    print(f"保存しました: {len(out)}件")


if __name__ == "__main__":
    main()
