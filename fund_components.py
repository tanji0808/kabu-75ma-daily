#!/usr/bin/env python3
"""
ファンダメンタルランクの項目ごとの値を、買い・空売りの候補（合図の日）ごとに、その時点の決算で計算する。
ランクの付け方を見直すための材料（どの項目が成績に効いているかを調べる）。

項目（kabu_75ma_search._fund_metrics と同じ計算＋自己資本比率）
  per, per_rel（業種の中央値に対する倍率）, per_neg（赤字）, pbr, pbr_rel, roe, om, om_rel, rg, eg,
  fg（予想EPS ÷ 実績EPS − 1）, dy（配当利回り％）, eqr（自己資本比率％、J-Quants の Eq ÷ TA）, rank（今の付け方）
入力 : backtest/buy_candidates_11y.csv.gz, backtest/short_candidates_11y.csv.gz（合図の日とコード）
出力 : backtest/fund_components_buy.csv.gz, backtest/fund_components_short.csv.gz
"""

from __future__ import annotations

import bisect
import os
import statistics
import sys

import pandas as pd

os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_rel_pit as sp  # noqa: E402

PERIOD = "11y"


def eq_ratio(recs: list[dict]):
    for r in reversed(recs):
        if "FinancialStatements" in str(r["DocType"]) and r["Eq"] == r["Eq"] and r["TA"] == r["TA"] and r["TA"]:
            return r["Eq"] / r["TA"] * 100
    return None


def main() -> None:
    fins = sp.load_fins()
    disc = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    master = k.load_prime_list()
    sec_of = dict(zip(master["コード"], master["業種"]))
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()}).sort_index()
    buy = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    short = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    need = sorted(set(buy["signal"]) | set(short["signal"]))
    need = [d for d in need if d >= "2017-01-01"]
    pit_cache: dict = {}
    comp: dict = {}
    for n, sd in enumerate(need):
        ts = pd.Timestamp(sd)
        if ts not in closes.index:
            continue
        row = closes.loc[ts]
        met, fund = {}, {}
        for c in clean:
            px = row.get(c)
            if px is None or px != px or px <= 0:
                continue
            nn = bisect.bisect_right(disc.get(c, []), sd)
            if not nn:
                continue
            if (c, nn) not in pit_cache:
                recs = fins[c][:nn]
                pit_cache[(c, nn)] = (sp.pit_fund(recs), eq_ratio(recs))
            f, eqr = pit_cache[(c, nn)]
            fund[c] = f
            m = k._fund_metrics(f, float(px))
            m["eqr"] = eqr
            met[c] = m
        st = {}
        for sec in set(sec_of.get(c) for c in met):
            arr = [met[c] for c in met if sec_of.get(c) == sec]

            def med(key, lo, hi):
                v = [x[key] for x in arr if x[key] is not None and lo < x[key] < hi]
                return statistics.median(v) if v else None
            st[sec] = {"per": med("per", 0, 200), "pbr": med("pbr", 0, 30), "om": med("om", -100, 100)}
        cm = {c: float(x) for c, x in row.items() if x == x}
        ranks = k.fund_ranks(master, clean, fund, close_map=cm)
        comp[sd] = (met, st, ranks)
        if n % 250 == 0:
            print(f"  {sd}（{n + 1}/{len(need)}日）")

    def build(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for r in df[df["signal"] >= "2017-01-01"].drop_duplicates().itertuples():
            if r.signal not in comp:
                continue
            met, st, ranks = comp[r.signal]
            m = met.get(r.code)
            if not m:
                continue
            S = st.get(sec_of.get(r.code), {})
            rel = lambda a, b: (a / b) if (a is not None and b) else None
            fg = (m["feps"] / m["eps"] - 1) if (m["eps"] and m["feps"] is not None and m["eps"] > 0) else None
            out.append({"code": r.code, "signal": r.signal, "rank": ranks.get(r.code), "per": m["per"], "per_neg": m["perNeg"],
                        "per_rel": rel(m["per"], S.get("per")), "pbr": m["pbr"], "pbr_rel": rel(m["pbr"], S.get("pbr")),
                        "roe": m["roe"], "om": m["om"], "om_rel": rel(m["om"], S.get("om")) if (S.get("om") or 0) > 0 else None,
                        "rg": m["rg"], "eg": m["eg"], "fg": fg, "dy": m["dy"], "eqr": m["eqr"]})
        return pd.DataFrame(out)

    build(buy).to_csv("backtest/fund_components_buy.csv.gz", index=False, compression="gzip")
    build(short).to_csv("backtest/fund_components_short.csv.gz", index=False, compression="gzip")
    print("保存しました")


if __name__ == "__main__":
    main()
