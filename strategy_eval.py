#!/usr/bin/env python3
"""
採用候補の2つの手法を、実際のコストとリスクを減らす工夫で検証し直す（10年・上場廃止銘柄を含む）。

手法
  空売り : 候補の表（short_candidates_11y）で、バンドウォーク5日以上・出来高20万株以上・TOPIXより10日で5%以上弱い・
           その時点のランク5以下・買い残÷発行済株式1.0%以上・買い残の4週間の増加2割未満・上方修正なし。
           点数（TOPIXより8%以上弱い／下方修正／25日線より3%以上上／買い残2.5%以上）で 0点50万〜3点以上200万円。
           25日線逆指値（short_bw2_backtest の lk25c）。同時10銘柄・建玉1,000万円まで。
  買い   : 候補の表（buy_candidates_11y）で、その時点のランク6以上・5日+3%以下・RSI60以下。
           金額はランクで 6:50万・7:100万・8:150万・9〜10:200万、さらに点数（20日でTOPIXより8%以上強い／
           半年高値から45%以上下落／決算から10日以内／TOPIXが20日で3%以上下落／直近の増益率−30%以下）で
           0点×1・1点×1.5・2点以上×2。今の売り方（引き上げ式ロスカット）。同時10銘柄・1,000万円まで。
比べるもの
  base     : これまでの検証と同じ（往復0.1%）
  cost     : 売買のたびに値段が0.1%不利（往復0.2%）。空売りは権利付最終日をまたいだら配当相当額を払う
  earn     : cost に加えて、決算発表をまたがない（発表の前に手じまい。発表まで2営業日以内なら入らない）
  rights   : cost に加えて、空売りは権利付最終日をまたがない（その日の終値で買い戻し。2営業日以内なら入らない）
  both     : earn と rights の両方
出力 : backtest/strategy_eval.json, backtest/strategy_trades_<手法>_<比べるもの>.csv
"""

from __future__ import annotations

import bisect
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import buy_candidates as bc  # noqa: E402
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_bw2_backtest as b2  # noqa: E402
import short_rel_pit as sp  # noqa: E402

PERIOD = "11y"
SLIP = float(os.environ.get("SLIP", "0.001"))      # 片道
W_SHORT = [0.5, 1, 1.5, 2, 2]
LOT_BUY = {6: 0.5, 7: 1.0, 8: 1.5}
VARIANTS = ["base", "cost", "earn", "rights", "both"]


def last_cum_days(index: pd.DatetimeIndex, fy_end_months: set[int]) -> set[int]:
    """権利付最終日（権利確定日＝月末の2営業日前、2019年7月より前は3営業日前）の位置。"""
    out = set()
    s = pd.Series(range(len(index)), index=index)
    for (y, m), g in s.groupby([index.year, index.month]):
        if m not in fy_end_months:
            continue
        last = g.iloc[-1]                     # 月の最終営業日（権利確定日とみなす）
        lag = 3 if pd.Timestamp(y, m, 1) < pd.Timestamp(2019, 7, 1) else 2
        if last - lag >= 0:
            out.add(int(last - lag))
    return out


def earnings_cut(dates: list[str], disc: list[tuple], e: int) -> int | None:
    """持ち始め（e）以降で最初の決算発表に対し、何日目の終値までに手じまうか。"""
    sd = dates[e]
    for dd, tm in disc:
        if dd < sd:
            continue
        pos = bisect.bisect_left(dates, dd)
        if pos >= len(dates):
            return None
        return pos if str(tm) >= "15:00" else pos - 1
    return None


def main() -> None:
    import yfinance as yf  # noqa: F401
    fins = sp.load_fins()
    disc_times = {}
    fy_months, divs = {}, {}
    for c, recs in fins.items():
        st = [(r["DiscDate"], r["DiscTime"]) for r in recs if "FinancialStatements" in str(r["DocType"])]
        disc_times[c] = sorted(set(st))
        ends = [r["CurFYEn"] for r in recs if r["CurFYEn"] == r["CurFYEn"] and r["CurFYEn"]]
        if ends:
            m = pd.Timestamp(ends[-1]).month
            fy_months[c] = {m, (m + 6 - 1) % 12 + 1}   # 期末と中間（半年前）
        divs[c] = ([r["DiscDate"] for r in recs], [sp.pit_fund(recs[:i + 1]).get("div") for i in range(len(recs))])
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        clean[code] = df

    # ---------- 空売り
    sc = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str})
    sc = sc[sc["signal"] >= "2017-01-01"].copy()
    sc["score"] = ((sc.rel10 <= -8).astype(int) + (sc.rev == -1).astype(int) + (sc.dev25 >= 3).astype(int)
                   + (sc.lsh >= 2.5).astype(int))
    sc = sc[(sc.walk >= 5) & (sc.vol >= 200000) & (sc.rel10 <= -5) & (sc["rank"] <= 5) & (sc.lsh >= 1.0)
            & (sc.lchg < 0.2) & (sc.rev != 1)]
    # ---------- 買い
    bcd = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str})
    bcd = bcd[(bcd["signal"] >= "2017-01-01") & (bcd["rank"] >= 6) & (bcd.up5 <= 3) & (bcd.rsi <= 60)
              & bcd.cur_ret.notna()].copy()
    bcd["score"] = ((bcd.rel20 >= 8).astype(int) + (bcd["drop"] >= 45).astype(int) + (bcd.dsince <= 10).astype(int)
                    + (bcd.tpx20 <= -3).astype(int) + (bcd.eg_last <= -0.3).astype(int))
    bcd["eg"] = bcd["eg"].fillna(-9)

    res = {}
    for side, cand in (("short", sc), ("buy", bcd)):
        for var in VARIANTS:
            if side == "buy" and var in ("rights",):
                continue
            rows = []
            for x in cand.itertuples():
                df = clean.get(x.code)
                if df is None:
                    continue
                dates = [d.strftime("%Y-%m-%d") for d in df.index]
                i = bisect.bisect_left(dates, x.signal)
                if i >= len(dates) or dates[i] != x.signal or i + 1 >= len(dates):
                    continue
                e = i + 1
                cut = None
                if var in ("earn", "both"):
                    ec = earnings_cut(dates, disc_times.get(x.code, []), e)
                    if ec is not None:
                        if ec <= e + 1:
                            continue
                        cut = ec
                lcd = last_cum_days(df.index, fy_months.get(x.code, set())) if side == "short" else set()
                if side == "short" and var in ("rights", "both"):
                    nxt = [p for p in lcd if p >= e]
                    if nxt:
                        rc = min(nxt)
                        if rc <= e + 1:
                            continue
                        cut = rc if cut is None else min(cut, rc)
                sub = df.iloc[:cut + 1] if cut is not None else df
                if side == "short":
                    t = b2.simulate(x.code, sub, i, "lk25c", {})
                    w = W_SHORT[min(int(x.score), 4)]
                    notional, pnl = t["notional"] * w, t["pnl"] * w
                    ret = t["ret_pct"]
                    entry_d, exit_d = t["entry_date"], t["exit_date"]
                    shares = t["shares"] * w
                    div_paid = 0.0
                    if var != "base":
                        xi = bisect.bisect_left(dates, exit_d)
                        for p in lcd:
                            if e <= p < xi:          # 権利付最終日の終値を持ち越した → 配当相当額を払う
                                dd = dates[p]
                                ds, dv = divs.get(x.code, ([], []))
                                n = bisect.bisect_right(ds, dd)
                                v = next((dv[q] for q in range(n - 1, -1, -1) if dv[q] is not None), None)
                                if v:
                                    div_paid += v / 2 * shares
                        pnl -= div_paid
                    rows.append({"code": x.code, "signal": x.signal, "entry_date": entry_d, "exit_date": exit_d,
                                 "notional": notional, "pnl": pnl, "ret_pct": ret, "score": int(x.score),
                                 "div_paid": div_paid, "why": t["why"]})
                else:
                    s_ = pd.Series(sub["Close"].to_numpy(float))
                    m25 = s_.rolling(25).mean()
                    sd = s_.rolling(25).std(ddof=0)
                    a = {"o": sub["Open"].to_numpy(float), "c": sub["Close"].to_numpy(float), "l": sub["Low"].to_numpy(float),
                         "d1": (m25 - sd).to_numpy(), "u1": (m25 + sd).to_numpy(), "m25": m25.to_numpy(),
                         "dates": dates[:len(sub)]}
                    r = bc.sim(a, i, None, False, "u1")
                    if r is None and cut is not None:        # 決算の前で手じまい（その日の終値）
                        px = a["c"][cut]
                        r = {"ret": round((px / a["o"][e] - 1) * 100, 2), "exit": dates[cut], "entry": dates[e], "reason": "決算前"}
                    if r is None:
                        continue
                    w = LOT_BUY.get(int(x.rank), 2.0) * [1, 1.5, 2][min(int(x.score), 2)]
                    rows.append({"code": x.code, "signal": x.signal, "entry_date": r["entry"], "exit_date": r["exit"],
                                 "notional": 1e6 * w, "pnl": 1e4 * r["ret"] * w, "ret_pct": r["ret"], "eg": x.eg,
                                 "score": int(x.score), "why": r["reason"]})
            cost = 0.001 if var == "base" else 2 * SLIP
            if side == "short":
                sb.MAX_POS, sb.MAX_NOTIONAL = 10, 10_000_000
                taken = sb.portfolio(rows)
            else:
                import buy_grid as bg
                taken = bg.portfolio(rows)
            d = pd.DataFrame(taken).sort_values("exit_date")
            d["net"] = d["pnl"] - d["notional"] * cost
            eq = d["net"].cumsum()
            d["y"] = d["exit_date"].str[:4]
            by = {y: {"trades": int(len(g)), "win": round(float((g["pnl"] > 0).mean() * 100), 1), "net": int(g["net"].sum())}
                  for y, g in d.groupby("y")}
            res[f"{side}_{var}"] = {"trades": int(len(d)), "net": int(d["net"].sum()), "dd": int((eq - eq.cummax()).min()),
                                    "win": round(float((d["pnl"] > 0).mean() * 100), 1), "years": by,
                                    "div_paid": int(d["div_paid"].sum()) if "div_paid" in d else 0}
            d.to_csv(f"backtest/strategy_trades_{side}_{var}.csv", index=False, encoding="utf-8-sig")
            r_ = res[f"{side}_{var}"]
            print(f"{side:5s} {var:6s} {r_['trades']}回 {r_['net'] / 1e4:+.0f}万 落ち込み{r_['dd'] / 1e4:.0f}万 配当相当額{r_['div_paid'] / 1e4:.0f}万")
    json.dump(res, open("backtest/strategy_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
