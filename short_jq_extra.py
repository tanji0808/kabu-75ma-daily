#!/usr/bin/env python3
"""
J-Quants のデータを使った空売りの追加検証

A. 貸借銘柄だけ（空売りできる銘柄）
   サイトのプリセットと同じルール（バンドウォーク崩れ×TOPIXより弱い×ランク4以下、−2σ割れで利確）の
   取引の一覧（5年・10年）を、J-Quants の上場銘柄一覧で「貸借」の銘柄だけに絞る。
   ※ 貸借区分は今の区分（過去に区分が変わった銘柄は分からない）。
B. 決算の直後に下げた銘柄を売る（J-Quants の決算データがある約2年）
   決算（決算短信）の発表に反応した日に、終値が前日より X%以上下げ、出来高が20日平均の2倍以上
   → 次の営業日の始値で空売り。15時以降の発表は翌営業日を反応した日とする。
   X = 3・5・8%。絞り込み：なし／ランク4以下（その時点の決算で計算）／貸借銘柄だけ／両方。
   買い戻し：m2（−2σ割れで利確・+1σ超えで損切り）、lk25c（25日線逆指値）、h20（20営業日後の始値で買い戻し）。
資金・費用は short_bw_backtest.py と同じ（1回約100万円、同時10銘柄、貸株料年1.1%）。往復0.1%の売買コストも引く。
出力 : backtest/short_jq_extra.json, backtest/short_earn_trades_<パターン>.csv
"""

from __future__ import annotations

import bisect
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
import short_rel_pit as sp  # noqa: E402

OUT = "backtest"
COST = 0.001


def year_stats(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"trades": 0}
    d = d.copy()
    d["net"] = d["pnl"] - d["notional"] * COST
    d["y"] = d["exit_date"].str[:4]
    eq = d.sort_values("exit_date")["pnl"].cumsum()
    return {"trades": int(len(d)), "win_rate": round(float((d["pnl"] > 0).mean() * 100), 1),
            "total_pnl": int(d["pnl"].sum()), "after_cost": int(d["net"].sum()),
            "max_drawdown": int((eq - eq.cummax()).min()), "avg_ret_pct": round(float(d["ret_pct"].mean()), 2),
            "by_year": {y: {"trades": int(len(g)), "win_rate": round(float((g["pnl"] > 0).mean() * 100), 1),
                            "after_cost": int(g["net"].sum())} for y, g in d.groupby("y")}}


def shortable_codes() -> set[str]:
    m = pd.read_csv("data/jq/master.csv", dtype=str)
    m["code"] = m["Code"].str[:4]
    return set(m.loc[m["MrgnNm"].fillna("").str.contains("貸借"), "code"])


def part_a(shortable: set[str]) -> dict:
    out = {}
    for suf in ("", "_11y"):
        p = f"{OUT}/short_rel_trades_L10_G5_f4__m2{suf}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, dtype={"code": str})
        out[suf or "_6y"] = {"all": year_stats(d), "shortable_only": year_stats(d[d["code"].isin(shortable)]),
                             "not_shortable": year_stats(d[~d["code"].isin(shortable)])}
    return out


def hold_exit(code: str, df: pd.DataFrame, i: int, days: int = 20) -> dict:
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    e = i + 1
    x = min(len(c) - 1, e + days)
    entry, px = o[e], (o[x] if e + days <= len(c) - 1 else c[-1])
    sh = max(100, int(bw.LOT_YEN // entry // 100 * 100))
    dd = (df.index[x] - df.index[e]).days
    fee = entry * sh * bw.LEND_FEE * max(dd, 1) / 365
    return {"code": code, "signal": df.index[i].strftime("%Y-%m-%d"), "entry_date": df.index[e].strftime("%Y-%m-%d"),
            "exit_date": df.index[x].strftime("%Y-%m-%d"), "entry": round(entry, 1), "exit": round(px, 1), "shares": sh,
            "notional": round(entry * sh), "days": dd, "why": "20営業日", "pnl": round((entry - px) * sh - fee),
            "fee": round(fee), "ret_pct": round((entry - px) / entry * 100, 2)}


def part_b(shortable: set[str]) -> dict:
    fins = sp.load_fins()
    disc = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="3y")
    clean = {}
    for c, df in prices.items():
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        if len(df) > 60:
            clean[c] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()})
    ev = []   # (code, 反応した日の位置, 下落率, 出来高倍率)
    for code, df in clean.items():
        c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        idx = df.index
        seen = set()
        for r in fins.get(code, []):
            if "FinancialStatements" not in str(r["DocType"]):
                continue
            dd = pd.Timestamp(r["DiscDate"])
            after = str(r["DiscTime"]) >= "15:00"
            pos = idx.searchsorted(dd, side="right" if after else "left")
            if pos < 21 or pos >= len(c) - 2 or pos in seen:
                continue
            seen.add(pos)
            drop = c[pos] / c[pos - 1] - 1
            vr = v[pos] / vavg[pos] if vavg[pos] > 0 else 0
            ev.append((code, pos, drop, vr))
    print(f"決算に反応した日: {len(ev)}件")
    need = sorted({clean[c].index[p] for c, p, d, v in ev if d <= -0.03 and v >= 2})
    rank = {}
    for ts in need:
        sd = ts.strftime("%Y-%m-%d")
        fund = {}
        for c in clean:
            n = bisect.bisect_right(disc.get(c, []), sd)
            if n:
                fund[c] = sp.pit_fund(fins[c][:n])
        cm = {c: float(x) for c, x in closes.loc[ts].items() if x == x}
        rank[ts] = k.fund_ranks(master, clean, fund, close_map=cm)
    res = {}
    for X in (3, 5, 8):
        for flt in ("", "f4", "shortable", "f4+shortable"):
            for ex in ("m2", "lk25c", "h20"):
                name = f"drop{X}{('_' + flt) if flt else ''}__{ex}"
                raw = []
                for code, pos, drop, vr in ev:
                    if drop > -X / 100 or vr < 2:
                        continue
                    df = clean[code]
                    if "f4" in flt:
                        rk = rank.get(df.index[pos], {}).get(code)
                        if rk is None or rk > 4:
                            continue
                    if "shortable" in flt and code not in shortable:
                        continue
                    if df["Volume"].iloc[pos] < bw.MIN_VOLUME:
                        continue
                    t = hold_exit(code, df, pos) if ex == "h20" else b2.simulate(code, df, pos, ex, {})
                    raw.append(t)
                taken = sb.portfolio(raw)
                if X == 5 and flt in ("", "f4+shortable") and ex in ("m2", "h20"):
                    pd.DataFrame(taken).to_csv(f"{OUT}/short_earn_trades_{name}.csv", index=False, encoding="utf-8-sig")
                res[name] = year_stats(pd.DataFrame(taken)) if taken else {"trades": 0}
                s = res[name]
                print(f"{name:28s} {s.get('trades', 0):4d}回 費用後 {s.get('after_cost', 0):>10,}円 勝率 {s.get('win_rate')}%")
    return res


def main() -> None:
    shortable = shortable_codes()
    print(f"貸借銘柄: {len(shortable)}")
    a = part_a(shortable)
    for kname, v in a.items():
        print(kname, {x: (y.get("trades"), y.get("after_cost")) for x, y in v.items()})
    b = part_b(shortable)
    with open(f"{OUT}/short_jq_extra.json", "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "shortable_count": len(shortable),
                   "A_shortable": a, "B_earnings_drop": b}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
