#!/usr/bin/env python3
"""
空売り「バンドウォーク崩れ×TOPIXより弱い×ファンダメンタル4以下」を、
その時点で公表されていた決算（J-Quants の決算の概要）でランクを付け直して検証する。

比べるもの（同じ期間・同じ売買ルール）
  pit   : 合図の日までに公表されていた決算でランクを計算（後から知った情報を使わない）
  now   : 今の財務データ（data/fundamentals.json）でランクを計算（これまでの検証と同じ）
  nofund: ファンダメンタルの条件なし
売買ルール : サイトのプリセットと同じ。バンドウォーク崩れ（10日のうち7日以上+1σの上 → 終値で+1σ割れ）、
            10営業日でTOPIX(1306)より5%以上弱い、出来高50万株以上、ランク4以下。翌営業日の始値で空売り。
            m2 = −2σ割れで利確、lk25c = 25日線逆指値＋−2σ割れで利確。損切りはどちらも+1σ超え。
期間 : 決算データ（data/jq/fins_summary.csv.gz）で、プライム銘柄の8割以上に通期決算がそろった日から、
       決算データの最後の開示日まで。
J-Quants の決算の概要には有利子負債がないので、ランクの「負債比率」（0.5点）は全銘柄で使わない。
出力 : backtest/short_rel_pit.json, backtest/short_rel_pit_trades_<比較>__<買い戻し>.csv
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

PERIOD = os.environ.get("BT_PERIOD", "6y")
OUT = "backtest"
SUF = "" if PERIOD == "6y" else f"_{PERIOD}"
SUF += os.environ.get("BT_TAG", "")   # 例 _delisted（上場廃止銘柄を含めたとき）
COST = 0.001
L, G, FMAX = 10, 5, 4
NUM = ["Sales", "OP", "NP", "EPS", "BPS", "Eq", "TA", "DivAnn", "FDivAnn", "NxFDivAnn",
       "FSales", "FOP", "FNP", "FEPS", "NxFSales", "NxFOP", "NxFNp", "NxFEPS", "ShOutFY", "TrShFY"]


def load_fins() -> dict[str, list[dict]]:
    df = pd.read_csv("data/jq/fins_summary.csv.gz", dtype=str)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = df["Code"].str[:4]
    df = df.sort_values(["DiscDate", "DiscTime"])
    out: dict[str, list[dict]] = {}
    for code, g in df.groupby("code"):
        out[code] = g.to_dict("records")
    return out


def _v(x):
    return None if x is None or x != x else float(x)


def pit_fund(recs: list[dict]) -> dict:
    """その時点までの開示（古い順）から、スクリーナーと同じ項目の財務データを作る。"""
    stm = [r for r in recs if "FinancialStatements" in str(r["DocType"])]
    fy = [r for r in stm if r["CurPerType"] == "FY"]
    f: dict = {}
    if fy:
        y = fy[-1]
        f["eps"] = _v(y["EPS"])
        if _v(y["NP"]) is not None and _v(y["Eq"]):
            f["roe"] = y["NP"] / y["Eq"]
        if _v(y["OP"]) is not None and _v(y["Sales"]):
            f["om"] = y["OP"] / y["Sales"]
    for r in reversed(stm):
        if _v(r["BPS"]):
            f["bps"] = r["BPS"]
            break
    # 予想EPS・予想配当：いちばん新しい開示（予想の修正も含む）。通期決算の開示では「来期の予想」
    for r in reversed(recs):
        is_fy = r["CurPerType"] == "FY" and "FinancialStatements" in str(r["DocType"])
        fe = _v(r["NxFEPS"]) if is_fy else _v(r["FEPS"])
        if fe is not None:
            f["feps"] = fe
            break
    for r in reversed(recs):
        is_fy = r["CurPerType"] == "FY" and "FinancialStatements" in str(r["DocType"])
        dv = _v(r["NxFDivAnn"]) if is_fy else _v(r["FDivAnn"])
        if dv is None:
            dv = _v(r["DivAnn"])
        if dv is not None:
            f["div"] = dv
            break
    # 増収率・増益率：最新の決算と、1年前の同じ期間（四半期累計）を比べる
    if stm:
        s = stm[-1]
        end = pd.Timestamp(s["CurPerEn"]) if s["CurPerEn"] == s["CurPerEn"] else None
        prev = None
        if end is not None:
            for r in reversed(stm[:-1]):
                if r["CurPerType"] == s["CurPerType"] and r["CurPerEn"] == r["CurPerEn"]:
                    gap = (end - pd.Timestamp(r["CurPerEn"])).days
                    if 340 <= gap <= 390:
                        prev = r
                        break
        if prev is not None:
            if _v(s["Sales"]) is not None and _v(prev["Sales"]):
                f["rg"] = s["Sales"] / prev["Sales"] - 1
            if _v(s["NP"]) is not None and _v(prev["NP"]) and prev["NP"] > 0:
                f["eg"] = s["NP"] / prev["NP"] - 1
        elif fy and _v(fy[-1]["Sales"]):
            fs = _v(s["NxFSales"]) if s["CurPerType"] == "FY" else _v(s["FSales"])
            if fs is not None:
                f["rg"] = fs / fy[-1]["Sales"] - 1
    return f


def main() -> None:
    import yfinance as yf
    fins = load_fins()
    disc_dates = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    topix = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    fund_now = json.load(open("data/fundamentals.json", encoding="utf-8"))
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) >= bw.WARMUP + 20:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()})

    def fund_at(d: str) -> dict:
        out = {}
        for code in clean:
            ds = disc_dates.get(code)
            if not ds:
                continue
            n = bisect.bisect_right(ds, d)
            if n:
                out[code] = pit_fund(fins[code][:n])
        return out

    # 期間：プライム銘柄の8割以上に通期決算がそろった日 〜 決算データの最後の開示日
    all_d = sorted({r["DiscDate"] for v in fins.values() for r in v})
    last_disc = all_d[-1]
    w0 = None
    first_px = min(df.index[bw.WARMUP].strftime("%Y-%m-%d") for df in clean.values())
    for d in [x for x in all_d if x >= first_px][::5]:
        have = sum(1 for c in clean if any(r["CurPerType"] == "FY" and r["DiscDate"] <= d for r in fins.get(c, [])))
        if have >= 0.8 * len(clean):
            w0 = d
            break
    print(f"検証期間: {w0} 〜 {last_disc}（決算データ {len(fins)}社）")

    # 合図（ファンダメンタル以外の条件まで）
    cand = {}
    for code, df in clean.items():
        c = df["Close"].to_numpy(float)
        v = df["Volume"].to_numpy(float)
        t = topix.reindex(df.index).ffill().to_numpy()
        rows = []
        for i in bw.find_signals(df)["BW"]:
            sd = df.index[i].strftime("%Y-%m-%d")
            if not (w0 <= sd <= last_disc) or v[i] < bw.MIN_VOLUME or i < L or t[i - L] != t[i - L]:
                continue
            mr = t[i] / t[i - L] - 1
            if c[i] / c[i - L] - 1 - mr <= -G / 100:
                rows.append((i, sd))
        cand[code] = rows
    need = sorted({sd for rows in cand.values() for _, sd in rows})
    rank_pit, rank_now = {}, {}
    for sd in need:
        ts = pd.Timestamp(sd)
        cm = {c: float(x) for c, x in closes.loc[ts].items() if x == x}
        rank_pit[sd] = k.fund_ranks(master, clean, fund_at(sd), close_map=cm)
        rank_now[sd] = k.fund_ranks(master, clean, fund_now, close_map=cm)
    print(f"ランクを計算した日数: {len(need)}")

    raw = {f"{m}__{ex}": [] for m in ("pit", "now", "nofund") for ex in ("m2", "lk25c")}
    agree = {"both": 0, "pit_only": 0, "now_only": 0, "neither": 0}
    for code, df in clean.items():
        cache: dict = {}
        for _, sd in cand[code]:
            a = (rank_pit[sd].get(code) is not None and rank_pit[sd][code] <= FMAX)
            b = (rank_now[sd].get(code) is not None and rank_now[sd][code] <= FMAX)
            agree["both" if a and b else "pit_only" if a else "now_only" if b else "neither"] += 1
        for m in ("pit", "now", "nofund"):
            for ex in ("m2", "lk25c"):
                busy = ""
                for i, sd in cand[code]:
                    if sd <= busy:
                        continue
                    if m != "nofund":
                        rk = (rank_pit if m == "pit" else rank_now)[sd].get(code)
                        if rk is None or rk > FMAX:
                            continue
                    t = b2.simulate(code, df, i, ex, cache)
                    raw[f"{m}__{ex}"].append(t)
                    busy = t["exit_date"]
    res = {}
    for name, tr in raw.items():
        taken = sb.portfolio(tr)
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_rel_pit_trades_{name}{SUF}.csv"), index=False, encoding="utf-8-sig")
        st = sb.stats(taken)
        if taken:
            d = pd.DataFrame(taken)
            d["net"] = d["pnl"] - d["notional"] * COST
            d["m"] = d["exit_date"].str[:7]
            st["after_cost"] = int(d["net"].sum())
            st["avg_ret_pct"] = round(float(d["ret_pct"].mean()), 2)
            st["by_half_after_cost"] = {h: int(g["net"].sum()) for h, g in
                                        d.groupby(d["exit_date"].str[:4] + np.where(d["exit_date"].str[5:7] <= "06", "前半", "後半"))}
        res[name] = st
        print(f"{name:14s} {st.get('trades', 0):4d}回 損益 {st.get('total_pnl', 0):>10,}円 費用後 {st.get('after_cost', 0):>10,}円 "
              f"勝率 {st.get('win_rate')}%")
    print("ランク4以下の判定の一致:", agree)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"short_rel_pit{SUF}.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "window": [w0, last_disc],
                   "companies_with_fins": len(fins), "rank_agreement": agree, "results": res}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
