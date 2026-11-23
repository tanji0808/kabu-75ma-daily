#!/usr/bin/env python3
"""
買い（75日線ブレイク）の候補をすべて洗い出し、いろいろな売り方の結果と判断材料を1つの表にまとめる
（10年・上場廃止銘柄を含む）。この表で条件・売り方の組み合わせを手元で試す（2017〜2021年で選び、2022〜2026年で確かめる）。

候補 : backtest.signal_days（スクリーナーの「本日突破」と同じ。急騰除外の条件はかけず、材料として残す）。
判断材料（合図の日の時点で分かっていたもの）
  up5 : 5日の騰落率（％）、rsi : RSI(14)（backtest.py と同じ計算）
  rel10, rel20 : 10日・20日の騰落率 − TOPIX(1306)の騰落率（％）
  tpx_ma75, tpx_ma200 : TOPIXが75日線・200日線より上か、tpx20 : TOPIXの20日の騰落率（％）
  vratio : 出来高 ÷ 20日平均、ma25up : 25日線が5日前より上か、gap75 : 75日線からの乖離（％）
  drop : 半年の高値からの下落率（％）
  rank, eg, rg, roe, rev, cap : その時点の決算から（short_candidates.py と同じ）
  lsh, lchg, mratio : 信用残（short_candidates.py と同じ）
  dsince : 直近の決算発表からの日数、eg_last : その決算の増益率
売り方（どれも最初の損切りは −1σ割れ、最長120営業日。名前_ret / _exit / _days）
  cur   : 今のやり方（損切り幅ぶん上がったら利益確保ラインへ、+1σより上なら+1σまで引き上げ）
  t5, t10 : cur に加えて、5日・10日たっても一度も+1σを超えなければ翌日の始値で売る
  half  : cur に加えて、初めて+1σを超えた翌日の始値で半分を売る
  ma25  : 利益確保ラインに乗ったあとは、25日線まで引き上げる（+1σの代わり）
  low10 : 利益確保ラインに乗ったあとは、直近10日の安値まで引き上げる
  t10half : t10 と half の両方
出力 : backtest/buy_candidates_11y.csv.gz
"""

from __future__ import annotations

import bisect
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("K_DELISTED", "1")
os.environ.setdefault("BT_PERIOD", "11y")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_candidates as sc  # noqa: E402
import short_rel_pit as sp  # noqa: E402

PERIOD = os.environ["BT_PERIOD"]
LOOSE = os.environ.get("BUY_LOOSE") == "1"   # 「約3か月の下落」の縛りを外して全部出し、期間ごとの下落の材料を付ける
LB = (20, 40, 60, 90, 120, 180, 250)
if LOOSE:
    k.MA_DECLINE_MIN, k.BELOW_RATIO_MIN = -9.0, 0.0
EXITS = {"cur": (None, False, "u1"), "t5": (5, False, "u1"), "t10": (10, False, "u1"), "half": (None, True, "u1"),
         "ma25": (None, False, "ma25"), "low10": (None, False, "low10"), "t10half": (10, True, "u1")}


def sim(a: dict, t: int, time_stop, half: bool, trail: str):
    o, c, l, d1, u1, m25 = a["o"], a["c"], a["l"], a["d1"], a["u1"], a["m25"]
    n = len(c)
    e = t + 1
    if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
        return None
    entry = o[e]
    R = entry - d1[t] if d1[t] < entry else entry * 0.03
    lock = stop = None
    touched, half_px, x_sig, reason = False, None, None, None
    for j in range(e, min(n, e + bt.MAX_HOLD)):
        if half and touched and half_px is None and j > e:
            half_px = o[j]                                  # 初めて+1σを超えた翌日の始値で半分
        if stop is None:
            if c[j] < d1[j]:
                x_sig, reason = j, "m1s"
                break
            if time_stop and not touched and c[j] <= u1[j] and j - e + 1 >= time_stop:
                x_sig, reason = j, "time_stop"
                break
            if c[j] >= entry + R:
                lock = stop = entry + R
        else:
            if c[j] < stop:
                x_sig, reason = j, ("lock" if stop == lock else "trail")
                break
            if trail == "u1" and c[j] > u1[j]:
                stop = max(stop, u1[j])
            elif trail == "ma25":
                stop = max(stop, m25[j])
            elif trail == "low10":
                stop = max(stop, l[max(e, j - 9):j + 1].min())
        touched = touched or c[j] > u1[j]
    if x_sig is None:
        if e + bt.MAX_HOLD - 1 < n:
            x_sig, reason = e + bt.MAX_HOLD - 1, "time"
        else:
            return None                                     # 検証期間の終わりで持ったまま
    x = x_sig + 1 if x_sig + 1 < n else x_sig
    px = o[x] if x_sig + 1 < n else c[x_sig]
    if half_px is not None:
        px = (half_px + px) / 2
    return {"ret": round((px / entry - 1) * 100, 2), "exit": a["dates"][x], "entry": a["dates"][e], "days": int(x - e),
            "reason": reason}


# 利確の工夫（引き上げ式の続き）。最初の損切り（−1σ割れ）と、損切り幅ぶん上がったら利益確保、までは今と同じ。
#   line  : 引き上げる線（u1=+1σ／u15=+1.5σ）
#   after2: 終値が+2σを超えたあと、さらに守る方法（back2=終値が+2σを割ったら売る／low3・low5=直近3・5日の安値まで引き上げ／
#           u15=+1.5σまで引き上げ／pk5・pk8=最高値（終値）から5%・8%まで引き上げ）
#   tp2   : +2σで利確（limit=前日の+2σに指値して全部／half=そこで半分だけ）
#   rsi   : RSIが hi 以上になったあと lo を下回ったら売る
#   chand : 最高値（終値）−ATR(14)×k まで引き上げ（シャンデリア）
#   vclx  : +2σより上で出来高が20日平均の3倍以上の日の翌日に売る（出来高の天井）
EXITS2 = {"x_u15": dict(line="u15"), "x_back2": dict(after2="back2"), "x_low3": dict(after2="low3"), "x_low5": dict(after2="low5"),
          "x_a2u15": dict(after2="u15"), "x_pk5": dict(after2="pk5"), "x_pk8": dict(after2="pk8"),
          "x_tp2": dict(tp2="limit"), "x_tp2half": dict(tp2="half"), "x_rsi": dict(rsi=(75, 70)), "x_rsi80": dict(rsi=(80, 70)),
          "x_ch3": dict(chand=3.0), "x_ch2": dict(chand=2.0), "x_vclx": dict(vclx=True),
          "x_back2rsi": dict(after2="back2", rsi=(75, 70)), "x_low5half": dict(after2="low5", tp2="half"),
          "x_u15low5": dict(line="u15", after2="low5")}


def sim2(a: dict, t: int, line="u1", after2=None, tp2=None, rsi=None, chand=None, vclx=False):
    o, h, c, l, d1, u1, u2, u15 = a["o"], a["h"], a["c"], a["l"], a["d1"], a["u1"], a["u2"], a["u15"]
    n = len(c)
    e = t + 1
    if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
        return None
    entry = o[e]
    R = entry - d1[t] if d1[t] < entry else entry * 0.03
    lock = stop = None
    x_sig = reason = None
    half_px = None
    hc = entry
    above2 = False
    rsi_hi = False
    tl = u15 if line == "u15" else u1
    for j in range(e, min(n, e + bt.MAX_HOLD)):
        # +2σの指値（前日の+2σ）。日中に届いたら約定
        if tp2 and j > e and np.isfinite(u2[j - 1]) and h[j] >= u2[j - 1] and entry < u2[j - 1]:
            px_ = max(o[j], u2[j - 1])
            if tp2 == "limit":
                return {"ret": round((px_ / entry - 1) * 100, 2), "exit": a["dates"][j], "days": int(j - e), "reason": "tp2"}
            if half_px is None:
                half_px = px_
        hc = max(hc, c[j])
        if stop is None:
            if c[j] < d1[j]:
                x_sig, reason = j, "m1s"
                break
            if c[j] >= entry + R:
                lock = stop = entry + R
        else:
            if c[j] < stop:
                x_sig, reason = j, ("lock" if stop == lock else "trail")
                break
            if c[j] > tl[j]:
                stop = max(stop, tl[j])
            if above2 and after2:
                if after2 == "back2" and c[j] < u2[j]:
                    x_sig, reason = j, "back2"
                    break
                if after2 == "low3":
                    stop = max(stop, l[max(e, j - 2):j + 1].min())
                elif after2 == "low5":
                    stop = max(stop, l[max(e, j - 4):j + 1].min())
                elif after2 == "u15":
                    stop = max(stop, u15[j])
                elif after2 in ("pk5", "pk8"):
                    stop = max(stop, hc * (0.95 if after2 == "pk5" else 0.92))
            if chand:
                stop = max(stop, hc - chand * a["atr"][j])
        if rsi:
            if a["rsi"][j] >= rsi[0]:
                rsi_hi = True
            elif rsi_hi and a["rsi"][j] < rsi[1] and c[j] > entry:
                x_sig, reason = j, "rsi"
                break
        if vclx and c[j] > u2[j] and a["vavg"][j] > 0 and a["v"][j] >= 3 * a["vavg"][j]:
            x_sig, reason = j, "vclx"
            break
        above2 = above2 or c[j] > u2[j]
    if x_sig is None:
        if e + bt.MAX_HOLD - 1 < n:
            x_sig, reason = e + bt.MAX_HOLD - 1, "time"
        else:
            return None
    x = x_sig + 1 if x_sig + 1 < n else x_sig
    px = o[x] if x_sig + 1 < n else c[x_sig]
    if half_px is not None:
        px = (half_px + px) / 2
    return {"ret": round((px / entry - 1) * 100, 2), "exit": a["dates"][x], "days": int(x - e), "reason": reason}


def main() -> None:
    import yfinance as yf
    topix = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    topix.index = pd.to_datetime(topix.index).tz_localize(None).normalize()
    fins = sp.load_fins()
    disc = {c: [r["DiscDate"] for r in v] for c, v in fins.items()}
    mg = sc.margin_table()
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        if len(df) >= 260:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()}).sort_index()
    t75, t200 = topix.rolling(75).mean(), topix.rolling(200).mean()
    cand = []
    arrays = {}
    for code, df in clean.items():
        sigs = bt.signal_days(df, "")
        if not len(sigs):
            continue
        c, h, l, v = (df[x].to_numpy(float) for x in ("Close", "High", "Low", "Volume"))
        s = pd.Series(c)
        m25 = s.rolling(25).mean()
        sd = s.rolling(25).std(ddof=0)
        ma75 = s.rolling(75).mean().to_numpy()
        d_ = s.diff()
        up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn)).to_numpy()
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        tp = topix.reindex(df.index).ffill()
        tpv = tp.to_numpy()
        b75 = (tp > t75.reindex(df.index).ffill()).to_numpy()
        b200 = (tp > t200.reindex(df.index).ffill()).to_numpy()
        tr_ = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        atr = np.r_[np.nan, pd.Series(tr_).rolling(14).mean().to_numpy()]
        arrays[code] = {"o": df["Open"].to_numpy(float), "h": h, "c": c, "l": l, "d1": (m25 - sd).to_numpy(),
                        "u1": (m25 + sd).to_numpy(), "u15": (m25 + 1.5 * sd).to_numpy(), "u2": (m25 + 2 * sd).to_numpy(),
                        "m25": m25.to_numpy(), "rsi": rsi, "atr": atr, "v": v, "vavg": pd.Series(v).rolling(20).mean().to_numpy(),
                        "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        m25v = m25.to_numpy()
        for t in sigs:
            row = {"code": code, "t": int(t), "signal": df.index[t].strftime("%Y-%m-%d"),
                   "up5": (c[t] / c[t - 5] - 1) * 100, "rsi": rsi[t], "vratio": v[t] / vavg[t] if vavg[t] > 0 else np.nan,
                   "ma25up": bool(m25v[t] > m25v[t - 5]), "gap75": (c[t] / ma75[t] - 1) * 100,
                   "drop": (1 - l[max(0, t - 120):t + 1].min() / h[max(0, t - 120):t + 1].max()) * 100,
                   "tpx_ma75": bool(b75[t]), "tpx_ma200": bool(b200[t]),
                   "tpx20": (tpv[t] / tpv[t - 20] - 1) * 100 if tpv[t - 20] == tpv[t - 20] else np.nan}
            if LOOSE:
                below = c < ma75
                for dd in LB:
                    if t - dd >= 74:
                        row[f"dec{dd}"] = (ma75[t - dd] / ma75[t] - 1) * 100
                        row[f"below{dd}"] = float(below[t - dd:t - 5].mean())
            for L in (10, 20):
                if tpv[t - L] == tpv[t - L]:
                    row[f"rel{L}"] = (c[t] / c[t - L] - 1 - (tpv[t] / tpv[t - L] - 1)) * 100
            cand.append(row)
    print(f"買いの候補: {len(cand)}件")
    by_date: dict = {}
    for r in cand:
        by_date.setdefault(r["signal"], []).append(r)
    pit_cache: dict = {}
    for n_, (sd_, rows) in enumerate(sorted(by_date.items())):
        ts = pd.Timestamp(sd_)
        fund = {}
        for c in clean:
            nn = bisect.bisect_right(disc.get(c, []), sd_)
            if nn:
                if (c, nn) not in pit_cache:
                    pit_cache[(c, nn)] = sp.pit_fund(fins[c][:nn])
                fund[c] = pit_cache[(c, nn)]
        cm = {c: float(x) for c, x in closes.loc[ts].items() if x == x} if ts in closes.index else {}
        ranks = k.fund_ranks(master, clean, fund, close_map=cm)
        for r in rows:
            c = r["code"]
            r["rank"] = ranks.get(c)
            f = fund.get(c, {})
            for x in ("eg", "rg", "roe"):
                r[x] = f.get(x)
            nn = bisect.bisect_right(disc.get(c, []), sd_)
            recs = fins.get(c, [])[:nn]
            r["rev"] = sc.rev_flag(recs, sd_) if nn else 0
            stm = [x for x in recs if "FinancialStatements" in str(x["DocType"])]
            if stm:
                r["dsince"] = (ts - pd.Timestamp(stm[-1]["DiscDate"])).days
                r["eg_last"] = f.get("eg")
            sh = None
            for rec in reversed(recs):
                if rec["ShOutFY"] == rec["ShOutFY"] and rec["ShOutFY"]:
                    sh = rec["ShOutFY"] - (rec["TrShFY"] if rec["TrShFY"] == rec["TrShFY"] else 0)
                    break
            if sh and c in cm:
                r["cap"] = sh * cm[c] / 1e8
            g = mg.get(c)
            if g:
                j = bisect.bisect_right(g[0], sd_) - 1
                if j >= 0:
                    lv, sv = g[1][j], g[2][j]
                    r["mratio"] = lv / sv if sv > 0 else np.inf
                    if sh:
                        r["lsh"] = lv / sh * 100
                    if j >= 4 and g[1][j - 4] > 0:
                        r["lchg"] = lv / g[1][j - 4] - 1
        if n_ % 200 == 0:
            print(f"  {sd_}（{n_ + 1}/{len(by_date)}日）")
    for r in cand:
        a = arrays[r["code"]]
        for name, prm in EXITS2.items():
            res = sim2(a, r["t"], **prm)
            if res:
                for x, val in res.items():
                    r[f"{name}_{x}"] = val
        for name, (ts_, hf, tr) in EXITS.items():
            res = sim(a, r["t"], ts_, hf, tr)
            if res is None:
                continue
            for x, val in res.items():
                if x == "entry":
                    r["entry"] = val
                else:
                    r[f"{name}_{x}"] = val
    out = pd.DataFrame(cand).drop(columns=["t"])
    os.makedirs("backtest", exist_ok=True)
    out.to_csv("backtest/buy_candidates_loose_11y.csv.gz" if LOOSE else "backtest/buy_candidates_11y.csv.gz", index=False, compression="gzip")
    print(f"保存しました: {len(out)}件")


if __name__ == "__main__":
    main()
