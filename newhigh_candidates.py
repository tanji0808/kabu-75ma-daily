#!/usr/bin/env python3
"""
新高値ブレイクの候補をすべて洗い出し、いろいろな売り方の結果と判断材料を1つの表にまとめる
（東証プライム＋上場廃止銘柄・約10年）。この表で6人の手法の条件を手元で試す（2017〜2021年で選び、2022〜2026年で確かめる）。

候補（合図の日 t）: 終値が直近250営業日（52週）の高値を初めて上に抜けた日。
  その高値（ピボット）が付いたのが5営業日以上前（＝少なくとも1週間の保ち合い・ベースがある）。
  20日平均の売買代金3,000万円以上。翌営業日の始値で買う。
判断材料（合図の日の時点で分かっていたもの）
  ブレイク   : base_len（ベースの日数）, base_depth（ベースの深さ％）, brk（ピボットからの上抜け幅％）, ath（10年高値も更新）,
               hist_days（データの日数）, vratio（出来高÷50日平均）, vdry（直前10日の出来高÷50日平均）,
               tight10（直前10日の値幅÷終値％）, vcp（直前10日の値幅 ÷ ベース全体の値幅）, gap（当日の始値の窓％）
  トレンド   : ma50/ma150/ma200 の並び（trend_tmpl＝ミネルヴィニのトレンドテンプレート）, ma200up（200日線が20日前より上）,
               wk30up（30週線＝150日線が上向き）, above_low（52週安値からの上昇率％）, dev25, rsi
  強さ       : rs_pct（相対的株価強度のパーセンタイル。3・6・9・12か月の騰落率の加重）, rel20/rel60（TOPIXとの差％）
  市場       : tpx_ma200（TOPIXが200日線より上）, tpx20, tpx_new（TOPIXも52週高値圏）
  決算・需給 : rank, eg, rg, roe, rev, cap, dsince, lsh, lchg, mratio（buy_candidates.py と同じ）
売り方（名前_ret / _exit / _days。最長120営業日。損切りは日中の安値で判定し、窓を開けて下げたらその日の始値）
  s8      : −8%損切りのみ（あとは120日で手じまい）
  s8tp20 / s8tp25 : −8%損切り＋20%・25%で利確
  s8ma25 / s8ma50 : −8%損切り＋終値が25日線・50日線を割ったら翌日の始値で売る
  s8tr10 / s8tr15 : −8%損切り＋最高値（終値）から10%・15%下がったら売る
  s8be50  : −8%損切り。終値で+10%になったら損切りを買値へ。あとは50日線割れで売る
  s6ma50 / s10ma50 : 損切り−6%・−10%＋50日線割れ
  box50   : 損切りはピボットの3%下（ダーバスの箱に戻ったら売る。ただし最大−10%）＋50日線割れ
  cur     : 今の買いのやり方（−1σ割れで損切り・引き上げ式）
出力 : backtest/newhigh_candidates_11y.csv.gz
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
import buy_candidates as bc  # noqa: E402
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_candidates as sc  # noqa: E402
import short_rel_pit as sp  # noqa: E402

PERIOD = os.environ["BT_PERIOD"]
MAX_HOLD = 120
EXITS = {"s8": dict(stop=0.08), "s8tp20": dict(stop=0.08, tp=0.20), "s8tp25": dict(stop=0.08, tp=0.25),
         "s8ma25": dict(stop=0.08, ma=25), "s8ma50": dict(stop=0.08, ma=50), "s8tr10": dict(stop=0.08, trail=0.10),
         "s8tr15": dict(stop=0.08, trail=0.15), "s8be50": dict(stop=0.08, ma=50, be=0.10),
         "s6ma50": dict(stop=0.06, ma=50), "s10ma50": dict(stop=0.10, ma=50), "box50": dict(stop=0.10, ma=50, box=True),
         "s6tp25": dict(stop=0.06, tp=0.25), "s10tp25": dict(stop=0.10, tp=0.25), "s8tp30": dict(stop=0.08, tp=0.30),
         "s8tp40": dict(stop=0.08, tp=0.40), "s5tp15": dict(stop=0.05, tp=0.15), "s8tp25h60": dict(stop=0.08, tp=0.25, hold=60),
         "s8tp25be": dict(stop=0.08, tp=0.25, be=0.10), "s8tp25c": dict(stop=0.08, tp=0.25, entry="close"),
         "s8tp25p": dict(stop=0.08, tp=0.25, entry="pull"), "s8tp25tr": dict(stop=0.08, tp=0.25, trail=0.15)}


def sim(a: dict, t: int, stop: float, tp=None, ma=None, trail=None, be=None, box=False, hold=None, entry="open"):
    """entry: open=翌日の始値で買う／close=合図の日の終値で買う（翌日から判定）／pull=10日以内にピボットまで押したらそこで買う（初押し）"""
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    n = len(c)
    mh = hold or MAX_HOLD
    e = t + 1
    if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
        return None
    if entry == "close":
        px0, start = c[t], t + 1
    elif entry == "pull":
        pv = a["pivot"][t]
        px0 = start = None
        for j in range(t + 1, min(n, t + 11)):
            if l[j] <= pv:
                px0, start = min(o[j], pv), j
                break
        if px0 is None:
            return {"ret": np.nan, "exit": None, "days": None, "reason": "no_pull"} if t + 11 < n else None
    else:
        px0, start = o[e], e
    res = _run(a, start, px0, stop, tp, ma, trail, be, box, t, mh, entry == "close")
    if res and entry != "open":
        res["in"] = a["dates"][start if entry == "pull" else t]
    return res


def _run(a, e, entry, stop, tp, ma, trail, be, box, t, mh, full_first):
    """full_first: 1日目もその日の始値から判定する（終値で買ったとき）"""
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    n = len(c)
    sl = entry * (1 - stop)
    if box:
        sl = max(sl, a["pivot"][t] * 0.97) if a["pivot"][t] * 0.97 < entry else sl
    mav = a.get(f"ma{ma}") if ma else None
    hc = entry
    px = xd = None
    reason = None
    for j in range(e, min(n, e + mh)):
        if (j > e or full_first) and o[j] <= sl:
            px, xd, reason = o[j], j, "stop"
            break
        if l[j] <= sl:
            px, xd, reason = sl, j, "stop"
            break
        if tp and h[j] >= entry * (1 + tp):
            px, xd, reason = max(o[j] if (j > e or full_first) else entry, entry * (1 + tp)), j, "tp"
            break
        hc = max(hc, c[j])
        if be and c[j] >= entry * (1 + be):
            sl = max(sl, entry)
        if trail:
            sl = max(sl, hc * (1 - trail))
        if mav is not None and c[j] < mav[j]:
            if j + 1 < n:
                px, xd, reason = o[j + 1], j + 1, "ma"
            else:
                return None
            break
    if px is None:
        last = e + mh - 1
        if last + 1 >= n:
            return None                                     # 検証期間の終わりで持ったまま
        px, xd, reason = o[last + 1], last + 1, "time"
    return {"ret": round((px / entry - 1) * 100, 2), "exit": a["dates"][xd], "days": int(xd - e), "reason": reason}


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
    # 相対的株価強度（IBD風）: 3か月×2 + 6・9・12か月 の騰落率を全銘柄の中で順位（0〜100）
    rs_raw = 0.4 * closes.pct_change(63, fill_method=None) + 0.2 * closes.pct_change(126, fill_method=None) \
        + 0.2 * closes.pct_change(189, fill_method=None) + 0.2 * closes.pct_change(252, fill_method=None)
    rs_pct = rs_raw.rank(axis=1, pct=True) * 100
    tpx = topix
    t200 = tpx.rolling(200).mean()
    t_hi = tpx.rolling(250).max()
    cand, arrays = [], {}
    for code, df in clean.items():
        c, h, l, v, o = (df[x].to_numpy(float) for x in ("Close", "High", "Low", "Volume", "Open"))
        n = len(c)
        s = pd.Series(c)
        ma = {m: s.rolling(m).mean().to_numpy() for m in (25, 50, 150, 200)}
        sd = s.rolling(25).std(ddof=0).to_numpy()
        hs = pd.Series(h)
        hi250 = hs.rolling(250).max().shift(1).to_numpy()          # 前日までの52週高値（ピボット）
        hi_all = hs.cummax().shift(1).to_numpy()
        lo250 = pd.Series(l).rolling(250).min().to_numpy()
        vavg50 = pd.Series(v).rolling(50).mean().shift(1).to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().to_numpy()
        d_ = s.diff()
        up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn)).to_numpy()
        tp_ = tpx.reindex(df.index).ffill().to_numpy()
        t200v = t200.reindex(df.index).ffill().to_numpy()
        thv = t_hi.reindex(df.index).ffill().to_numpy()
        rsv = rs_pct[code].reindex(df.index).to_numpy() if code in rs_pct else np.full(n, np.nan)
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        arrays[code] = {"o": o, "h": h, "l": l, "c": c, "ma25": ma[25], "ma50": ma[50], "pivot": hi250, "dates": dates,
                        "d1": ma[25] - sd, "u1": ma[25] + sd, "m25": ma[25]}
        for t in range(260, n - 1):
            if not (c[t] > hi250[t] and c[t - 1] <= hi250[t - 1]) or tv20[t] < 3e7:
                continue
            win = h[t - 250:t]
            pi = t - 250 + int(np.argmax(win))
            base_len = t - pi
            if base_len < 5:
                continue
            base_lo = l[pi:t].min()
            r10 = (h[t - 10:t].max() - l[t - 10:t].min())
            rb = (h[pi:t].max() - base_lo)
            row = {"code": code, "t": t, "signal": dates[t], "close": c[t],
                   "base_len": base_len, "base_depth": (1 - base_lo / hi250[t]) * 100, "brk": (c[t] / hi250[t] - 1) * 100,
                   "ath": bool(c[t] > hi_all[t]), "hist_days": t, "vratio": v[t] / vavg50[t] if vavg50[t] > 0 else np.nan,
                   "vdry": v[t - 10:t].mean() / vavg50[t] if vavg50[t] > 0 else np.nan, "tight10": r10 / c[t] * 100,
                   "vcp": r10 / rb if rb > 0 else np.nan, "gap": (o[t] / c[t - 1] - 1) * 100, "tv20": tv20[t] / 1e8,
                   "trend_tmpl": bool(c[t] > ma[50][t] > ma[150][t] > ma[200][t]), "ma200up": bool(ma[200][t] > ma[200][t - 20]),
                   "wk30up": bool(ma[150][t] > ma[150][t - 5]), "above_low": (c[t] / lo250[t] - 1) * 100,
                   "dev25": (c[t] / ma[25][t] - 1) * 100, "rsi": rsi[t], "rs_pct": rsv[t],
                   "tpx_ma200": bool(tp_[t] > t200v[t]), "tpx20": (tp_[t] / tp_[t - 20] - 1) * 100,
                   "tpx_new": bool(tp_[t] >= thv[t] * 0.97)}
            for L in (20, 60):
                if tp_[t - L] == tp_[t - L]:
                    row[f"rel{L}"] = (c[t] / c[t - L] - 1 - (tp_[t] / tp_[t - L] - 1)) * 100
            cand.append(row)
    print(f"新高値ブレイクの候補: {len(cand)}件")
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
        e = r["t"] + 1
        r["entry"] = a["dates"][e] if e < len(a["dates"]) else None
        for name, prm in EXITS.items():
            res = sim(a, r["t"], **prm)
            if res:
                for x, val in res.items():
                    r[f"{name}_{x}"] = val
        res = bc.sim(a, r["t"], None, False, "u1")
        if res:
            r["cur_ret"], r["cur_exit"], r["cur_days"] = res["ret"], res["exit"], res["days"]
    out = pd.DataFrame(cand).drop(columns=["t"])
    os.makedirs("backtest", exist_ok=True)
    out.to_csv("backtest/newhigh_candidates_11y.csv.gz", index=False, compression="gzip")
    print("保存しました")


if __name__ == "__main__":
    main()
