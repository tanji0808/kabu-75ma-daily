#!/usr/bin/env python3
"""
空売り版 バンドウォーク崩れ のバックテスト（過去約5年・東証プライム）

ボリンジャーバンドは25日・スクリーナーと同じ計算。

売り（空売り）の合図
  BW : バンドウォーク崩れ
       - 合図の前の10営業日のうち7日以上、終値が +1σ より上（バンドウォークで上昇）
       - 前日の終値は +1σ より上で、その日の終値が +1σ を下回った
  DT : ダブルトップのあとの +1σ割れ
       - 直近60営業日のうちに、10日続けて7日以上 +1σ より上の時期がある（バンドウォークで上昇）
       - 直近60営業日に、高さの差が3%以内の山が2つ（10営業日以上離れ、2つ目は直近20営業日以内）、
         山と山の間の谷が低い方の山より3%以上下（ダブルトップ）
       - 前日の終値は +1σ より上で、その日の終値が +1σ を下回った
  _f4 / _f2 : さらに、合図の日のファンダメンタルランク（スクリーナーと同じ10点満点）が4以下／2以下
        ※ 財務データは今の値しかないため、PER・PBRなどは合図の日の株価で計算し、利益・ROEなどは今の値を使う
  出来高50万株以上。合図の翌営業日の始値で空売りする。

買い戻し
  利確 : 終値が −1σ を下回ったら、翌営業日の始値で買い戻す
  損切り : 終値が +1σ を上回ったら、翌営業日の始値で買い戻す
  最長60営業日で翌営業日の始値で買い戻す。

資金 : 1回あたり約100万円分（100株単位）。同時に最大10銘柄、建玉1,000万円まで（保証金300万円）。
       貸株料 年1.1%。売買手数料・逆日歩・配当相当額・税金は含めない。
注意 : 今のプライム銘柄だけで検証している（途中で上場廃止になった銘柄は入っていない）。
出力 : backtest/short_bw_summary.json, backtest/short_bw_trades_<パターン>.csv
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

PERIOD = os.environ.get("BT_PERIOD", "6y")
MIN_VOLUME = 500_000
LOT_YEN = 1_000_000
LEND_FEE = 0.011
MAX_HOLD = 60
WARMUP = 90
FRANK_MAX = 4
OUT = "backtest"

PATTERNS = {
    "BW": "バンドウォーク崩れ（+1σ割れで売り）",
    "BW_f4": "バンドウォーク崩れ＋ファンダメンタル4以下",
    "BW_f2": "バンドウォーク崩れ＋ファンダメンタル2以下",
    "DT": "ダブルトップ後の+1σ割れ",
    "DT_f4": "ダブルトップ後の+1σ割れ＋ファンダメンタル4以下",
    "DT_f2": "ダブルトップ後の+1σ割れ＋ファンダメンタル2以下",
}


def bands(c: np.ndarray):
    s = pd.Series(c)
    m = s.rolling(25).mean()
    sd = s.rolling(25).std(ddof=0)
    return (m + sd).to_numpy(), (m - sd).to_numpy()


def double_top(h: np.ndarray, l: np.ndarray, i: int) -> bool:
    lo = max(5, i - 60)
    peaks = [j for j in range(lo, i - 1) if h[j] == h[max(0, j - 5):min(i + 1, j + 6)].max()]
    for b in reversed(peaks):
        if b < i - 20:
            break
        for a in peaks:
            if b - a < 10:
                continue
            if abs(h[b] / h[a] - 1) > 0.03:
                continue
            if l[a:b + 1].min() <= min(h[a], h[b]) * 0.97:
                return True
    return False


def find_signals(df: pd.DataFrame) -> dict[str, list[int]]:
    c, h, l, v = (df[x].to_numpy(float) for x in ("Close", "High", "Low", "Volume"))
    u1, _ = bands(c)
    above = c > u1
    walk10 = pd.Series(above.astype(int)).rolling(10).sum().to_numpy()
    out = {"BW": [], "DT": []}
    for i in range(WARMUP, len(c) - 1):
        if np.isnan(u1[i]) or np.isnan(u1[i - 1]) or v[i] < MIN_VOLUME:
            continue
        if not (c[i - 1] > u1[i - 1] and c[i] < u1[i]):
            continue
        if walk10[i - 1] >= 7:
            out["BW"].append(i)
        if np.nanmax(walk10[i - 60:i]) >= 7 and double_top(h, l, i):
            out["DT"].append(i)
    return out


def simulate(code: str, df: pd.DataFrame, i: int) -> dict | None:
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    u1, d1 = bands(c)
    e = i + 1
    entry = o[e]
    shares = int(LOT_YEN // entry // 100 * 100)
    if shares <= 0:
        shares = 100
    n = len(c)
    exit_i = None
    for j in range(e, min(n - 1, e + MAX_HOLD)):
        if c[j] < d1[j]:
            exit_i, why = j + 1, "利確（−1σ割れ）"
            break
        if c[j] > u1[j]:
            exit_i, why = j + 1, "損切り（+1σ超え）"
            break
    if exit_i is None:
        if e + MAX_HOLD <= n - 1:
            exit_i, why = e + MAX_HOLD, "期限"
            px = o[exit_i]
        else:
            exit_i, why = n - 1, "保有中（最終日の終値で評価）"
            px = c[n - 1]
    else:
        px = o[exit_i]
    days = (df.index[exit_i] - df.index[e]).days
    fee = entry * shares * LEND_FEE * max(days, 1) / 365
    return {"code": code, "signal": df.index[i].strftime("%Y-%m-%d"),
            "entry_date": df.index[e].strftime("%Y-%m-%d"), "exit_date": df.index[exit_i].strftime("%Y-%m-%d"),
            "entry": round(entry, 1), "exit": round(px, 1), "shares": shares, "notional": round(entry * shares),
            "days": days, "why": why, "pnl": round((entry - px) * shares - fee), "fee": round(fee),
            "ret_pct": round((entry - px) / entry * 100, 2)}


def main() -> None:
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    fpath = "data/fundamentals.json"
    fund = json.load(open(fpath, encoding="utf-8")) if os.path.exists(fpath) else {}
    os.makedirs(OUT, exist_ok=True)
    clean = {}
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) >= WARMUP + 20:
            clean[code] = df
    closes = pd.DataFrame({c: d["Close"] for c, d in clean.items()})
    raw = {p: [] for p in PATTERNS}
    sigs_all = {}
    for code, df in clean.items():
        sigs_all[code] = find_signals(df)
    # 合図の日ごとに、その日の終値でファンダメンタルランクを計算する（同じ日は使い回す）
    need = sorted({clean[cd].index[i] for cd, s in sigs_all.items() for kk in s for i in s[kk]})
    rank_on = {}
    for d in need:
        row = closes.loc[d] if d in closes.index else None
        if row is None:
            continue
        cm = {c: float(v) for c, v in row.items() if v == v}
        rank_on[d] = k.fund_ranks(master, clean, fund, close_map=cm)
    print(f"ファンダメンタルランクを計算した日数: {len(rank_on)}")
    nsig = {"BW": 0, "DT": 0}
    for code, df in clean.items():
        for base in ("BW", "DT"):
            idx = sigs_all[code][base]
            nsig[base] += len(idx)
            for p in (base, base + "_f4", base + "_f2"):
                busy = ""
                for i in idx:
                    sd = df.index[i].strftime("%Y-%m-%d")
                    if sd <= busy:
                        continue
                    if "_f" in p:
                        r = rank_on.get(df.index[i], {}).get(code)
                        if r is None or r > int(p.split("_f")[1]):
                            continue
                    t = simulate(code, df, i)
                    if t:
                        raw[p].append(t)
                        busy = t["exit_date"]
    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(clean), "signals": nsig,
               "rules": {"lot_yen": LOT_YEN, "max_pos": sb.MAX_POS, "max_notional": sb.MAX_NOTIONAL,
                         "lend_fee": LEND_FEE, "max_hold": MAX_HOLD, "frank_max": FRANK_MAX}, "patterns": {}}
    for p, name in PATTERNS.items():
        taken = sb.portfolio(raw[p])
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_bw_trades_{p}.csv"), index=False, encoding="utf-8-sig")
        st = sb.stats(taken)
        allst = sb.stats(raw[p])
        if raw[p]:
            allst["avg_ret_pct"] = round(float(np.mean([t["ret_pct"] for t in raw[p]])), 2)
        if taken:
            st["avg_ret_pct"] = round(float(np.mean([t["ret_pct"] for t in taken])), 2)
        summary["patterns"][p] = {"name": name, "all_signals": allst, "portfolio": st}
        print(f"{p} {name}: {st.get('trades')}回 損益 {st.get('total_pnl', 0):,}円 勝率 {st.get('win_rate')}%")
    with open(os.path.join(OUT, "short_bw_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
