#!/usr/bin/env python3
"""
75日線ブレイク手法のバックテスト（過去約5年・東証プライム）

買い : 3ヶ月以上の下落のあと、終値が75日線を下から上に抜けた日（スクリーナーの「本日突破」と同じ条件、
       出来高50万株以上）の「翌営業日の始値」で買う。
利確 : 買ったあと、終値がボリンジャーバンド(25日)の +1σ より上に乗り、そのあと終値が +1σ を下回ったら、
       翌営業日の始値で売る。
損切り（3通りを比較）:
   A : 終値が75日線を下回ったら、翌営業日の始値で売る
   B : 終値が買値の -8% 以下になったら、翌営業日の始値で売る
   C : 損切りなし
どれも、最長 120 営業日（約半年）持ったら翌営業日の始値で売る。
売買手数料・税金は含めない。同じ銘柄を持っている間は、その銘柄の新しい合図は無視する。

比較対象 : 同じ日に買って同じ日に売った場合の TOPIX 連動ETF（1306）の損益。

出力 : web/backtest.html（結果ページ）, backtest/summary.json, backtest/trades_<A|B|C>.csv
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

PERIOD = os.environ.get("BT_PERIOD", "6y")      # 指標の計算に半年ほど使うので、検証期間は約5年半
MIN_VOLUME = k.MIN_VOLUME
MAX_HOLD = 120
STOP_B = 0.08
VARIANTS = {
    "A": "75日線割れで損切り",
    "B": f"買値−{int(STOP_B * 100)}%で損切り",
    "C": "損切りなし",
}
REASON = {"take": "利確（+1σ割れ）", "stop": "損切り", "time": "期間満了"}


# ------------------------------------------------------------------ 合図
def signal_days(df: pd.DataFrame) -> np.ndarray:
    """スクリーナーの「本日突破」と同じ条件が成り立った日の位置を返す。"""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma = c.rolling(k.MA).mean()
    dd, cw = k.DOWN_DAYS, k.CROSS_WITHIN
    decline = (ma.shift(dd) / ma - 1) >= k.MA_DECLINE_MIN
    below = (c < ma).astype(float).where(ma.notna())
    below_ratio = below.shift(cw + 1).rolling(dd - cw).mean() >= k.BELOW_RATIO_MIN   # 区間 [t-60, t-6]
    drop = (1 - l.rolling(k.DROP_LOOKBACK + 1).min() / h.rolling(k.DROP_LOOKBACK + 1).max()) >= k.DROP_MIN
    gap = c / ma - 1
    cross = (c.shift(1) <= ma.shift(1)) & (c > ma) & (gap > 0) & (gap <= k.CROSS_GAP_MAX)
    vol = v >= MIN_VOLUME
    sig = decline & below_ratio & drop & cross & vol
    need = max(k.MA + dd, k.DROP_LOOKBACK) + 5          # スクリーナーと同じく、十分なデータがある日だけ
    idx = np.flatnonzero(sig.fillna(False).to_numpy())
    return idx[idx >= need - 1]


# ------------------------------------------------------------------ 売買
def simulate(df: pd.DataFrame, sigs: np.ndarray, variant: str) -> list[dict]:
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    ma75 = df["Close"].rolling(k.MA).mean().to_numpy()
    ma25 = df["Close"].rolling(25).mean()
    u1 = (ma25 + df["Close"].rolling(25).std(ddof=0)).to_numpy()
    dates = df.index
    n = len(df)
    trades, free_from = [], 0
    for t in sigs:
        if t < free_from:
            continue
        e = t + 1
        if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
            continue
        entry = o[e]
        touched, exit_sig, reason = False, None, None
        for j in range(e, min(n, e + MAX_HOLD)):
            if touched and c[j] < u1[j]:
                exit_sig, reason = j, "take"
                break
            if variant == "A" and c[j] < ma75[j]:
                exit_sig, reason = j, "stop"
                break
            if variant == "B" and c[j] <= entry * (1 - STOP_B):
                exit_sig, reason = j, "stop"
                break
            if c[j] > u1[j]:
                touched = True
        if exit_sig is None:
            if e + MAX_HOLD - 1 < n:
                exit_sig, reason = e + MAX_HOLD - 1, "time"
            else:
                free_from = n          # まだ持っている途中（検証期間の終わり）なので集計しない
                continue
        x = exit_sig + 1
        if x >= n:                      # 売りの合図が最終日に出た：その日の終値で売ったことにする
            x, px = exit_sig, c[exit_sig]
        else:
            px = o[x]
        trades.append({
            "signal": dates[t].strftime("%Y-%m-%d"), "entry": dates[e].strftime("%Y-%m-%d"),
            "exit": dates[x].strftime("%Y-%m-%d"), "buy": round(float(entry), 1), "sell": round(float(px), 1),
            "ret": round((px / entry - 1) * 100, 2), "days": int(x - e), "reason": reason,
            "touched": touched,
        })
        free_from = x + 1
    return trades


def bench_return(bench: pd.DataFrame | None, d1: str, d2: str) -> float | None:
    if bench is None:
        return None
    try:
        a = bench["Open"].asof(pd.Timestamp(d1))
        b = bench["Open"].asof(pd.Timestamp(d2))
        return round((b / a - 1) * 100, 2) if a and b and a == a and b == b else None
    except Exception:
        return None


# ------------------------------------------------------------------ 集計
def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    r = tr["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    ex = tr["excess"].dropna()
    # 連敗（売った日の順）
    streak = mx = 0
    for x in tr.sort_values("exit")["ret"]:
        streak = streak + 1 if x <= 0 else 0
        mx = max(mx, streak)
    by_year = []
    for y, g in tr.groupby(tr["entry"].str[:4]):
        by_year.append({"year": y, "n": len(g), "win": round((g["ret"] > 0).mean() * 100, 1),
                        "avg": round(g["ret"].mean(), 2), "sum": round(g["ret"].sum(), 1),
                        "bench": round(g["bench"].mean(), 2) if g["bench"].notna().any() else None})
    by_reason = []
    for rs, g in tr.groupby("reason"):
        by_reason.append({"reason": REASON[rs], "n": len(g), "win": round((g["ret"] > 0).mean() * 100, 1),
                          "avg": round(g["ret"].mean(), 2), "days": round(g["days"].mean(), 1)})
    bins = list(np.arange(-30, 32.5, 2.5))
    clipped = r.clip(-30, 29.99)
    hist = np.histogram(clipped, bins=bins)[0].tolist()
    cum = tr.sort_values("exit").assign(cs=lambda d: d["ret"].cumsum())
    cum_pts = [[d, round(v, 1)] for d, v in zip(cum["exit"], cum["cs"])]
    if len(cum_pts) > 400:
        idx = np.linspace(0, len(cum_pts) - 1, 400).astype(int)
        cum_pts = [cum_pts[i] for i in idx]
    return {
        "n": int(len(tr)), "win": round((r > 0).mean() * 100, 1), "avg": round(r.mean(), 2),
        "median": round(r.median(), 2), "avg_win": round(wins.mean(), 2) if len(wins) else None,
        "avg_loss": round(losses.mean(), 2) if len(losses) else None,
        "payoff": round(wins.mean() / -losses.mean(), 2) if len(wins) and len(losses) and losses.mean() < 0 else None,
        "pf": round(wins.sum() / -losses.sum(), 2) if len(losses) and losses.sum() < 0 else None,
        "days": round(tr["days"].mean(), 1), "max_streak": int(mx),
        "best": round(r.max(), 2), "worst": round(r.min(), 2),
        "bench_avg": round(tr["bench"].mean(), 2) if tr["bench"].notna().any() else None,
        "excess_avg": round(ex.mean(), 2) if len(ex) else None,
        "beat": round((ex > 0).mean() * 100, 1) if len(ex) else None,
        "touched": round(tr["touched"].mean() * 100, 1),
        "by_year": by_year, "by_reason": by_reason, "hist": hist, "bins": bins, "cum": cum_pts,
    }


# ------------------------------------------------------------------ 実行
def run(master: pd.DataFrame, prices: dict, bench: pd.DataFrame | None) -> dict:
    all_tr = {v: [] for v in VARIANTS}
    first, last = None, None
    for _, m in master.iterrows():
        df = prices.get(m["コード"])
        if df is None or len(df) < 200:
            continue
        df = df.dropna(subset=["Open", "Close"])
        first = df.index[0] if first is None else min(first, df.index[0])
        last = df.index[-1] if last is None else max(last, df.index[-1])
        sigs = signal_days(df)
        for v in VARIANTS:
            for t in simulate(df, sigs, v):
                t.update({"code": m["コード"], "name": m["銘柄名"], "sector": m["業種"]})
                t["bench"] = bench_return(bench, t["entry"], t["exit"])
                t["excess"] = None if t["bench"] is None else round(t["ret"] - t["bench"], 2)
                all_tr[v].append(t)
    os.makedirs("backtest", exist_ok=True)
    result = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(prices),
              "data_from": first.strftime("%Y-%m-%d") if first is not None else None,
              "data_to": last.strftime("%Y-%m-%d") if last is not None else None,
              "min_volume": MIN_VOLUME, "max_hold": MAX_HOLD, "variants": {}}
    for v, name in VARIANTS.items():
        tr = pd.DataFrame(all_tr[v])
        if not tr.empty:
            tr = tr.sort_values("entry").reset_index(drop=True)
            tr.to_csv(f"backtest/trades_{v}.csv", index=False, encoding="utf-8-sig")
        s = summarize(tr)
        s["name"] = name
        s["first_entry"] = tr["entry"].min() if not tr.empty else None
        s["recent"] = tr.sort_values("entry", ascending=False).head(300).to_dict("records") if not tr.empty else []
        result["variants"][v] = s
    with open("backtest/summary.json", "w", encoding="utf-8") as f:
        json.dump({**result, "variants": {v: {kk: vv for kk, vv in s.items() if kk != "recent"}
                                          for v, s in result["variants"].items()}}, f, ensure_ascii=False, indent=1)
    return result


def main() -> None:
    master = k.load_prime_list()
    print(f"過去の株価を取得します（期間 {PERIOD}）…")
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    bench = None
    try:
        import yfinance as yf
        b = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=False, progress=False)
        if isinstance(b.columns, pd.MultiIndex):
            b.columns = b.columns.get_level_values(0)
        bench = b[["Open", "Close"]].dropna()
    except Exception as e:
        print(f"TOPIX連動ETFの取得に失敗しました（{e}）。比較なしで続けます。")
    result = run(master, prices, bench)
    for v, s in result["variants"].items():
        if s.get("n"):
            print(f"{v} {s['name']}: {s['n']}回 勝率{s['win']}% 平均{s['avg']}% "
                  f"平均利益{s['avg_win']}% 平均損失{s['avg_loss']}% TOPIX比{s['excess_avg']}%")
    from backtest_page import build
    build(result, "web/backtest.html")
    print("結果ページ: web/backtest.html")


if __name__ == "__main__":
    main()
