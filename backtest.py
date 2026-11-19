#!/usr/bin/env python3
"""
75日線ブレイク手法のバックテスト（過去約5年・東証プライム）

買い : 3ヶ月以上の下落のあと、終値が75日線を下から上に抜けた日（スクリーナーの「本日突破」と同じ条件、
       出来高50万株以上）の「翌営業日の始値」で買う。
利確 : 買ったあと、終値がボリンジャーバンド(25日)の +1σ より上に乗り、そのあと終値が +1σ を下回ったら、
       翌営業日の始値で売る。
売り方（4通り）と、買いの絞り込みの有無（2通り）の組み合わせ8パターンを比較する。
   s1  : +1σに乗った後 +1σ割れで利確／終値が75日線を下回ったら損切り（もとのルール）
   ma25: 終値が25日線を下回ったら売る
   m1s : 終値が −1σ を下回ったら売る
   s2  : +2σに届いた後 +1σ割れで利確／終値が75日線を下回ったら損切り
   絞り込み f : 合図の日に 25日線が5日前より上 かつ 出来高が過去20日平均の1.5倍以上
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
# 売り方 × 買いの絞り込み の組み合わせ
EXITS = {
    "s1": "+1σに乗った後+1σ割れで利確／75日線割れで損切り（今のルール）",
    "ma25": "25日線割れで売る",
    "m1s": "−1σ割れで売る",
    "s2": "+2σに届いた後+1σ割れで利確／75日線割れで損切り",
}
FILTERS = {"": "絞り込みなし", "f": "25日線上向き＋出来高1.5倍"}
SHORT = {"s1": "今のルール", "ma25": "25日線割れで売る", "m1s": "−1σ割れで売る", "s2": "+2σ後の+1σ割れ"}
VARIANTS = {}
for fk, fname in FILTERS.items():
    for ek, ename in EXITS.items():
        VARIANTS[f"{ek}{('_' + fk) if fk else ''}"] = {"exit": ek, "filter": fk, "name": ename, "fname": fname,
                                                       "short": SHORT[ek], "tp": None}
# +10%で利益確定（指値）を足したパターン
TP = 0.10
for base in ("ma25", "ma25_f", "m1s"):
    cfg = dict(VARIANTS[base])
    cfg.update(tp=TP, name=cfg["name"] + f"＋{int(TP * 100)}%で利確", short=cfg["short"] + f"＋{int(TP * 100)}%利確")
    VARIANTS[base + "_tp"] = cfg
REASON = {"tp": "+10%で利確", "take": "利確（+1σ割れ）", "stop": "損切り（75日線割れ）", "time": "期間満了",
          "ma25": "25日線割れ", "m1s": "−1σ割れ"}


# ------------------------------------------------------------------ 合図
def signal_days(df: pd.DataFrame, filt: str = "") -> np.ndarray:
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
    if filt == "f":   # 案3：25日線が上向き（5日前より上）かつ 出来高が過去20日平均の1.5倍以上
        ma25 = c.rolling(25).mean()
        sig = sig & (ma25 > ma25.shift(5)) & (v >= 1.5 * v.shift(1).rolling(20).mean())
    need = max(k.MA + dd, k.DROP_LOOKBACK) + 5          # スクリーナーと同じく、十分なデータがある日だけ
    idx = np.flatnonzero(sig.fillna(False).to_numpy())
    return idx[idx >= need - 1]


# ------------------------------------------------------------------ 売買
def simulate(df: pd.DataFrame, sigs: np.ndarray, exit_rule: str, tp: float | None = None) -> list[dict]:
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    hi = df["High"].to_numpy(float)
    ma75 = df["Close"].rolling(k.MA).mean().to_numpy()
    m25s = df["Close"].rolling(25).mean()
    sd = df["Close"].rolling(25).std(ddof=0)
    ma25, u1, u2, d1 = m25s.to_numpy(), (m25s + sd).to_numpy(), (m25s + 2 * sd).to_numpy(), (m25s - sd).to_numpy()
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
        touched1 = touched2 = False
        exit_sig, reason, tp_px = None, None, None
        target = entry * (1 + tp) if tp else None
        for j in range(e, min(n, e + MAX_HOLD)):
            # 利確の指値（場中）：終値で判断する売りより先に約定する
            if target is not None:
                if j > e and o[j] >= target:            # 朝から目標以上で始まった → 始値で売れる
                    exit_sig, reason, tp_px = j, "tp", o[j]; break
                if hi[j] >= target:                      # 場中に目標に届いた → 目標の値段で売れる
                    exit_sig, reason, tp_px = j, "tp", target; break
            if exit_rule == "s1":
                if touched1 and c[j] < u1[j]:
                    exit_sig, reason = j, "take"; break
                if c[j] < ma75[j]:
                    exit_sig, reason = j, "stop"; break
            elif exit_rule == "s2":
                if touched2 and c[j] < u1[j]:
                    exit_sig, reason = j, "take"; break
                if c[j] < ma75[j]:
                    exit_sig, reason = j, "stop"; break
            elif exit_rule == "ma25":
                if c[j] < ma25[j]:
                    exit_sig, reason = j, "ma25"; break
            elif exit_rule == "m1s":
                if c[j] < d1[j]:
                    exit_sig, reason = j, "m1s"; break
            touched1 = touched1 or c[j] > u1[j]
            touched2 = touched2 or c[j] > u2[j]
        if exit_sig is None:
            if e + MAX_HOLD - 1 < n:
                exit_sig, reason = e + MAX_HOLD - 1, "time"
            else:
                free_from = n          # まだ持っている途中（検証期間の終わり）なので集計しない
                continue
        if tp_px is not None:           # 利確の指値はその日のうちに売れている
            x, px = exit_sig, tp_px
        elif exit_sig + 1 >= n:         # 売りの合図が最終日に出た：その日の終値で売ったことにする
            x, px = exit_sig, c[exit_sig]
        else:
            x = exit_sig + 1
            px = o[x]
        trades.append({
            "signal": dates[t].strftime("%Y-%m-%d"), "entry": dates[e].strftime("%Y-%m-%d"),
            "exit": dates[x].strftime("%Y-%m-%d"), "buy": round(float(entry), 1), "sell": round(float(px), 1),
            "ret": round((px / entry - 1) * 100, 2), "days": int(x - e), "reason": reason,
            "touched": bool(touched2 if exit_rule == "s2" else touched1),
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


# ------------------------------------------------------------------ ファンダメンタルランク別（参考）
RANK_GROUPS = [("8以上", 8, 10), ("6以上", 6, 10), ("5以下", 0, 5)]


def attach_ranks(master: pd.DataFrame, prices: dict, all_tr: dict) -> None:
    """合図の日の終値で計算したファンダメンタルランクを各取引に付ける（財務データは今のもの）。"""
    fpath = os.environ.get("FUND_PATH", "data/fundamentals.json")
    if not os.path.exists(fpath):
        return
    with open(fpath, encoding="utf-8") as f:
        fund = json.load(f)
    closes = pd.DataFrame({c: df["Close"] for c, df in prices.items()}).sort_index().ffill()
    dates = sorted({t["signal"] for tr in all_tr.values() for t in tr})
    cache = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts not in closes.index:
            continue
        cache[d] = k.fund_ranks(master, prices, fund, close_map=closes.loc[ts].to_dict())
    for tr in all_tr.values():
        for t in tr:
            t["frank"] = cache.get(t["signal"], {}).get(t["code"])


def rank_stats(tr: pd.DataFrame, last_date: str) -> list[dict]:
    if tr.empty or "frank" not in tr:
        return []
    one_year_ago = (pd.Timestamp(last_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    out = []
    for period, sub in (("全期間", tr), ("直近1年", tr[tr["entry"] >= one_year_ago])):
        rows = [("すべて", sub)] + [(g, sub[sub["frank"].between(lo, hi)]) for g, lo, hi in RANK_GROUPS]
        for g, x in rows:
            if not len(x):
                out.append({"period": period, "group": g, "n": 0})
                continue
            r, ex = x["ret"], x["excess"].dropna()
            out.append({"period": period, "group": g, "n": int(len(x)), "win": round((r > 0).mean() * 100, 1),
                        "avg": round(r.mean(), 2), "median": round(r.median(), 2),
                        "excess": round(ex.mean(), 2) if len(ex) else None,
                        "pf": round(r[r > 0].sum() / -r[r <= 0].sum(), 2) if (r <= 0).any() and r[r <= 0].sum() < 0 else None})
    return out


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
        sig_by_filter = {fk: signal_days(df, fk) for fk in FILTERS}
        for v, cfg in VARIANTS.items():
            for t in simulate(df, sig_by_filter[cfg["filter"]], cfg["exit"], cfg.get("tp")):
                t.update({"code": m["コード"], "name": m["銘柄名"], "sector": m["業種"]})
                t["bench"] = bench_return(bench, t["entry"], t["exit"])
                t["excess"] = None if t["bench"] is None else round(t["ret"] - t["bench"], 2)
                all_tr[v].append(t)
    attach_ranks(master, prices, all_tr)
    os.makedirs("backtest", exist_ok=True)
    result = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(prices),
              "data_from": first.strftime("%Y-%m-%d") if first is not None else None,
              "data_to": last.strftime("%Y-%m-%d") if last is not None else None,
              "min_volume": MIN_VOLUME, "max_hold": MAX_HOLD, "variants": {}}
    for v, cfg in VARIANTS.items():
        tr = pd.DataFrame(all_tr[v])
        if not tr.empty:
            tr = tr.sort_values("entry").reset_index(drop=True)
            tr.to_csv(f"backtest/trades_{v}.csv", index=False, encoding="utf-8-sig")
        s = summarize(tr)
        s["by_rank"] = rank_stats(tr, result["data_to"]) if result["data_to"] else []
        s.update({"name": cfg["name"], "fname": cfg["fname"], "short": cfg["short"], "exit_rule": cfg["exit"], "filter": cfg["filter"]})
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
            print(f"{v} {s['fname']}／{s['name']}: {s['n']}回 勝率{s['win']}% 平均{s['avg']}% "
                  f"平均利益{s['avg_win']}% 平均損失{s['avg_loss']}% TOPIX比{s['excess_avg']}%")
    from backtest_page import build
    build(result, "web/backtest.html")
    print("結果ページ: web/backtest.html")


if __name__ == "__main__":
    main()
