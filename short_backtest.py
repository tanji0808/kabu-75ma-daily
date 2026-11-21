#!/usr/bin/env python3
"""
空売り版 75日線割れ手法のバックテスト（過去約5年・東証プライム）

売り（空売り）の合図 : スクリーナーの「★ 75日線割れ（本の手法の逆）」と同じ考え方
    - 75日線が60営業日前より2%以上高い（上向き）
    - 直近60営業日の75%以上の日で終値が75日線の上
    - 過去120営業日の安値から15%以上上昇している
    - その日に終値が75日線を上から下に割った（割れ幅3%以内）
    - 出来高50万株以上
  → 翌営業日の始値で空売りする。

買い戻し（パターンごと）
    A : 損切り = 合図の日の75日線 +1%（高値が届いたらその値で）／利確 = 損切り幅の2倍下（安値が届いたらその値で）
    B : 損切りはAと同じ／利確の指値なし。終値が25日線を上回ったら翌日の始値で買い戻す
    C : 損切り = 合図の日までの20日高値 +1%／利確 = 損切り幅の2倍下
  どれも最長60営業日で翌日の始値で買い戻す。同じ日に損切りと利確の両方に届いたら損切りとみなす（厳しめ）。
  窓を開けて損切り・利確の値を飛び越えたときは、その日の始値で約定したとみなす。

資金管理（空売りプランと同じ）
    1回の許容損失 3万円 → 株数 = 3万円 ÷ 1株あたりの損切り幅（100株単位）
    同時に持つのは最大10銘柄。保証金は建玉の30%として、建玉の合計は1,000万円まで（保証金300万円）。
    貸株料 年1.1%（日割り）を差し引く。売買手数料・逆日歩・配当相当額・税金は含めない。

注意 : 今のプライム銘柄だけで検証している（途中で上場廃止になった銘柄は入っていない）。
出力 : backtest/short_summary.json, backtest/short_trades_<A|B|C>.csv
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

PERIOD = os.environ.get("BT_PERIOD", "6y")
MIN_VOLUME = 500_000
RISK_YEN = 30_000
MAX_POS = 10
MAX_NOTIONAL = 10_000_000
LEND_FEE = 0.011
MAX_HOLD = 60
BUF = 0.01
R_MULT = 2.0
WARMUP = 140          # 指標の計算に使う日数。これより前は合図を出さない
OUT = "backtest"

PATTERNS = {
    "A": "75日線+1%で損切り／損切り幅の2倍で利確",
    "B": "75日線+1%で損切り／終値が25日線を上回ったら買い戻し",
    "C": "20日高値+1%で損切り／損切り幅の2倍で利確",
}


def signals(df: pd.DataFrame) -> list[dict]:
    o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    ma75 = pd.Series(c).rolling(75).mean().to_numpy()
    out = []
    for i in range(WARMUP, n - 1):
        m, mp = ma75[i], ma75[i - 1]
        if np.isnan(m) or np.isnan(mp) or np.isnan(ma75[i - 60]):
            continue
        if not (c[i - 1] >= mp and c[i] < m and c[i] / m - 1 >= -0.03):
            continue
        if v[i] < MIN_VOLUME:
            continue
        if m / ma75[i - 60] - 1 < 0.02:
            continue
        win = slice(i - 60, i)
        if np.mean(c[win] > ma75[win]) < 0.75:
            continue
        if h[i - 120:i + 1].max() / l[i - 120:i + 1].min() - 1 < 0.15:
            continue
        out.append({"i": i, "ma75": m, "hi20": h[i - 19:i + 1].max()})
    return out


def simulate(code: str, df: pd.DataFrame, sig: dict, pat: str) -> dict | None:
    o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    ma25 = pd.Series(c).rolling(25).mean().to_numpy()
    e = sig["i"] + 1
    entry = o[e]
    base = sig["hi20"] if pat == "C" else sig["ma75"]
    stop = base * (1 + BUF)
    per = stop - entry
    if per <= 0:          # 寄り付きで損切りラインより上に窓を開けた → 見送り
        return None
    target = entry - per * R_MULT if pat in ("A", "C") else None
    shares = int(RISK_YEN // per // 100 * 100)
    if shares <= 0:
        return None
    n = len(c)
    exit_i, exit_px, why = None, None, None
    for j in range(e, min(n, e + MAX_HOLD)):
        if j > e:  # 前日の終値で決まった買い戻し
            if pat == "B" and not np.isnan(ma25[j - 1]) and c[j - 1] > ma25[j - 1]:
                exit_i, exit_px, why = j, o[j], "25日線超え"
                break
        if h[j] >= stop:
            exit_i, exit_px, why = j, max(o[j], stop) if j > e else max(entry, stop), "損切り"
            break
        if target is not None and l[j] <= target:
            exit_i, exit_px, why = j, min(o[j], target) if j > e else target, "利確"
            break
    if exit_i is None:
        j = min(n - 1, e + MAX_HOLD)
        if j >= n - 1 and e + MAX_HOLD > n - 1:
            exit_i, exit_px, why = n - 1, c[n - 1], "保有中（最終日の終値で評価）"
        else:
            exit_i, exit_px, why = j, o[j], "期限"
    days = (df.index[exit_i] - df.index[e]).days
    gross = (entry - exit_px) * shares
    fee = entry * shares * LEND_FEE * max(days, 1) / 365
    return {"code": code, "signal": df.index[sig["i"]].strftime("%Y-%m-%d"),
            "entry_date": df.index[e].strftime("%Y-%m-%d"), "exit_date": df.index[exit_i].strftime("%Y-%m-%d"),
            "entry": round(entry, 1), "stop": round(stop, 1), "target": round(target, 1) if target else None,
            "exit": round(exit_px, 1), "shares": shares, "notional": round(entry * shares),
            "days": days, "why": why, "pnl": round(gross - fee), "fee": round(fee),
            "ret_pct": round((entry - exit_px) / entry * 100, 2)}


def portfolio(trades: list[dict]) -> list[dict]:
    """同時10銘柄・建玉1,000万円までの制約で、合図の早い順に受ける。"""
    taken, open_ = [], []
    for t in sorted(trades, key=lambda x: (x["entry_date"], x["code"])):
        open_ = [x for x in open_ if x["exit_date"] > t["entry_date"]]
        if len(open_) >= MAX_POS or sum(x["notional"] for x in open_) + t["notional"] > MAX_NOTIONAL:
            continue
        if any(x["code"] == t["code"] for x in open_):
            continue
        open_.append(t)
        taken.append(t)
    return taken


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"trades": 0}
    df = pd.DataFrame(tr).sort_values("exit_date")
    eq = df["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    df["year"] = df["exit_date"].str[:4]
    wins = df[df["pnl"] > 0]
    loss = df[df["pnl"] <= 0]
    return {
        "trades": int(len(df)), "total_pnl": int(df["pnl"].sum()), "win_rate": round(len(wins) / len(df) * 100, 1),
        "avg_win": int(wins["pnl"].mean()) if len(wins) else 0, "avg_loss": int(loss["pnl"].mean()) if len(loss) else 0,
        "profit_factor": round(wins["pnl"].sum() / -loss["pnl"].sum(), 2) if len(loss) and loss["pnl"].sum() < 0 else None,
        "max_drawdown": int(dd), "lend_fee_total": int(df["fee"].sum()),
        "avg_days": round(df["days"].mean(), 1),
        "by_year": {y: {"pnl": int(g["pnl"].sum()), "trades": int(len(g))} for y, g in df.groupby("year")},
        "by_reason": {r: int(len(g)) for r, g in df.groupby("why")},
        "first": df["entry_date"].min(), "last": df["exit_date"].max(),
    }


def main() -> None:
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=PERIOD)
    os.makedirs(OUT, exist_ok=True)
    raw = {p: [] for p in PATTERNS}
    nsig = 0
    for code, df in prices.items():
        df = df[(df["Open"] > 0) & (df["Low"] > 0)]
        # 明らかに壊れた値（前日終値の1.6倍超・0.4倍未満の始値・高値・安値）がある日を除く
        pc = df["Close"].shift(1)
        bad = (df[["Open", "High", "Close"]].max(axis=1) > pc * 1.6) | (df[["Open", "Low", "Close"]].min(axis=1) < pc * 0.4)
        df = df[~bad]
        # 上場廃止・再上場などで30日以上データが途切れている銘柄は、途切れた後だけを使う
        gaps = np.where(np.diff(df.index.values).astype("timedelta64[D]").astype(int) > 30)[0]
        if len(gaps):
            df = df.iloc[gaps[-1] + 1:]
        # 20日の中央値から3倍以上・3分の1以下に離れた値がある日も除く
        med = df["Close"].rolling(20, min_periods=5, center=True).median()
        df = df[(df[["Open", "High", "Close"]].max(axis=1) < med * 3) & (df[["Open", "Low", "Close"]].min(axis=1) > med / 3)]
        if len(df) < WARMUP + 20:
            continue
        sigs = signals(df)
        nsig += len(sigs)
        for p in PATTERNS:
            busy_until = ""
            for s in sigs:      # 同じ銘柄を持っている間の新しい合図は無視
                if df.index[s["i"]].strftime("%Y-%m-%d") <= busy_until:
                    continue
                t = simulate(code, df, s, p)
                if t:
                    raw[p].append(t)
                    busy_until = t["exit_date"]
    summary = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": len(prices), "signals": nsig,
               "rules": {"risk_yen": RISK_YEN, "max_pos": MAX_POS, "max_notional": MAX_NOTIONAL,
                         "lend_fee": LEND_FEE, "max_hold": MAX_HOLD}, "patterns": {}}
    try:
        import yfinance as yf
        b = yf.download("1306.T", period=PERIOD, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
        b = b[b.index >= b.index[0] + pd.Timedelta(days=200)]
        summary["topix_etf_return_pct"] = round(float(b.iloc[-1] / b.iloc[0] - 1) * 100, 1)
        summary["topix_period"] = [b.index[0].strftime("%Y-%m-%d"), b.index[-1].strftime("%Y-%m-%d")]
    except Exception as e:
        print("TOPIX ETF を取得できませんでした", e)
    for p, name in PATTERNS.items():
        taken = portfolio(raw[p])
        pd.DataFrame(taken).to_csv(os.path.join(OUT, f"short_trades_{p}.csv"), index=False, encoding="utf-8-sig")
        summary["patterns"][p] = {"name": name, "all_signals": stats(raw[p]), "portfolio": stats(taken)}
        st = summary["patterns"][p]["portfolio"]
        print(f"{p} {name}: {st.get('trades')}回 損益 {st.get('total_pnl', 0):,}円 勝率 {st.get('win_rate')}%")
    with open(os.path.join(OUT, "short_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
