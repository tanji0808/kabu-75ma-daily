"""
資金管理の改善案を比べるシミュレーション（毎日の資産を追いながら売買を判断する）

土台：採用ルール（急騰除外・ランク6以上・引き上げ式ロスカット）の取引記録 backtest/trades_trail_calm5_rsi60.csv
自己資金 1,000万円、建玉は最大 2,000万円まで。

比べる工夫（数字は事前に決めた値。①の損の上限だけ、2021〜2024年のデータで平均の金額が100万円前後になるよう決める）
  ① 値動きで金額を変える：1回の損（買値→最初のロスカット）が資金の一定割合になるように買う
  ② 同じ業種は同時に3銘柄まで
  ③ TOPIX（1306）が200日線より下の日は、新しく買う金額を半分に
  ④ 資産がピークから15%以上減ったら、新しく買う金額を半分に（10%以内に戻ったら元に戻す）
  ⑤ 使っていない資金（自己資金−建玉）を TOPIX ETF（1306）で持つ
  ⑥ 現物中心：金利は自己資金を超えた分（借りた分）にだけかかる（基準は建玉全体に年3%）
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402

EQUITY0, MAX_EXPO, BASE_SIZE, BASE_CAP, RATE = 10_000_000, 20_000_000, 1_000_000, 20, 0.03
SPLIT = "2025-01-01"

t = pd.read_csv("backtest/trades_trail_calm5_rsi60.csv", dtype={"code": str})
t = t[t["frank"] >= 6].copy()
t["entry"], t["exit"], t["signal"] = (pd.to_datetime(t[c]) for c in ("entry", "exit", "signal"))
codes = sorted(t["code"].unique())
prices = k.download_prices(codes + ["1306"], period="6y")
def norm(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df
prices = {c: norm(df) for c, df in prices.items()}

# 最初のロスカットの幅（買値 − 合図の日の −1σ）
def stop_pct(r):
    df = prices.get(r["code"])
    if df is None or r["signal"] not in df.index:
        return np.nan
    c = df["Close"]
    d1 = (c.rolling(25).mean() - c.rolling(25).std(ddof=0)).loc[r["signal"]]
    return (r["buy"] - d1) / r["buy"] if d1 < r["buy"] else 0.03
t["stop_pct"] = t.apply(stop_pct, axis=1)
t = t.dropna(subset=["stop_pct"])
t = t.sort_values(["entry", "frank", "stop_pct"], ascending=[True, False, True]).reset_index(drop=True)

days = pd.bdate_range(t["entry"].min(), t["exit"].max())
close = pd.DataFrame({c: prices[c]["Close"] for c in codes if c in prices}).reindex(days).ffill()
etf = etf_ma200 = etf_ret = None
if "1306" in prices:
    raw = prices["1306"]["Close"].dropna()
    r_ = raw.pct_change().fillna(0)
    jumps = r_[r_.abs() > 0.2]
    if len(jumps):
        print("ETF（1306）の株価の飛び（分割などとみなして補正）:", {str(d.date()): round(float(v) * 100, 1) for d, v in jumps.items()})
    r_[r_.abs() > 0.2] = 0.0                         # 株式分割などによる見かけの急変を除く
    idx = (1 + r_).cumprod()                         # 分割の影響を除いた指数
    etf = idx.reindex(days).ffill()
    etf_ma200 = idx.rolling(200).mean().reindex(days).ffill()
    etf_ret = etf.pct_change().fillna(0)

# ①の損の上限：2021〜2024年の取引で、平均の金額が BASE_SIZE になる割合
train = t[t["entry"] < SPLIT]
RISK = float(BASE_SIZE * train["stop_pct"].median() / EQUITY0)
print(f"①の1回の損の上限：資金の {RISK * 100:.2f}%（最初のロスカット幅の中央値 {train['stop_pct'].median() * 100:.1f}%）")


def simulate(vol=False, sector=False, regime=False, ddctl=False, idle_etf=False, cash=False, etf_trend=False):
    by_entry = {d: g for d, g in t.groupby("entry")}
    open_pos = []          # dict(code, sector, shares, cost, exit)
    realized = etf_pnl = interest = 0.0
    peak, half_dd = EQUITY0, False
    eq_series = []
    taken = 0
    prev_idle = 0.0
    for d in days:
        # 1) 売り（今日の始値）
        still = []
        for p in open_pos:
            if p["exit"] == d:
                realized += p["shares"] * p["sell"] - p["cost"]
            else:
                still.append(p)
        open_pos = still
        # 2) ⑤ 前日の遊休資金に今日のETFの値動き
        if idle_etf and etf_ret is not None:
            etf_pnl += prev_idle * float(etf_ret.loc[d])
        # 3) 買い（今日の始値）
        equity_now = eq_series[-1][1] if eq_series else EQUITY0
        for _, r in by_entry.get(d, pd.DataFrame()).iterrows():
            expo = sum(p["cost"] for p in open_pos)
            if not vol and len(open_pos) >= BASE_CAP:
                break
            if sector and sum(p["sector"] == r["sector"] for p in open_pos) >= 3:
                continue
            size = (RISK * equity_now / r["stop_pct"]) if vol else BASE_SIZE
            size = min(size, 3 * BASE_SIZE)
            factor = 1.0
            if regime and etf is not None and etf.loc[:d].iloc[-1] < etf_ma200.loc[:d].iloc[-1]:
                factor *= 0.5
            if ddctl and half_dd:
                factor *= 0.5
            size *= factor
            if expo + size > MAX_EXPO:
                size = MAX_EXPO - expo
            if size < 200_000:
                continue
            open_pos.append({"code": r["code"], "sector": r["sector"], "shares": size / r["buy"], "cost": size,
                             "sell": r["sell"], "exit": r["exit"]})
            taken += 1
        # 4) 今日の終値で評価・金利
        expo = sum(p["cost"] for p in open_pos)
        unreal = sum(p["shares"] * close.at[d, p["code"]] - p["cost"] for p in open_pos if p["code"] in close)
        equity = EQUITY0 + realized + unreal + etf_pnl - interest
        borrowed = max(0.0, expo - equity) if cash else expo
        interest += borrowed * RATE / 245
        equity -= borrowed * RATE / 245
        prev_idle = max(0.0, equity - expo) if idle_etf else 0.0
        if etf_trend and etf is not None and etf.loc[d] < etf_ma200.loc[d]:
            prev_idle = 0.0                          # TOPIXが200日線の下にある間はETFを持たない
        peak = max(peak, equity)
        if ddctl:
            if equity < peak * 0.85:
                half_dd = True
            elif equity >= peak * 0.90:
                half_dd = False
        eq_series.append((d, equity, expo))
    s = pd.DataFrame(eq_series, columns=["d", "eq", "expo"]).set_index("d")
    return s, taken, interest


def metrics(s, taken, interest, lo=None, hi=None):
    x = s.loc[lo:hi] if lo or hi else s
    eq = x["eq"]
    start = eq.iloc[0] if lo else EQUITY0
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    dd = eq.cummax() - eq
    return {"pnl": round(float(eq.iloc[-1] - start)), "annual": round(float((eq.iloc[-1] - start) / yrs)),
            "max_dd": round(float(dd.max())), "max_dd_pct": round(float((dd / eq.cummax()).max() * 100), 1),
            "avg_expo": round(float(x["expo"].mean())), "max_expo": round(float(x["expo"].max()))}


CONFIGS = [
    ("基準（100万円×20銘柄・建玉全体に金利）", {}),
    ("⑥ 現物中心（借りた分だけ金利）", {"cash": True}),
    ("⑥＋① 値動きで金額を変える", {"cash": True, "vol": True}),
    ("⑥＋② 同じ業種は3銘柄まで", {"cash": True, "sector": True}),
    ("⑥＋③ TOPIXが200日線の下なら半分", {"cash": True, "regime": True}),
    ("⑥＋④ 資産が−15%なら半分", {"cash": True, "ddctl": True}),
    ("⑥＋⑤ 遊休資金をTOPIX ETFに", {"cash": True, "idle_etf": True}),
    ("⑥＋③＋⑤（効いた3つ）", {"cash": True, "regime": True, "idle_etf": True}),
    ("⑥＋③＋⑤（ETFも200日線の下では持たない）", {"cash": True, "regime": True, "idle_etf": True, "etf_trend": True}),
    ("⑥＋①②③④（リスク対策すべて）", {"cash": True, "vol": True, "sector": True, "regime": True, "ddctl": True}),
    ("⑥＋①②③④＋⑤（全部）", {"cash": True, "vol": True, "sector": True, "regime": True, "ddctl": True, "idle_etf": True}),
]
out = {"risk_pct": round(RISK * 100, 3), "results": []}
for name, cfg in CONFIGS:
    s, taken, interest = simulate(**cfg)
    yearly = s["eq"].groupby(s.index.year).last()
    by_year = (yearly - yearly.shift(1).fillna(EQUITY0)).round().astype(int).to_dict()
    res = {"name": name, "trades": taken, "interest": round(interest), "all": metrics(s, taken, interest),
           "train": metrics(s, taken, interest, hi=pd.Timestamp(SPLIT) - pd.Timedelta(days=1)),
           "test": metrics(s, taken, interest, lo=pd.Timestamp(SPLIT)),
           "by_year": {str(k_): v for k_, v in by_year.items()}}
    out["results"].append(res)
    a = res["all"]
    print(f"{name}: 年{a['annual'] / 1e4:.0f}万 最大落込{a['max_dd'] / 1e4:.0f}万({a['max_dd_pct']}%) "
          f"前半 年{res['train']['annual'] / 1e4:.0f}万/落込{res['train']['max_dd_pct']}% 後半 年{res['test']['annual'] / 1e4:.0f}万/落込{res['test']['max_dd_pct']}% "
          f"取引{taken} 金利{interest / 1e4:.0f}万 平均建玉{a['avg_expo'] / 1e4:.0f}万")
os.makedirs("backtest", exist_ok=True)
with open("backtest/portfolio2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
