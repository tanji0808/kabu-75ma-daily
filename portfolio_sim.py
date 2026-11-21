"""
資金配分のシミュレーション（毎日の含み損益まで計算）
バックテストの取引記録（③ −1σ割れで売る・ランク6以上）を使い、
1銘柄 SIZE 円・同時に最大 CAP 銘柄（合図が重なった日はランクの高い順）で売買した場合の
毎日の資産、最大の落ち込み、信用の維持率を計算する。金利は年 RATE（信用の買い建て）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402

VARIANT = os.environ.get("PF_VARIANT", "m1s")
SIZE = float(os.environ.get("PF_SIZE", 1_000_000))
CAP = int(os.environ.get("PF_CAP", 20))
EQUITY = float(os.environ.get("PF_EQUITY", 10_000_000))
RATE = float(os.environ.get("PF_RATE", 0.03))
MIN_RANK = int(os.environ.get("PF_MIN_RANK", 6))

t = pd.read_csv(f"backtest/trades_{VARIANT}.csv", dtype={"code": str})
t = t[t["frank"] >= MIN_RANK].copy()
t["entry"], t["exit"] = pd.to_datetime(t["entry"]), pd.to_datetime(t["exit"])
t = t.sort_values(["entry", "frank"], ascending=[True, False]).reset_index(drop=True)

# 同時保有の上限で、実際に買える取引を選ぶ
taken, openp = [], []
for i, r in t.iterrows():
    openp = [x for x in openp if x > r["entry"]]
    if len(openp) < CAP:
        openp.append(r["exit"])
        taken.append(i)
tt = t.loc[taken].reset_index(drop=True)
print(f"合図 {len(t)}件のうち、買えたのは {len(tt)}件（上限 {CAP}銘柄）")

prices = k.download_prices(sorted(tt["code"].unique()), period="6y")
days = pd.bdate_range(tt["entry"].min(), tt["exit"].max())
raw = pd.DataFrame({c: df["Close"] for c, df in prices.items()})
raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
close = raw.reindex(days).ffill()
print(f"株価を取得できた銘柄 {raw.shape[1]} / {tt['code'].nunique()}、株価の日付 {raw.index.min().date()}〜{raw.index.max().date()}")

realized = pd.Series(0.0, index=days)
unreal = pd.Series(0.0, index=days)
expo = pd.Series(0.0, index=days)
interest = pd.Series(0.0, index=days)
for _, r in tt.iterrows():
    shares = SIZE / r["buy"]
    hold = (days >= r["entry"]) & (days < r["exit"])
    if r["code"] in close:
        unreal[hold] += shares * close.loc[hold, r["code"]].to_numpy() - SIZE
    expo[hold] += SIZE
    interest[hold] += SIZE * RATE / 245
    realized[days >= r["exit"]] += SIZE * r["ret"] / 100
print(f"含み損益が計算できなかった日: {int(unreal.isna().sum())}日 / {len(days)}日")
unreal = unreal.fillna(0)
pnl = realized + unreal - interest.cumsum()
equity = EQUITY + pnl
dd = (equity.cummax() - equity)
ratio = (equity / expo).where(expo > 0)          # 委託保証金維持率（概算）= 資産 ÷ 建玉
yrs = (days[-1] - days[0]).days / 365.25
by_year = pnl.groupby(pnl.index.year).last().diff().fillna(pnl.groupby(pnl.index.year).last())
out = {
    "variant": VARIANT, "size": SIZE, "cap": CAP, "equity": EQUITY, "rate": RATE,
    "trades": int(len(tt)), "signals": int(len(t)), "years": round(yrs, 2),
    "final_pnl": round(float(pnl.iloc[-1])), "annual_pnl": round(float(pnl.iloc[-1] / yrs)),
    "max_dd": round(float(dd.max())), "max_dd_date": str(dd.idxmax().date()),
    "max_dd_pct": round(float((dd / equity.cummax()).max() * 100), 1),
    "min_ratio": round(float(ratio.min() * 100), 1), "min_ratio_date": str(ratio.idxmin().date()),
    "days_ratio_below": {str(x): int((ratio < x / 100).sum()) for x in (20, 25, 30)},
    "max_expo": round(float(expo.max())), "avg_expo": round(float(expo[expo > 0].mean())),
    "max_positions": int((expo / SIZE).max()), "avg_positions": round(float((expo / SIZE).mean()), 1),
    "interest_total": round(float(interest.sum())),
    "by_year": {str(y): round(float(v)) for y, v in by_year.items()},
    "worst_unreal": round(float(unreal.min())), "worst_unreal_date": str(unreal.idxmin().date()),
}
os.makedirs("backtest", exist_ok=True)
with open(f"backtest/portfolio_{VARIANT}_{CAP}x{int(SIZE / 1e4)}.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
