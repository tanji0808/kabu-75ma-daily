#!/usr/bin/env python3
"""空売りの候補（backtest/short_candidates_11y.csv.gz）に、合図の日までの「上昇の長さ」の材料を付ける。
  rise{N} : N営業日前から今日までの75日線の上昇率（％）
  above{N}: N営業日前〜6営業日前に、終値が75日線より上だった日の割合
出力 : backtest/short_rise_11y.csv.gz"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
os.environ.setdefault("BT_PERIOD", "11y")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402

LB = (20, 40, 60, 90, 120, 180, 250)


def main():
    cand = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period=os.environ["BT_PERIOD"])
    out = []
    for code, g in cand.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        c = df["Close"].to_numpy(float)
        ma = pd.Series(c).rolling(75).mean().to_numpy()
        above = c > ma
        idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)}
        for sd in g.signal:
            t = idx.get(sd)
            if t is None:
                continue
            r = {"code": code, "signal": sd}
            for n in LB:
                if t - n >= 74:
                    r[f"rise{n}"] = (ma[t] / ma[t - n] - 1) * 100
                    r[f"above{n}"] = float(above[t - n:t - 5].mean())
            out.append(r)
    pd.DataFrame(out).to_csv("backtest/short_rise_11y.csv.gz", index=False, compression="gzip")
    print(f"保存しました: {len(out)}件")


if __name__ == "__main__":
    main()
