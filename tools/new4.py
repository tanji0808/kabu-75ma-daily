"""新しい手法の材料（3つ分）。出力 backtest/new_pullback.csv.gz / new_kenri.csv.gz / new_squeeze.csv.gz
A 押し目買い：上昇トレンド（終値>75日線、75日線が20日前より上、直近20日以内に52週高値）の銘柄が、25日線まで押した日（安値≤25日線・終値>75日線）。
B 権利取り：3月・9月の権利付き最終日（月末から2営業日前）の N 営業日前の引けで買い、権利付き最終日の引けで売る（配当はもらわない）。
C 踏み上げ：20日高値を終値で抜け、出来高が20日平均の2倍以上の日（信用の売り残はあとで手元のデータで絞る）。
A・C は翌日の寄りで買い、いくつかの売り方の損益（％）を出す。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import newhigh_candidates as nh  # noqa: E402
EX = {"h5": dict(stop=0.99, hold=5), "h10": dict(stop=0.99, hold=10), "h20": dict(stop=0.99, hold=20),
      "s5h10": dict(stop=0.05, hold=10), "s6h20": dict(stop=0.06, hold=20), "s8h20": dict(stop=0.08, hold=20),
      "s6tp10": dict(stop=0.06, tp=0.10, hold=20), "s6tr": dict(stop=0.06, arm=0.15, armlock=0.0, armtrail=0.10, hold=60),
      "s8tr": dict(stop=0.08, arm=0.25, armlock=0.0, armtrail=0.15, hold=120)}


def main():
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    A, B, Cq = [], [], []
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) < 300:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        s = pd.Series(c); m25 = s.rolling(25).mean().to_numpy(); m75 = s.rolling(75).mean().to_numpy()
        hs = pd.Series(h); hi250 = hs.rolling(250).max().to_numpy(); hi20p = pd.Series(c).rolling(20).max().shift(1).to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().to_numpy(); va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        dates = df.index.strftime("%Y-%m-%d").tolist(); n = len(c)
        a = {"o": o, "h": h, "l": l, "c": c, "ma25": m25, "ma50": s.rolling(50).mean().to_numpy(), "pivot": np.full(n, np.nan), "dates": dates}
        last_hi = -999
        for t in range(260, n - 2):
            if h[t] >= hi250[t]:
                last_hi = t
            if dates[t] < "2016-12-01" or not (tv20[t] >= 1e8):
                continue
            row = None
            # A 押し目
            if c[t] > m75[t] and m75[t] > m75[t - 20] and t - last_hi <= 20 and l[t] <= m25[t] and c[t - 1] > m25[t - 1]:
                row = {"code": code, "date": dates[t], "dev75": c[t] / m75[t] - 1, "dd_hi": c[t] / hi250[t] - 1, "clv": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
                       "vr": v[t] / va[t] if va[t] else np.nan, "tv20": tv20[t], "close": c[t], "ret5": c[t] / c[t - 5] - 1}
                for nm, prm in EX.items():
                    x = nh.sim(a, t, **prm)
                    if x:
                        row[nm], row[nm + "x"] = x["ret"] / 100, x["exit"]
                A.append(row)
            # C 踏み上げ候補
            if c[t] > hi20p[t] and va[t] and v[t] >= 2 * va[t]:
                r2 = {"code": code, "date": dates[t], "vr": v[t] / va[t], "day": c[t] / c[t - 1] - 1, "clv": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
                      "dev75": c[t] / m75[t] - 1, "near250": c[t] / hi250[t] - 1, "tv20": tv20[t], "close": c[t], "ret20": c[t] / c[t - 20] - 1}
                for nm, prm in EX.items():
                    x = nh.sim(a, t, **prm)
                    if x:
                        r2[nm], r2[nm + "x"] = x["ret"] / 100, x["exit"]
                Cq.append(r2)
        # B 権利取り
        idx = pd.DatetimeIndex(df.index)
        for y in range(2017, 2027):
            for mth in (3, 9):
                mdays = [i for i, d in enumerate(idx) if d.year == y and d.month == mth]
                if len(mdays) < 5:
                    continue
                lt = mdays[-1] - 2          # 権利付き最終日（月末の2営業日前。2019年7月より前は3営業日前）
                if pd.Timestamp(y, mth, 1) < pd.Timestamp(2019, 7, 1):
                    lt = mdays[-1] - 3
                if lt < 30 or not (tv20[lt] >= 3e7):
                    continue
                r3 = {"code": code, "y": y, "m": mth, "last": dates[lt], "close": c[lt], "tv20": tv20[lt],
                      "drop_ex": o[lt + 1] / c[lt] - 1 if lt + 1 < n else np.nan}
                for N in (3, 5, 10, 15, 20, 30):
                    r3[f"b{N}"] = c[lt] / c[lt - N] - 1
                    r3[f"b{N}d"] = dates[lt - N]
                B.append(r3)
    pd.DataFrame(A).to_csv("backtest/new_pullback.csv.gz", index=False, compression="gzip")
    pd.DataFrame(B).to_csv("backtest/new_kenri.csv.gz", index=False, compression="gzip")
    pd.DataFrame(Cq).to_csv("backtest/new_squeeze.csv.gz", index=False, compression="gzip")
    print("保存しました", len(A), len(B), len(Cq))


if __name__ == "__main__":
    main()
