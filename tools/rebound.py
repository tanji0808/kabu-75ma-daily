"""リバウンド（短期の逆張り）の材料：急に売られすぎた銘柄（終値で判定）を翌日の寄りで買い、短い期間で売る。
合図の条件（どれかを満たした日を全部出す。あとで絞る）：25日線から−12%以上下、RSI(14)25以下、5日で−12%以上、−2.5σ割れ
出力 backtest/rebound.csv.gz
  oc0  : 翌日の寄りで買って、その日の引けで売る（デイトレ）
  tpN  : 翌日の寄りで買って、その日のうちに +N% に届けば指値で売り、届かなければ引け
  tpNsM: 同じ＋その日のうちに −M% に届けば逆指値で売り（同じ日に両方届いたら損切りが先と見なす）
  oo1/oc1/oc2/oc4 : 翌日の寄りで買って、翌々日の寄り／引け、3日目・5日目の引け"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402


def main():
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) < 100:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        s = pd.Series(c); m25 = s.rolling(25).mean().to_numpy(); sd = s.rolling(25).std(ddof=0).to_numpy()
        m75 = s.rolling(75).mean().to_numpy(); m200 = s.rolling(200).mean().to_numpy()
        d_ = s.diff(); up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn)).to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().to_numpy(); va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        dates = df.index.strftime("%Y-%m-%d").tolist(); n = len(c)
        dev = c / m25 - 1; z = (c - m25) / np.where(sd > 0, sd, np.nan)
        drop5 = np.r_[np.full(5, np.nan), c[5:] / c[:-5] - 1]
        for t in range(200, n - 6):
            if dates[t] < "2016-12-01" or not (tv20[t] >= 1e8):
                continue
            if not (dev[t] <= -0.12 or rsi[t] <= 25 or drop5[t] <= -0.12 or z[t] <= -2.5):
                continue
            e = t + 1
            if not (o[e] > 0):
                continue
            cd = 0
            while cd < 15 and c[t - cd] < c[t - cd - 1]:
                cd += 1
            r = {"code": code, "date": dates[t], "dev": dev[t], "z": z[t], "rsi": rsi[t], "drop5": drop5[t], "drop1": c[t] / c[t - 1] - 1,
                 "cd": cd, "vr": v[t] / va[t] if va[t] and va[t] > 0 else np.nan, "clv": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
                 "above200": c[t] > m200[t], "dev75": c[t] / m75[t] - 1, "tv20": tv20[t], "close": c[t],
                 "gap": o[e] / c[t] - 1, "oc0": c[e] / o[e] - 1, "oo1": o[e + 1] / o[e] - 1, "oc1": c[e + 1] / o[e] - 1,
                 "oc2": c[e + 2] / o[e] - 1, "oc4": c[e + 4] / o[e] - 1, "hi0": h[e] / o[e] - 1, "lo0": l[e] / o[e] - 1}
            for tp in (0.02, 0.03, 0.05):
                r[f"tp{int(tp*100)}"] = tp if h[e] >= o[e] * (1 + tp) else c[e] / o[e] - 1
                for st in (0.03, 0.05):
                    r[f"tp{int(tp*100)}s{int(st*100)}"] = -st if l[e] <= o[e] * (1 - st) else r[f"tp{int(tp*100)}"]
            out.append(r)
    pd.DataFrame(out).to_csv("backtest/rebound.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
