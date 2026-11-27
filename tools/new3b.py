"""新しい手法の材料（4つ分）。出力 backtest/new3b_*.csv.gz
D 窓あけ下げ（ギャップダウン）：寄りが前日終値の−5%以下。寄りで買って、その日〜数日の値動き（寄り値に対する比）を保存。
E 決算前：決算発表の直前の引けまでの値動き（N営業日前の引け→発表前最後の引け）と、発表翌日の寄り。
F 年末の節税売りの反発：年末のN営業日前の引けで、その年に大きく下げた銘柄を買い、1月の各日で売った場合。
G ストップ高引け：その日+10%以上で高値引け。翌日以降の値動き（終値に対する比）を保存（買い・空売りの両方を後で試す）。
"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_rel_pit as sp  # noqa: E402

P = 10   # 何日先まで値動きを保存するか


def path(o, h, l, c, t0, base, row, pre):
    n = len(c)
    for j in range(P + 1):
        i = t0 + j
        if i >= n:
            break
        row[f"{pre}o{j}"], row[f"{pre}h{j}"], row[f"{pre}l{j}"], row[f"{pre}c{j}"] = o[i] / base, h[i] / base, l[i] / base, c[i] / base


def main():
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    fins = sp.load_fins()
    D, E, F, G = [], [], [], []
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) < 300:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        s = pd.Series(c)
        m25 = s.rolling(25).mean().to_numpy(); m75 = s.rolling(75).mean().to_numpy()
        hi250 = pd.Series(h).rolling(250).max().to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().shift(1).to_numpy()
        va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        dates = df.index.strftime("%Y-%m-%d").tolist(); n = len(c)
        for t in range(260, n - 1):
            if dates[t] < "2017-01-01" or not (tv20[t] >= 5e7):
                continue
            # D ギャップダウン
            gap = o[t] / c[t - 1] - 1
            if gap <= -0.05:
                r = {"code": code, "date": dates[t], "gap": gap, "tv20": tv20[t], "pc": c[t - 1],
                     "dev25": c[t - 1] / m25[t - 1] - 1, "dev75": c[t - 1] / m75[t - 1] - 1, "ret5": c[t - 1] / c[t - 6] - 1,
                     "ret20": c[t - 1] / c[t - 21] - 1, "dd250": c[t - 1] / hi250[t - 1] - 1}
                path(o, h, l, c, t, o[t], r, "")
                D.append(r)
            # G ストップ高引け（+10%以上で高値引け）
            day = c[t] / c[t - 1] - 1
            if day >= 0.10 and c[t] >= h[t] * 0.999:
                r = {"code": code, "date": dates[t], "day": day, "tv20": tv20[t], "vr": v[t] / va[t] if va[t] else np.nan,
                     "dev25": c[t] / m25[t] - 1, "ret20": c[t - 1] / c[t - 21] - 1, "near250": c[t] / hi250[t] - 1, "close": c[t]}
                path(o, h, l, c, t, c[t], r, "")
                G.append(r)
        # E 決算前
        for rec in fins.get(code, []):
            if "FinancialStatements" not in str(rec.get("DocType")):
                continue
            dd = str(rec["DiscDate"])[:10]
            if dd < "2017-02-01":
                continue
            tm = str(rec.get("DiscTime") or "")
            i0 = int(np.searchsorted(np.array(dates), dd))   # 発表日（またはその次の営業日）
            if i0 >= n or i0 < 40:
                continue
            last = i0 if (dates[i0] == dd and tm >= "15:00") else i0 - 1   # 発表前の最後の引け
            if last < 30 or not (tv20[last] >= 5e7) or last + 1 >= n:
                continue
            r = {"code": code, "disc": dd, "time": tm, "per": rec.get("CurPerType"), "last": dates[last], "tv20": tv20[last],
                 "react": o[last + 1] / c[last] - 1, "react_c": c[last + 1] / c[last] - 1}
            for N in (1, 2, 3, 5, 7, 10, 15, 20):
                r[f"b{N}"] = c[last] / c[last - N] - 1
                r[f"b{N}d"] = dates[last - N]
            r["pre20"] = c[last - 20] / c[last - 40] - 1
            r["dev25"] = c[last - 5] / m25[last - 5] - 1
            E.append(r)
        # F 年末の反発
        idx = df.index
        for y in range(2016, 2026):
            dec = [i for i, d in enumerate(idx) if d.year == y and d.month == 12]
            jan = [i for i, d in enumerate(idx) if d.year == y + 1 and d.month == 1]
            yr = [i for i, d in enumerate(idx) if d.year == y]
            if len(dec) < 10 or len(jan) < 15 or not yr:
                continue
            for N in (1, 3, 5, 8, 10, 15):
                b = dec[-N] if N <= len(dec) else dec[0]
                if not (tv20[b] >= 3e7):
                    continue
                r = {"code": code, "y": y, "N": N, "buy": dates[b], "tv20": tv20[b], "ytd": c[b] / c[yr[0]] - 1,
                     "dd250": c[b] / hi250[b] - 1, "ret20": c[b] / c[b - 20] - 1, "dev25": c[b] / m25[b] - 1}
                r["x_dec"] = c[dec[-1]] / c[b] - 1          # 大納会の引け
                for j in (0, 2, 4, 9, 14):
                    if j < len(jan):
                        r[f"x_o{j + 1}"] = o[jan[j]] / c[b] - 1
                        r[f"x_c{j + 1}"] = c[jan[j]] / c[b] - 1
                        r[f"x_c{j + 1}d"] = dates[jan[j]]
                r["x_cjan"] = c[jan[-1]] / c[b] - 1; r["x_cjand"] = dates[jan[-1]]
                r["minl"] = min(l[b + 1:jan[min(14, len(jan) - 1)] + 1]) / c[b] - 1
                F.append(r)
    for nm, X in (("gapdown", D), ("earnings", E), ("yearend", F), ("limitup", G)):
        pd.DataFrame(X).to_csv(f"backtest/new3b_{nm}.csv.gz", index=False, compression="gzip")
    print("保存しました", len(D), len(E), len(F), len(G))


if __name__ == "__main__":
    main()
