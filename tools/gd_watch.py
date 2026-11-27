"""窓あけ下げを「前の晩に寄指で注文」する形の検証材料。出力 backtest/gd_watch.csv.gz と gd_watch_n.csv.gz
毎日：前日終値が25日線の−5%以下・直前20日の売買代金平均1億円以上・決算発表から3日以内でない銘柄を「監視リスト」とし、25日線からの離れ（大きい順）で順位を付ける。
そのうち寄りが前日終値の−5%以下だった銘柄（＝前日終値×0.95の寄指が約定）について、寄り値を基準にした3日分の値動きを保存。"""
import os, sys, bisect
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import short_rel_pit as sp  # noqa: E402


def main():
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    fins = sp.load_fins()
    W = []   # 監視リストの全行（date, code, dev25, gap, filled）
    P = {}   # 約定した行の値動き
    for code, df in prices.items():
        df = sb.clean(df)
        if len(df) < 300:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        m25 = pd.Series(c).rolling(25).mean().to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().shift(1).to_numpy()
        dates = df.index.strftime("%Y-%m-%d").tolist(); n = len(c)
        ds = sorted(str(r["DiscDate"])[:10] for r in fins.get(code, []) if "FinancialStatements" in str(r.get("DocType")))
        for t in range(260, n):
            if dates[t] < "2017-01-01":
                continue
            dev = c[t - 1] / m25[t - 1] - 1
            if not (dev <= -0.05 and tv20[t] >= 1e8):
                continue
            j = bisect.bisect_left(ds, dates[t])
            dd = (pd.Timestamp(dates[t]) - pd.Timestamp(ds[j - 1])).days if j else 999
            if dd < 3:
                continue
            gap = o[t] / c[t - 1] - 1
            W.append((dates[t], code, round(dev, 4), round(gap, 4)))
            if gap <= -0.05 and gap > -0.5:
                r = {}
                for q in range(3):
                    i = t + q
                    if i < n:
                        r.update({f"o{q}": o[i] / o[t], f"h{q}": h[i] / o[t], f"l{q}": l[i] / o[t], f"c{q}": c[i] / o[t]})
                P[(dates[t], code)] = r
    w = pd.DataFrame(W, columns=["date", "code", "dev25", "gap"])
    w["rank"] = w.groupby("date").dev25.rank(method="first")          # 1＝いちばん25日線から下
    w["n_watch"] = w.groupby("date").code.transform("size")
    f = w[(w.gap <= -0.05) & (w.gap > -0.5)].copy()
    pp = pd.DataFrame([P.get((d, c_), {}) for d, c_ in zip(f.date, f.code)], index=f.index)
    pd.concat([f, pp], axis=1).to_csv("backtest/gd_watch.csv.gz", index=False, compression="gzip")
    w.groupby("date").size().rename("n").to_csv("backtest/gd_watch_n.csv.gz", compression="gzip")
    print("保存しました", len(w), len(f))


if __name__ == "__main__":
    main()
