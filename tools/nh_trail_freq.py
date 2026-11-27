"""新高値の追いかけ（+25%に届いたら終値の最高値−15%に逆指値）を、毎日ではなく何営業日かに1回だけ見直す場合。
出力 backtest/nh_trail_freq.csv.gz（code, signal, {名前}_ret/_exit）。損切り−6%・最長120営業日は同じ。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
V = {}
for f in (1, 5, 10, 21):
    for tr in (0.10, 0.15, 0.20):
        for arm in (0.20, 0.25):
            V[f"f{f}_a{int(arm*100)}_t{int(tr*100)}"] = (f, arm, tr)


def run(o, h, l, c, e, F, arm, tr, stop=0.06, mh=120):
    n = len(c); entry = o[e]; line = entry * (1 - stop); hc = 0.0
    for j in range(e, min(n, e + mh)):
        if j > e and o[j] <= line:
            return o[j] / entry - 1, j
        if l[j] <= line:
            return line / entry - 1, j
        hc = max(hc, c[j])
        if (j - e) % F == F - 1 or F == 1:          # 見直しの日（引けのあと）に逆指値を入れ直す
            if hc >= entry * (1 + arm):
                line = max(line, entry, hc * (1 - tr))
    j = min(n, e + mh) - 1
    return c[j] / entry - 1, j


def main():
    cand = pd.read_csv("backtest/newhigh_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, g in cand.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
        dates = [d.strftime("%Y-%m-%d") for d in df.index]; pos = {d: i for i, d in enumerate(dates)}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None or t + 1 >= len(c):
                continue
            row = {"code": code, "signal": sig}
            for nm, (F, arm, tr) in V.items():
                r, j = run(o, h, l, c, t + 1, F, arm, tr)
                row[nm + "_ret"], row[nm + "_exit"] = round(r, 5), dates[j]
            out.append(row)
    pd.DataFrame(out).to_csv("backtest/nh_trail_freq.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
