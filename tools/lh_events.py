"""大量保有報告書の提出（data/jq/large_holdings.csv.gz）のあとの値動き。出力 backtest/lh_events.csv.gz
1行＝1書類（保有者は提出者の先頭）。翌営業日の寄りで買った場合の値動きと、いくつかの売り方の結果。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402


def run(o, l, c, e, stop, hold, tp=None):
    n = len(c); entry = o[e]; sl = entry * (1 - stop)
    for j in range(e, min(n, e + hold)):
        if j > e and o[j] <= sl:
            return o[j] / entry - 1, j
        if l[j] <= sl:
            return -stop, j
        if tp and c[j] >= entry * (1 + tp):
            return c[j] / entry - 1, j
    j = min(n, e + hold) - 1
    return (c[j] / entry - 1, j) if j >= e else (None, None)


def main():
    L = pd.read_csv("data/jq/large_holdings.csv.gz", dtype=str)
    L["code4"] = L.Code.str[:4]
    for col in ("TotalShsRatio", "TotalShsRatioLast", "ShsRatio", "ShsRatioLast", "OnMktBuy", "OnMktSell", "TotalOutStks"):
        L[col] = pd.to_numeric(L[col], errors="coerce")
    D = L.sort_values(["DocId"]).groupby("DocId").agg(code=("code4", "first"), sub=("SubDate", "first"), subt=("SubTime", "first"),
        typ=("LargeHldgTypeCode", "first"), tot=("TotalShsRatio", "first"), totl=("TotalShsRatioLast", "first"),
        hldr=("HldrName", "first"), htype=("LargeHldrTypeCode", "first"), purp=("HldgPurp", "first"), imp=("ImpProp", "first"),
        buy=("OnMktBuy", "sum"), sell=("OnMktSell", "sum"), nh=("HldrName", "size"), outs=("TotalOutStks", "first")).reset_index()
    master = k.load_prime_list()
    prices = k.download_prices(sorted(set(D.code) | set(master["コード"])), period="6y")
    out = []
    for code, g in D.groupby("code"):
        df = prices.get(code)
        if df is None or len(df) < 60:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        dates = df.index.strftime("%Y-%m-%d").to_numpy()
        tv20 = pd.Series(c * v).rolling(20).mean().to_numpy()
        hi250 = pd.Series(h).rolling(250, min_periods=60).max().to_numpy()
        for r in g.itertuples():
            e = int(np.searchsorted(dates, r.sub, side="right"))      # 提出日の翌営業日に寄りで買う
            if e >= len(c) or e < 25:
                continue
            x = dict(r._asdict()); x.pop("Index", None)
            x.update(entry_date=dates[e], tv20=tv20[e - 1], pre20=c[e - 1] / c[e - 21] - 1, dd250=c[e - 1] / hi250[e - 1] - 1,
                     gap=o[e] / c[e - 1] - 1, close=c[e - 1])
            for H in (1, 5, 20, 60, 120):
                j = e + H - 1
                x[f"r{H}"] = c[j] / o[e] - 1 if j < len(c) else np.nan
            for nm, st, hd, tp in (("s10h60", .10, 60, None), ("s10h120", .10, 120, None), ("s15h120", .15, 120, None), ("s10h20", .10, 20, None), ("s8h60tp30", .08, 60, .30)):
                rr, jj = run(o, l, c, e, st, hd, tp)
                x[nm], x[nm + "_exit"] = rr, (dates[jj] if jj is not None else None)
            out.append(x)
    pd.DataFrame(out).to_csv("backtest/lh_events.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
