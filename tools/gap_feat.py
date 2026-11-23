"""75日線の買いと空売りの候補に「翌朝の窓（始値÷合図の日の終値−1、%）」を付ける。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
master = k.load_prime_list()
prices = k.download_prices(master["コード"].tolist(), period="11y")
cl = {}
for code, df in prices.items():
    df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize(); cl[code] = df
for name in ("buy", "short"):
    c = pd.read_csv(f"backtest/{name}_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"]).drop_duplicates()
    out = []
    for r in c.itertuples():
        df = cl.get(r.code)
        if df is None:
            continue
        i = df.index.searchsorted(pd.Timestamp(r.signal))
        if i + 1 < len(df) and df.index[i] == pd.Timestamp(r.signal):
            out.append((r.code, r.signal, (df["Open"].iloc[i + 1] / df["Close"].iloc[i] - 1) * 100))
    pd.DataFrame(out, columns=["code", "signal", "gap_next"]).to_csv(f"backtest/gap_{name}.csv.gz", index=False, compression="gzip")
    print(name, len(out))
