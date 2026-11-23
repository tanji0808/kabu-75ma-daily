"""毎日の新高値の判定（forward_test2.nh_breakout）が、バックテストの候補表と同じ銘柄を拾うかを直近60営業日で確かめる。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import forward_test2 as f  # noqa: E402
master = k.load_prime_list()
prices = k.download_prices(master["コード"].tolist(), period="2y")
bt = pd.read_csv("backtest/newhigh_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
days = sorted(bt.signal.unique())[-60:]
got = []
for code, df in prices.items():
    df = sb.clean(df)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    c, h, v = (df[x].to_numpy(float) for x in ("Close", "High", "Volume"))
    ds = [d.strftime("%Y-%m-%d") for d in df.index]
    for D in days:
        if D not in ds:
            continue
        L = ds.index(D)
        if L < 260:
            continue
        ok, piv, bl = f.nh_breakout(c, h, v, L)
        if ok:
            got.append((code, D))
g = set(got)
b = set(map(tuple, bt[bt.signal.isin(days)][["code", "signal"]].values))
b = {x for x in b if x[0] in prices}
print(f"直近60営業日：毎日の判定 {len(g)}件・バックテスト {len(b)}件・共通 {len(g & b)}件")
print("毎日の判定だけ", sorted(g - b)[:15])
print("バックテストだけ", sorted(b - g)[:15])
# 市場の幅（新高値ブレイクの件数）を、バックテストの候補表から数えたものと比べる
cl = {}
for code, df in prices.items():
    df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize(); cl[code] = df
full = pd.read_csv("backtest/newhigh_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "entry"]).dropna()
full = full[full.code.isin(cl)]
ent = pd.to_datetime(full.entry)
for D in days[-40::8]:
    T = pd.Timestamp(D)
    bt_n = int(((ent >= T - pd.Timedelta(days=30)) & (ent <= T - pd.Timedelta(days=6))).sum())
    print(f"件数 {D}: 毎日の計算 {f.nb_count(cl, T)} ・バックテスト {bt_n}")
