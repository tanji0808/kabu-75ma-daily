"""新高値の売り方を細かく（損切り4〜10%×利確15%〜なし×最長保有60/120/250日）。出力 backtest/nh_exit_grid.csv.gz（code, signal, {名前}_ret/_exit）"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import newhigh_candidates as nh  # noqa: E402
GRID = {}
for st in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
    for tp in (0.15, 0.20, 0.25, 0.30, 0.40, 0.60, None):
        GRID[f"g{int(st*100)}_{int(tp*100) if tp else 'x'}"] = dict(stop=st, tp=tp)
    for hd in (60, 250):
        GRID[f"g{int(st*100)}_x_h{hd}"] = dict(stop=st, hold=hd)
        GRID[f"g{int(st*100)}_25_h{hd}"] = dict(stop=st, tp=0.25, hold=hd)


if os.environ.get("NH_GRID") == "arm":   # 途中で損切りラインを引き上げる形
    GRID = {}
    for st in (0.05, 0.06, 0.07):
        for tp in (0.40, 0.60, None):
            for arm in (0.20, 0.25, 0.30):
                for lk in sorted({arm, arm - 0.05, arm - 0.10, 0.10, 0.0}):
                    if lk < 0 or (st != 0.06 and tp != 0.60):
                        continue
                    GRID[f"a{int(st*100)}_{int(tp*100) if tp else 'x'}_{int(arm*100)}_{int(round(lk*100))}"] = dict(stop=st, tp=tp, arm=arm, armlock=lk)
                for tr in (0.10, 0.15, 0.20):
                    if st == 0.06:
                        GRID[f"a{int(st*100)}_{int(tp*100) if tp else 'x'}_{int(arm*100)}_t{int(tr*100)}"] = dict(stop=st, tp=tp, arm=arm, armlock=0.0, armtrail=tr)
OUT = "backtest/nh_exit_arm.csv.gz" if os.environ.get("NH_GRID") == "arm" else "backtest/nh_exit_grid.csv.gz"


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
        c = df.Close.to_numpy(float); s = pd.Series(c); m25 = s.rolling(25).mean(); sd = s.rolling(25).std(ddof=0)
        hs = pd.Series(df.High.to_numpy(float))
        a = {"o": df.Open.to_numpy(float), "h": df.High.to_numpy(float), "l": df.Low.to_numpy(float), "c": c,
             "ma25": m25.to_numpy(), "ma50": s.rolling(50).mean().to_numpy(), "pivot": hs.rolling(250).max().shift(1).to_numpy(),
             "d1": (m25 - sd).to_numpy(), "u1": (m25 + sd).to_numpy(), "m25": m25.to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        pos = {d: i for i, d in enumerate(a["dates"])}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None:
                continue
            row = {"code": code, "signal": sig}
            for nm, prm in GRID.items():
                r = nh.sim(a, t, **prm)
                if r:
                    row[nm + "_ret"], row[nm + "_exit"] = r["ret"], r["exit"]
            out.append(row)
    pd.DataFrame(out).to_csv(OUT, index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
