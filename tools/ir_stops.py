"""IRトレードの損切りのいろいろ（良いニュース＋その日+5%・高値近くで引けた候補だけ）。r+1 の寄りで買い、最長60営業日。
出力 backtest/ir_stops.csv.gz（code, rdate, {名前}_ret/_exit）"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import newhigh_candidates as nh  # noqa: E402
V = {"h60": dict(stop=0.99)}
for st in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
    V[f"i{int(st*100)}"] = dict(stop=st)
for st in (0.10, 0.12, 0.15, 0.20):
    V[f"c{int(st*100)}"] = dict(stop=st, cstop=True)
V["ma25"] = dict(stop=0.99, ma=25); V["ma50"] = dict(stop=0.99, ma=50)
V["i20ma50"] = dict(stop=0.20, ma=50)
for d_, th in ((10, 0.0), (20, 0.0), (20, 0.03), (30, 0.0)):
    V[f"ts{d_}_{int(th*100)}"] = dict(stop=0.99, ts=(d_, th))
    V[f"i15ts{d_}_{int(th*100)}"] = dict(stop=0.15, ts=(d_, th))
V["i15be10"] = dict(stop=0.15, be=0.10); V["i12be10"] = dict(stop=0.12, be=0.10); V["i20be15"] = dict(stop=0.20, be=0.15)
V["i15a15t15"] = dict(stop=0.15, arm=0.15, armlock=0.0, armtrail=0.15)


def main():
    E = pd.read_csv("backtest/ir_events.csv.gz", dtype={"code": str})
    E = E[(E.rdate >= "2017-01-01") & (E.tv20 >= 3e7) & (E.day >= 0.05) & (E.clv >= 0.7) & E.nentry.notna()]
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, g in E.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        c = df.Close.to_numpy(float); s = pd.Series(c)
        a = {"o": df.Open.to_numpy(float), "h": df.High.to_numpy(float), "l": df.Low.to_numpy(float), "c": c,
             "ma25": s.rolling(25).mean().to_numpy(), "ma50": s.rolling(50).mean().to_numpy(), "pivot": np.full(len(c), np.nan),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        pos = {d: i for i, d in enumerate(a["dates"])}
        for rd in g.rdate.unique():
            t = pos.get(rd)
            if t is None:
                continue
            row = {"code": code, "rdate": rd}
            for nm, prm in V.items():
                x = nh.sim(a, t, hold=60, **prm)
                if x:
                    row[nm + "_ret"], row[nm + "_exit"] = x["ret"], x["exit"]
            out.append(row)
    pd.DataFrame(out).to_csv("backtest/ir_stops.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
