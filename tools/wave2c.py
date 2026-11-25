"""第二波の買いのタイミングのいろいろ。第一波＝合図のあと終値が+2σを超える。損切りは「終値が75日線割れ」→+1σの追いかけ（共通）。
押し目 P10=終値が10日線以下 / P25=終値が25日線以下 / Pd1=終値が−1σ以下（どれも75日線の−2%より上を保つ）
きっかけ dip=押した翌朝すぐ / m5=終値が5日線を上抜け / ph=終値が前日の高値超え / m25=終値が25日線を上抜け /
         u1=終値が+1σを上抜け / peak=終値が第一波の高値（終値）を超える / vol=陽線で出来高が20日平均の1.5倍以上
出力 backtest/buy_wave2c.csv.gz"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import buy_candidates as bc  # noqa: E402
LOOK = 120
PULLS = ("P10", "P25", "Pd1")
TRIGS = {"dip": 1, "m5": 20, "ph": 20, "m25": 20, "u1": 30, "peak": 40, "vol": 20}


def find(a, t, pull, trig):
    o, c, h, m75, v, va = a["o"], a["c"], a["h"], a["m75"], a["v"], a["va"]
    pl = {"P10": a["m10"], "P25": a["m25"], "Pd1": a["d1"]}[pull]
    n = len(c); conf = touch = None; peak = -1
    for j in range(t + 1, min(n - 1, t + LOOK)):
        if c[j] < m75[j] * 0.98:
            return None
        if conf is None:
            if c[j] > a["u2"][j]:
                conf = j
            continue
        if touch is None:
            peak = max(peak, c[j])
            if c[j] <= pl[j]:
                touch = j
                if trig == "dip":
                    return j
            continue
        if j - touch > TRIGS[trig]:
            return None
        x = {"m5": c[j] > a["m5"][j] and c[j - 1] <= a["m5"][j - 1],
             "ph": c[j] > h[j - 1],
             "m25": c[j] > a["m25"][j] and c[j - 1] <= a["m25"][j - 1],
             "u1": c[j] > a["u1"][j] and c[j - 1] <= a["u1"][j - 1],
             "peak": c[j] > peak,
             "vol": c[j] > o[j] and c[j] > c[j - 1] and v[j] >= 1.5 * va[j]}[trig]
        if x:
            return j
    return None


def main():
    cand = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    cand = cand[cand.signal >= "2016-06-01"]
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, g in cand.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        c = df.Close.to_numpy(float); s = pd.Series(c); m25 = s.rolling(25).mean(); sd = s.rolling(25).std(ddof=0)
        vv = df.Volume.astype(float)
        a = {"o": df.Open.to_numpy(float), "h": df.High.to_numpy(float), "l": df.Low.to_numpy(float), "c": c,
             "u1": (m25 + sd).to_numpy(), "d1": (m25 - sd).to_numpy(), "m25": m25.to_numpy(), "u2": (m25 + 2 * sd).to_numpy(),
             "m75": s.rolling(75).mean().to_numpy(), "m5": s.rolling(5).mean().to_numpy(), "m10": s.rolling(10).mean().to_numpy(),
             "v": vv.to_numpy(), "va": vv.rolling(20).mean().shift(1).to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        a75 = dict(a, d1=a["m75"])
        pos = {d: i for i, d in enumerate(a["dates"])}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None:
                continue
            for pull in PULLS:
                for trig in TRIGS:
                    j = find(a, t, pull, trig)
                    if j is None:
                        continue
                    r = bc.sim(a75, j, None, False, "u1")
                    if r:
                        out.append({"code": code, "signal": sig, "kind": f"{pull}_{trig}", "rsig": a["dates"][j],
                                    "entry": r["entry"], "exit": r["exit"], "ret": r["ret"], "days": r["days"]})
    pd.DataFrame(out).to_csv("backtest/buy_wave2c.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
