"""第三波：第一波（+2σ超え）→押し目→第二波（第一波の高値を終値で更新）→2回目の押し目→もう一度上がり始めたところで買う。
押し目 P10=終値が10日線以下 / P25=終値が25日線以下（75日線の−2%より上を保つ）
きっかけ vol=陽線で出来高が20日平均の1.5倍以上 / peak=終値が第二波の高値を超える / m25=終値が25日線を上抜け / m5=終値が5日線を上抜け
損切り m75=終値が75日線割れ / d1=−1σ割れ（どちらも利益確保→+1σの追いかけ）。出力 backtest/buy_wave3.csv.gz"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import buy_candidates as bc  # noqa: E402
LOOK = 200
TW = {"vol": 20, "peak": 40, "m25": 20, "m5": 20}


def find(a, t, pull, trig):
    o, c, h, m75, v, va = a["o"], a["c"], a["h"], a["m75"], a["v"], a["va"]
    pl = a["m10"] if pull == "P10" else a["m25"]
    n = len(c); st = 0; peak1 = peak2 = -1; touch = None
    for j in range(t + 1, min(n - 1, t + LOOK)):
        if c[j] < m75[j] * 0.98:
            return None
        if st == 0:                      # 第一波
            if c[j] > a["u2"][j]:
                st = 1
        elif st == 1:                    # 第一波の高値を記録しながら1回目の押し目を待つ
            peak1 = max(peak1, c[j])
            if c[j] <= pl[j]:
                st = 2
        elif st == 2:                    # 第二波：第一波の高値を更新
            if c[j] > peak1:
                st = 3; peak2 = c[j]
            elif j > t + LOOK - 1:
                return None
        elif st == 3:                    # 第二波の高値を記録しながら2回目の押し目を待つ
            peak2 = max(peak2, c[j])
            if c[j] <= pl[j]:
                st = 4; touch = j
        else:
            if j - touch > TW[trig]:
                return None
            x = {"vol": c[j] > o[j] and c[j] > c[j - 1] and v[j] >= 1.5 * va[j],
                 "peak": c[j] > peak2,
                 "m25": c[j] > a["m25"][j] and c[j - 1] <= a["m25"][j - 1],
                 "m5": c[j] > a["m5"][j] and c[j - 1] <= a["m5"][j - 1]}[trig]
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
        pos = {d: i for i, d in enumerate(a["dates"])}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None:
                continue
            for pull in ("P10", "P25"):
                for trig in TW:
                    j = find(a, t, pull, trig)
                    if j is None:
                        continue
                    for exn, aa in (("m75", dict(a, d1=a["m75"])), ("d1", a)):
                        r = bc.sim(aa, j, None, False, "u1")
                        if r:
                            out.append({"code": code, "signal": sig, "kind": f"{pull}_{trig}_{exn}", "rsig": a["dates"][j], "since": j - t,
                                        "entry": r["entry"], "exit": r["exit"], "ret": r["ret"], "days": r["days"]})
    pd.DataFrame(out).to_csv("backtest/buy_wave3.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
