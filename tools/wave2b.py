"""第二波（+2σ超え→終値が25日線まで押す→25日線を抜け直した翌朝に買う、20日以内）の損切りのいろいろ。
最初の損切りの線だけを変え、損切り幅ぶん上がったら利益確保→+1σの追いかけ（または25日線・10日安値の追いかけ）は共通。
出力 backtest/buy_wave2b.csv.gz（code, signal, rsig, stop, ret, entry, exit, days）"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import buy_candidates as bc  # noqa: E402
from tools.wave2 import find25  # noqa: E402


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
        c = df.Close.to_numpy(float); h = df.High.to_numpy(float); l = df.Low.to_numpy(float)
        s = pd.Series(c); m25 = s.rolling(25).mean(); sd = s.rolling(25).std(ddof=0)
        tr = pd.concat([pd.Series(h - l), (pd.Series(h) - s.shift()).abs(), (pd.Series(l) - s.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().to_numpy()
        a = {"o": df.Open.to_numpy(float), "h": h, "l": l, "c": c,
             "u1": (m25 + sd).to_numpy(), "d1": (m25 - sd).to_numpy(), "m25": m25.to_numpy(), "u2": (m25 + 2 * sd).to_numpy(),
             "u15": (m25 + 1.5 * sd).to_numpy(), "m75": s.rolling(75).mean().to_numpy(), "m50": s.rolling(50).mean().to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        n = len(c)
        pos = {d: i for i, d in enumerate(a["dates"])}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None:
                continue
            f = find25(a, t, "u2", 1.0, 20, False)
            if not f:
                continue
            j, since, wait = f; touch = j - wait
            swing = l[touch:j + 1].min()
            const = lambda v: np.full(n, v)  # noqa: E731
            lines = {"m75": a["m75"], "m50": a["m50"], "d1": a["d1"], "d15": (m25 - 1.5 * sd).to_numpy(), "d2": (m25 - 2 * sd).to_numpy(),
                     "fix5": const(c[j] * 0.95), "fix7": const(c[j] * 0.93), "fix10": const(c[j] * 0.90), "fix12": const(c[j] * 0.88),
                     "swing": const(swing), "swing2": const(swing * 0.98), "atr2": const(c[j] - 2 * atr[j]), "atr3": const(c[j] - 3 * atr[j])}
            for nm, ln in lines.items():
                for tn in (("u1", "ma25", "low10") if nm in ("m75", "swing", "fix7") else ("u1",)):
                    r = bc.sim(dict(a, d1=ln), j, None, False, tn)
                    if r:
                        out.append({"code": code, "signal": sig, "rsig": a["dates"][j], "stop": nm + ("" if tn == "u1" else "+" + tn),
                                    "entry": r["entry"], "exit": r["exit"], "ret": r["ret"], "days": r["days"]})
    pd.DataFrame(out).to_csv("backtest/buy_wave2b.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
