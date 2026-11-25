"""75日線の買いの「第二波」：最初の上げ（+2σ超え）のあと押し目を作り、もう一度上がり始めたところで買う。
setup  P25 : 合図のあと+2σ超え → 終値が25日線以下に押す（75日線より上）→ 終値が25日線を上に抜け直した翌朝
       P25u: 同じ＋その日75日線が5日前より上（上向き）
       P75 : 合図のあと+2σ超え → 安値が75日線の+2%以内まで押す（終値は75日線の−2%より上）→ 終値が5日線を上に抜けた翌朝
       P75u: 同じ＋75日線が上向き
exit   cur : 今と同じ（−1σ割れ損切り→+1σの追いかけ） / m75 : 最初の損切りを「終値が75日線割れ」にしたもの
出力 backtest/buy_wave2.csv.gz"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import buy_candidates as bc  # noqa: E402
LOOK, TRIG = 120, 20


def find(a, t, kind):
    c, l, u2, m25, m75, m5 = (a[x] for x in ("c", "l", "u2", "m25", "m75", "m5"))
    n = len(c); conf = touch = None
    for j in range(t + 1, min(n - 1, t + LOOK)):
        if conf is None:
            if c[j] > u2[j]:
                conf = j
            continue
        if touch is None:
            if kind.startswith("P25") and c[j] <= m25[j] and c[j] > m75[j]:
                touch = j
            elif kind.startswith("P75") and l[j] <= m75[j] * 1.02 and c[j] > m75[j] * 0.98:
                touch = j
            if kind.startswith("P25") and c[j] < m75[j] * 0.98:
                return None
            continue
        if j - touch > TRIG or c[j] < m75[j] * 0.97:
            return None
        line = m25 if kind.startswith("P25") else m5
        if c[j] > line[j] and c[j - 1] <= line[j - 1]:
            if kind.endswith("u") and not m75[j] > m75[j - 5]:
                return None
            return j, j - t, j - touch
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
        a = {"o": df.Open.to_numpy(float), "h": df.High.to_numpy(float), "l": df.Low.to_numpy(float), "c": c,
             "u1": (m25 + sd).to_numpy(), "d1": (m25 - sd).to_numpy(), "m25": m25.to_numpy(), "u2": (m25 + 2 * sd).to_numpy(),
             "u15": (m25 + 1.5 * sd).to_numpy(), "m75": s.rolling(75).mean().to_numpy(), "m5": s.rolling(5).mean().to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        a75 = dict(a, d1=a["m75"])
        pos = {d: i for i, d in enumerate(a["dates"])}
        for sig in g.signal:
            t = pos.get(sig)
            if t is None:
                continue
            for kind in ("P25", "P25u", "P75", "P75u"):
                f = find(a, t, kind)
                if not f:
                    continue
                j, since, wait = f
                for exn, aa in (("cur", a), ("m75", a75)):
                    r = bc.sim(aa, j, None, False, "u1")
                    if r:
                        out.append({"code": code, "signal": sig, "kind": kind, "exn": exn, "rsig": a["dates"][j], "since": since,
                                    "wait": wait, "entry": r["entry"], "exit": r["exit"], "ret": r["ret"], "days": r["days"]})
    pd.DataFrame(out).to_csv("backtest/buy_wave2.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
