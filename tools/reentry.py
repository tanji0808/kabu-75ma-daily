"""75日線の買いの「第二波」：売ったあと60営業日以内に終値が+1σを上に抜け直したら翌朝買う。売り方は今と同じ（cur）。
出力 backtest/buy_reentry.csv.gz（code, signal＝最初の合図, wave, rsig＝買い直しの合図の日, gap＝売ってからの営業日, entry, exit, ret, days）。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import buy_candidates as bc  # noqa: E402
WIN, WAVES = 60, 3


def main():
    cand = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal", "cur_exit"])
    cand = cand[cand.cur_exit.notna()]
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
             "u1": (m25 + sd).to_numpy(), "d1": (m25 - sd).to_numpy(), "m25": m25.to_numpy(),
             "u15": (m25 + 1.5 * sd).to_numpy(), "u2": (m25 + 2 * sd).to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        pos = {d: i for i, d in enumerate(a["dates"])}
        u1 = a["u1"]
        for sig, ex in zip(g.signal, g.cur_exit):
            x = pos.get(ex)
            for w in range(2, 2 + WAVES):
                if x is None:
                    break
                hit = None
                for j in range(max(x, 1), min(len(c) - 1, x + WIN)):
                    if c[j] > u1[j] and c[j - 1] <= u1[j - 1]:
                        hit = j; break
                if hit is None:
                    break
                try:
                    r = bc.sim(a, hit, None, False, "u1")
                except Exception as e:  # noqa
                    print("err", code, e); r = None
                if not r:
                    break
                out.append({"code": code, "signal": sig, "wave": w, "rsig": a["dates"][hit], "gap": hit - x,
                            "entry": r["entry"], "exit": r["exit"], "ret": r["ret"], "days": r["days"], "reason": r["reason"],
                            "ext": round((c[hit] / a["m25"][hit] - 1) * 100, 2)})
                x = pos.get(r["exit"])
    pd.DataFrame(out).to_csv("backtest/buy_reentry.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
