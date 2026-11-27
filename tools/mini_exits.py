"""ミニ株版（注文はすべて寄り付き）の売り方：損切り・利確・追いかけはすべて終値で判定し、翌営業日の寄りで売る。
出力 backtest/mini_exits.csv.gz（kind=nh/ir, code, signal, ret, exit）"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402


def run_close(o, c, e, stop, tp=None, arm=None, trail=None, hold=120):
    """e＝買う日（寄りで買う）。終値で判定して翌日の寄りで売る。hold 営業日目の翌日の寄りで手じまい。"""
    n = len(c); entry = o[e]; line = entry * (1 - stop); hc = 0.0
    for j in range(e, min(n, e + hold)):
        hc = max(hc, c[j])
        if arm is not None and hc >= entry * (1 + arm):
            line = max(line, entry, hc * (1 - trail))
        hit = c[j] <= line or (tp is not None and c[j] >= entry * (1 + tp))
        if hit or j == e + hold - 1:
            if j + 1 >= n:
                return None
            return o[j + 1] / entry - 1, j + 1
    return None


def main():
    nhc = pd.read_csv("backtest/newhigh_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    E = pd.read_csv("backtest/ir_events.csv.gz", dtype={"code": str})
    E = E[(E.rdate >= "2017-01-01") & (E.tv20 >= 3e7) & (E.day >= 0.05) & (E.clv >= 0.7) & E.nentry.notna()][["code", "rdate"]].rename(columns={"rdate": "signal"})
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for kind, cand, prm in (("nh", nhc, dict(stop=0.06, arm=0.20, trail=0.20, hold=120)), ("nh25", nhc, dict(stop=0.06, arm=0.25, trail=0.15, hold=120)),
                            ("ir", E, dict(stop=0.10, hold=60))):
        for code, g in cand.groupby("code"):
            df = prices.get(code)
            if df is None:
                continue
            df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            o, c = df.Open.to_numpy(float), df.Close.to_numpy(float)
            dates = [d.strftime("%Y-%m-%d") for d in df.index]; pos = {d: i for i, d in enumerate(dates)}
            for sig in g.signal.unique():
                t = pos.get(sig)
                if t is None or t + 1 >= len(c):
                    continue
                x = run_close(o, c, t + 1, **prm)
                if x:
                    out.append({"kind": kind, "code": code, "signal": sig, "ret": round(x[0], 5), "exit": dates[x[1]]})
    pd.DataFrame(out).to_csv("backtest/mini_exits.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
