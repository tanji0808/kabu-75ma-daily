"""空売りの買い戻しのいろいろ（候補表の全候補）。出力 backtest/short_exits2.csv.gz（code, signal, {名前}_ret/_exit/_days）。
ret は空売りの利益率（％、貸株料 年1.1% を引いた後）。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
MAX_HOLD, FEE = 60, 0.011
MODES = ["lk25c", "lk25", "notp", "d25c", "a2h5", "a2h3", "a2d1", "half2", "rsi", "lock", "pre15", "a2h5_d25"]


def ssim(A, i, mode):
    o, h, l, c, u1, u15, d1, d2, d25, m25, rsi, dates = (A[x] for x in ("o", "h", "l", "c", "u1", "u15", "d1", "d2", "d25", "m25", "rsi", "dates"))
    n = len(c); e = i + 1
    if e >= n - 1:
        return None
    entry = o[e]
    armed = after2 = False
    stop = None; half_px = None; rlow = False
    ex = px = None
    stop_pre = u15 if mode == "pre15" else u1
    for j in range(e, min(n - 1, e + MAX_HOLD)):
        if armed:
            # 日中の買い戻しライン（前日の値）
            if mode in ("lock",):
                line = d1[j - 1]
            else:
                line = m25[j - 1]
            if after2 and mode in ("a2h5", "a2h3", "half2", "a2h5_d25"):
                kk = 5 if mode != "a2h3" else 3
                line = min(line, h[max(e, j - kk):j].max())
            if after2 and mode == "a2d1":
                line = min(line, d1[j - 1])
            if mode == "lk25" and o[j] <= d2[j - 1]:
                ex, px = j, o[j]; break
            if o[j] >= line:
                ex, px = j, o[j]; break
            if h[j] >= line:
                ex, px = j, line; break
            if mode == "lk25" and l[j] <= d2[j - 1]:
                ex, px = j, d2[j - 1]; break
            if mode == "rsi":
                if rsi[j] < 25: rlow = True
                elif rlow and rsi[j] > 30:
                    ex = j + 1; break
            if c[j] < d2[j]:
                if mode in ("lk25c", "lock", "pre15", "rsi"):
                    ex = j + 1; break
                if mode == "half2" and half_px is None:
                    half_px = o[j + 1]
                after2 = True
            if mode in ("d25c", "a2h5_d25") and c[j] < d25[j]:
                ex = j + 1; break
            continue
        if c[j] < d1[j]:
            armed = True
            if mode in ("lk25c", "lock", "pre15") and c[j] < d2[j]:
                ex = j + 1; break
            if c[j] < d2[j]:
                after2 = True
                if mode == "half2": half_px = o[j + 1]
            continue
        if c[j] > stop_pre[j]:
            ex = j + 1; break
    if ex is None:
        if e + MAX_HOLD <= n - 1:
            ex = e + MAX_HOLD
        else:
            return None
    if px is None:
        px = o[ex]
    if half_px is not None:
        px = (px + half_px) / 2
    days = (pd.Timestamp(dates[ex]) - pd.Timestamp(dates[e])).days
    ret = (entry - px) / entry * 100 - FEE * max(days, 1) / 365 * 100
    return round(ret, 2), dates[ex], days


def main():
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    cand = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"]).drop_duplicates()
    out = []
    for code, g in cand.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        c = df.Close.to_numpy(float); s = pd.Series(c); m = s.rolling(25).mean(); sd = s.rolling(25).std(ddof=0)
        d_ = s.diff(); up = d_.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        A = {"o": df.Open.to_numpy(float), "h": df.High.to_numpy(float), "l": df.Low.to_numpy(float), "c": c,
             "u1": (m + sd).to_numpy(), "u15": (m + 1.5 * sd).to_numpy(), "d1": (m - sd).to_numpy(), "d2": (m - 2 * sd).to_numpy(),
             "d25": (m - 2.5 * sd).to_numpy(), "m25": m.to_numpy(), "rsi": (100 - 100 / (1 + up / dn)).to_numpy(),
             "dates": [d.strftime("%Y-%m-%d") for d in df.index]}
        pos = {d: i for i, d in enumerate(A["dates"])}
        for sd_ in g.signal:
            i = pos.get(sd_)
            if i is None:
                continue
            row = {"code": code, "signal": sd_}
            for md in MODES:
                r = ssim(A, i, md)
                if r:
                    row[f"{md}_ret"], row[f"{md}_exit"], row[f"{md}_days"] = r
            out.append(row)
    pd.DataFrame(out).to_csv("backtest/short_exits2.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
