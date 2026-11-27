"""市場全体の信用評価損益率（推定）。出力 backtest/margin_pl.csv.gz
J-Quants の週次の信用買い残（data/jq/margin_interest.csv.gz）と各銘柄の株価から、
直近26週（信用の期日6か月）に増えた買い残をその週の株価で買ったとみなして平均の買値を出し、
（今の株価−平均の買値）×買い残 の合計 ÷ 平均の買値×買い残 の合計 を「評価損益率」とする。"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402


def main():
    M = pd.read_csv("data/jq/margin_interest.csv.gz", dtype={"Code": str}, usecols=["Date", "Code", "LongVol"])
    M["code"] = M.Code.str[:4]; M["LongVol"] = pd.to_numeric(M.LongVol, errors="coerce")
    M = M.groupby(["code", "Date"], as_index=False).LongVol.sum()
    codes = sorted(set(M.code))
    prices = k.download_prices(codes, period="11y")
    parts = []
    for code, g in M.groupby("code"):
        df = prices.get(code)
        if df is None or len(df) < 60:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        c = df["Close"].astype(float)
        g = g.sort_values("Date").copy(); d = pd.to_datetime(g.Date)
        g["P"] = c.reindex(c.index.union(d)).ffill().reindex(d).to_numpy()
        g = g[g.P.notna() & g.LongVol.notna()]
        if len(g) < 30:
            continue
        dl = g.LongVol.diff().clip(lower=0).fillna(0)
        num = (dl * g.P).rolling(26, min_periods=8).sum(); den = dl.rolling(26, min_periods=8).sum()
        cost = (num / den).where(den > 0, g.P.rolling(26, min_periods=8).mean())
        g["pl"] = (g.P - cost) * g.LongVol; g["base"] = cost * g.LongVol
        parts.append(g[["Date", "code", "pl", "base"]].dropna())
    A = pd.concat(parts)
    A["r"] = A.pl / A.base
    w = A.groupby("Date").agg(pl=("pl", "sum"), base=("base", "sum"), n=("code", "size"))
    w["rate"] = w.pl / w.base * 100
    B = A[(A.r > -0.45) & (A.r < 0.8)]          # 株式分割などで株価と買い残の単位がずれた銘柄を外す
    w2 = B.groupby("Date").agg(pl2=("pl", "sum"), base2=("base", "sum"), n2=("code", "size"))
    w = w.join(w2); w["rate2"] = w.pl2 / w.base2 * 100
    w["med"] = A.groupby("Date").r.median() * 100
    print("外した行", len(A) - len(B), "／", len(A))
    w.reset_index().to_csv("backtest/margin_pl.csv.gz", index=False, compression="gzip")
    print("保存しました", len(w)); print(w.tail(10))


if __name__ == "__main__":
    main()
