"""買い・空売りの候補の合図の日の終値（100株単位の計算用）。出力 backtest/sig_close.csv.gz（code, signal, close）"""
import os, sys
import pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402


def main():
    a = pd.read_csv("backtest/buy_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    b = pd.read_csv("backtest/short_candidates_11y.csv.gz", dtype={"code": str}, usecols=["code", "signal"])
    cand = pd.concat([a, b]).drop_duplicates()
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, g in cand.groupby("code"):
        df = prices.get(code)
        if df is None:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize().strftime("%Y-%m-%d")
        cl = df.Close
        for s in g.signal:
            if s in cl.index:
                out.append({"code": code, "signal": s, "close": float(cl[s])})
    pd.DataFrame(out).to_csv("backtest/sig_close.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
