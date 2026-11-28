"""外部要因（米国株・VIX・為替・金利・日経・半導体・原油）の日足を20年分 data/macro.csv.gz に保存する。"""
import pandas as pd, yfinance as yf
T = {"vix": "^VIX", "spx": "^GSPC", "n225": "^N225", "usdjpy": "JPY=X", "us10y": "^TNX", "sox": "^SOX", "oil": "CL=F", "nasdaq": "^IXIC",
     "dow": "^DJI", "nkd": "NKD=F", "niy": "NIY=F"}
out = {}
for k, s in T.items():
    d = yf.download(s, period="20y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    out[k] = d
    print(k, len(d), d.index.min().date(), d.index.max().date())
pd.DataFrame(out).sort_index().to_csv("data/macro.csv.gz", compression="gzip")
