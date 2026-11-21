"""
引き上げ式ロスカットの取引で、「最初のロスカット（−1σ）で損切りになった取引」と
「利益確保ラインまで上がった取引」の違いを調べる。

特徴量は、合図の日（買う前日）の終値の時点で分かるもの（＋参考として事後の市場の動き）。
1つずつの比較（5分位ごとの損切り率）と、時期を分けた検証（2024年までで学習→2025年以降で確認）を行う。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kabu_75ma_search as k  # noqa: E402

t = pd.read_csv("backtest/trades_trail.csv", dtype={"code": str})
t["stop"] = (t["reason"] == "m1s").astype(int)
print("取引", len(t), "損切り率", round(t["stop"].mean() * 100, 1))

# ---------------------------------------------------------------- 株価・指数
prices = k.download_prices(sorted(t["code"].unique()), period="6y")
import yfinance as yf  # noqa: E402
idx = {}
for sym in ["^DJI", "^IXIC", "^GSPC", "^VIX", "1306.T", "JPY=X"]:
    try:
        d = yf.download(sym, period="6y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        idx[sym] = d[["Open", "Close"]].dropna()
    except Exception as e:
        print("指数の取得に失敗", sym, e)

def us_move(sym, day, pts=False):
    """日本の合図の日と同じ日付のアメリカの取引（日本時間では買う日の朝までに終わる）の変化。"""
    d = idx.get(sym)
    if d is None or day not in d.index:
        return np.nan
    i = d.index.get_loc(day)
    if i == 0:
        return np.nan
    a, b = d["Close"].iloc[i - 1], d["Close"].iloc[i]
    return float(b - a) if pts else float((b / a - 1) * 100)

def us_after(sym, day, n=5, thr=-500):
    d = idx.get(sym)
    if d is None:
        return np.nan
    ch = d["Close"].diff()
    w = ch[(ch.index > day) & (ch.index <= day + pd.Timedelta(days=n * 7 // 5 + 2))]
    return float((w <= thr).any())

topix = idx.get("1306.T")
if topix is not None:
    tc = topix["Close"]
    topix_ma75 = tc.rolling(75).mean()

with open(os.environ.get("FUND_PATH", "data/fundamentals.json"), encoding="utf-8") as f:
    fund = json.load(f)

cache = {}
def ind(code):
    if code in cache:
        return cache[code]
    df = prices.get(code)
    if df is None:
        cache[code] = None
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    c, v = df["Close"], df["Volume"]
    m25, m75 = c.rolling(25).mean(), c.rolling(75).mean()
    sd = c.rolling(25).std(ddof=0)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    cache[code] = dict(df=df, c=c, o=df["Open"], v=v, m25=m25, m75=m75, sd=sd, rsi=rsi)
    return cache[code]

rows = []
for _, r in t.iterrows():
    I = ind(r["code"])
    if I is None:
        continue
    day = pd.Timestamp(r["signal"])
    if day not in I["c"].index:
        continue
    i = I["c"].index.get_loc(day)
    if i < 130 or i + 1 >= len(I["c"]):
        continue
    c, o, v = I["c"].to_numpy(), I["o"].to_numpy(), I["v"].to_numpy()
    m25, m75, sd, rsi = I["m25"].to_numpy(), I["m75"].to_numpy(), I["sd"].to_numpy(), I["rsi"].to_numpy()
    f = fund.get(r["code"], {})
    met = k._fund_metrics(f, c[i])
    x = {
        "stop": r["stop"], "ret": r["ret"], "signal": r["signal"], "code": r["code"], "sector": r["sector"],
        # ファンダメンタル（今の財務データ＋その日の株価。先読みの影響あり）
        "Fランク": r["frank"],
        "PER": met["per"], "PBR": met["pbr"], "ROE": met["roe"], "営業利益率": met["om"],
        "売上成長率": met["rg"], "利益成長率": met["eg"], "配当利回り": met["dy"], "負債資本倍率": met["de"],
        "時価総額(億円)": (f.get("sh") or np.nan) * c[i] / 1e8,
        "株価": c[i],
        # 値動き（合図の日まで）
        "75日線からの上離れ(%)": (c[i] / m75[i] - 1) * 100,
        "75日線の傾き(20日,%)": (m75[i] / m75[i - 20] - 1) * 100,
        "25日線の傾き(5日,%)": (m25[i] / m25[i - 5] - 1) * 100,
        "出来高倍率(20日平均比)": v[i] / np.mean(v[i - 20:i]) if np.mean(v[i - 20:i]) > 0 else np.nan,
        "RSI": rsi[i],
        "バンド幅(±2σ÷25日線,%)": 4 * sd[i] / m25[i] * 100,
        "最初のロスカット幅(%)": (o[i + 1] - (m25[i] - sd[i])) / o[i + 1] * 100,
        "+1σとの位置(%)": (c[i] / (m25[i] + sd[i]) - 1) * 100,
        "直前5日の上昇(%)": (c[i] / c[i - 5] - 1) * 100,
        "直前20日の上昇(%)": (c[i] / c[i - 20] - 1) * 100,
        "半年高値からの下落(%)": (c[i] / I["df"]["High"].to_numpy()[i - 120:i + 1].max() - 1) * 100,
        "合図の日の上昇(%)": (c[i] / c[i - 1] - 1) * 100,
        "翌朝の窓(%)": (o[i + 1] / c[i] - 1) * 100,
        # 市場環境（買う前の晩のアメリカ／日本）
        "前夜のダウ(ドル)": us_move("^DJI", day, pts=True),
        "前夜のダウ(%)": us_move("^DJI", day),
        "前夜のナスダック(%)": us_move("^IXIC", day),
        "前夜のS&P500(%)": us_move("^GSPC", day),
        "VIX": float(idx["^VIX"]["Close"].asof(day)) if "^VIX" in idx else np.nan,
        "ドル円の前夜変化(%)": us_move("JPY=X", day),
        "TOPIXの75日線からの位置(%)": float((tc.asof(day) / topix_ma75.asof(day) - 1) * 100) if topix is not None else np.nan,
        "TOPIXの直前5日(%)": float((tc.asof(day) / tc.asof(day - pd.Timedelta(days=7)) - 1) * 100) if topix is not None else np.nan,
        # 参考：事後（買った後に起きたこと。事前には分からない）
        "[事後]5日以内にダウ500ドル超の下落": us_after("^DJI", day),
    }
    rows.append(x)
d = pd.DataFrame(rows)
d["signal"] = pd.to_datetime(d["signal"])
print("分析できた取引", len(d))
base = float(d["stop"].mean() * 100)

# ---------------------------------------------------------------- 1つずつの比較（5分位）
uni = []
feats = [c for c in d.columns if c not in ("stop", "ret", "signal", "code", "sector")]
for fcol in feats:
    s = d[[fcol, "stop", "ret", "signal"]].dropna()
    if len(s) < 200 or s[fcol].nunique() < 2:
        continue
    if s[fcol].nunique() <= 2:
        groups = s.groupby(fcol)
        bins = [(f"{fcol}={g}", gg) for g, gg in groups]
    else:
        q = pd.qcut(s[fcol], 5, duplicates="drop")
        bins = [(f"{iv.left:.2f}〜{iv.right:.2f}", gg) for iv, gg in s.groupby(q, observed=True)]
    tbl = []
    for lab, gg in bins:
        early, late = gg[gg["signal"] < "2025-01-01"], gg[gg["signal"] >= "2025-01-01"]
        tbl.append({"range": lab, "n": int(len(gg)), "stop": round(float(gg["stop"].mean() * 100), 1),
                    "ret": round(float(gg["ret"].mean()), 2),
                    "stop_early": round(float(early["stop"].mean() * 100), 1) if len(early) else None,
                    "stop_late": round(float(late["stop"].mean() * 100), 1) if len(late) else None,
                    "n_late": int(len(late))})
    spread = max(x["stop"] for x in tbl) - min(x["stop"] for x in tbl)
    # 前半と後半で、損切り率の並び（低い→高い）が同じ向きか
    e = [x["stop_early"] for x in tbl if x["stop_early"] is not None]
    l_ = [x["stop_late"] for x in tbl if x["stop_late"] is not None]
    consistent = None
    if len(e) == len(l_) == len(tbl) and len(tbl) >= 3:
        consistent = round(float(pd.Series(e).corr(pd.Series(l_), method="spearman")), 2)
    uni.append({"feature": fcol, "spread": round(spread, 1), "consistency": consistent, "bins": tbl})
uni.sort(key=lambda x: -x["spread"])

# 業種
sec = d.groupby("sector").agg(n=("stop", "size"), stop=("stop", "mean"), ret=("ret", "mean"))
sec = sec[sec["n"] >= 40].sort_values("stop")
sector = [{"sector": s_, "n": int(r_["n"]), "stop": round(r_["stop"] * 100, 1), "ret": round(r_["ret"], 2)} for s_, r_ in sec.iterrows()]

# ---------------------------------------------------------------- まとめて分析（時期を分けて検証）
model = {}
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = d[[c for c in feats if not c.startswith("[事後]")]]
    y = d["stop"]
    tr, te = d["signal"] < "2025-01-01", d["signal"] >= "2025-01-01"
    X = X[[col for col in X.columns if X.loc[tr, col].nunique(dropna=True) >= 2]]   # 値が1種類しかない列は除く
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                         random_state=0)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    # 重要度：その項目だけをバラバラに並べ替えたとき、後半の期間の予測精度（AUC）がどれだけ落ちるか
    rs = np.random.default_rng(0)
    Xte = X[te].copy()
    imp = []
    for col in X.columns:
        drops = []
        for _ in range(10):
            Xp = Xte.copy()
            Xp[col] = rs.permutation(Xp[col].to_numpy())
            drops.append(auc - roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1]))
        imp.append((col, float(np.mean(drops))))
    imp.sort(key=lambda z: -z[1])
    # 予測の確率で5つに分け、後半の期間での損切り率と損益
    qq = pd.qcut(p, 5, labels=False, duplicates="drop")
    tt = d[te].assign(q=qq)
    by_q = [{"group": int(g) + 1, "n": int(len(x)), "stop": round(float(x["stop"].mean() * 100), 1),
             "ret": round(float(x["ret"].mean()), 2)} for g, x in tt.groupby("q")]
    model = {"auc_test": round(float(auc), 3), "n_train": int(tr.sum()), "n_test": int(te.sum()),
             "importance": [{"feature": a, "auc_drop": round(float(b), 4)} for a, b in imp[:15]],
             "test_by_predicted_risk": by_q}
except Exception as e:
    import traceback
    traceback.print_exc()
    model = {"error": str(e)}

out = {"n": int(len(d)), "base_stop_rate": round(base, 1), "univariate": uni, "sector": sector, "model": model}
os.makedirs("backtest", exist_ok=True)
with open("backtest/factor_analysis.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print("完了", out["n"], "AUC", model.get("auc_test"))
