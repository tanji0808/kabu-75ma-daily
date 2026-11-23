#!/usr/bin/env python3
"""バックテストの取引一覧（backtest/mtm_trades.csv）を毎日の終値で評価し、含み損益を入れた資産の動きを出す。
出力: backtest/mtm_result.json（ルールごと：最大の落ち込み（含み損込み）・最大の含み損合計・1銘柄の最大含み損など）
      backtest/mtm_equity.csv.gz（毎日の確定損益・含み損益）"""
import json, os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402

COST = 0.002
tr = pd.read_csv("backtest/mtm_trades.csv", dtype={"code": str})
prices = k.download_prices(sorted(tr.code.unique()), period="11y")
px = {}
for c, df in prices.items():
    df = sb.clean(df)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    px[c] = df
days = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in px.values()])))
days = days[days >= "2017-01-01"]
res, eqs = {}, []
for rule, g in tr.groupby("rule"):
    real = pd.Series(0.0, index=days)
    unreal = pd.Series(0.0, index=days)
    neg = pd.Series(0.0, index=days)                      # 含み損の銘柄だけの合計
    nopen = pd.Series(0, index=days)
    worst_one = (0.0, None)
    for r in g.itertuples():
        df = px.get(r.code)
        if df is None:
            continue
        e, x = pd.Timestamp(r.entry), pd.Timestamp(r.exit)
        if e not in df.index:
            continue
        ep = df.loc[e, "Open"]
        cl = df.loc[(df.index >= e) & (df.index < x), "Close"]
        u = r.amt * (cl / ep - 1)
        unreal = unreal.add(u.reindex(days).fillna(0), fill_value=0)
        neg = neg.add(u.clip(upper=0).reindex(days).fillna(0), fill_value=0)
        nopen = nopen.add(pd.Series(1, index=u.index).reindex(days).fillna(0).astype(int), fill_value=0)
        if len(u) and u.min() < worst_one[0]:
            worst_one = (float(u.min()), f"{r.code} {r.entry}（{u.min() / r.amt * 100:.1f}%）")
        real[real.index >= x] += r.amt * r.ret - r.amt * COST
    eq = real + unreal
    dd = eq - eq.cummax()
    i = dd.idxmin()
    res[rule] = {"確定のみの最大落ち込み": round((real - real.cummax()).min() / 1e4), "含み損込みの最大落ち込み": round(dd.min() / 1e4),
                 "落ち込みの底": str(i.date()), "その山の日": str(eq[:i].idxmax().date()), "含み損益の合計の最低": round(unreal.min() / 1e4),
                 "含み損の銘柄だけの合計の最大": round(neg.min() / 1e4), "その日": str(neg.idxmin().date()),
                 "その日の保有数": int(nopen[neg.idxmin()]), "その日の含み損の銘柄数": None, "1銘柄の最大含み損": round(worst_one[0] / 1e4, 1), "銘柄": worst_one[1],
                 "最終": round(real.iloc[-1] / 1e4)}
    eqs.append(pd.DataFrame({"rule": rule, "date": days.strftime("%Y-%m-%d"), "real": real.values.round(), "unreal": unreal.values.round()}))
    print(rule, res[rule])
json.dump(res, open("backtest/mtm_result.json", "w"), ensure_ascii=False, indent=1)
pd.concat(eqs).to_csv("backtest/mtm_equity.csv.gz", index=False, compression="gzip")
