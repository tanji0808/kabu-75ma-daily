"""指定した銘柄を、今のルール（75日線・新高値・ボリンジャー・割安さ・ランク）の目で見る。CHECK_CODES=3231:910,1802:3105（コード:買値）"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import forward_test2 as f  # noqa: E402
pairs = [x.split(":") for x in os.environ.get("CHECK_CODES", "").split(",") if x]
master = k.load_prime_list()
prices = k.download_prices(master["コード"].tolist(), period="2y")
fund = json.load(open("data/fundamentals.json", encoding="utf-8")) if os.path.exists("data/fundamentals.json") else {}
ranks = k.fund_ranks(master, prices, fund)
fins = f.fins_recent()
val = f.valuation(master, prices, fins)
tpx = f.topix()
sr_map, _ = f.sector_short_ratio()
jm = pd.read_csv("data/jq/master.csv", dtype=str); sec33 = {c[:4]: int(x) for c, x in zip(jm.Code, jm.S33) if str(x).isdigit()}
mg, mgd = f.margin_latest()
rs = f.rs_table(prices)
nb = f.nb_count(prices)
names = dict(zip(master["コード"], master["銘柄名"]))
for code, cost in pairs:
    df = prices.get(code)
    if df is None:
        print(code, "株価なし"); continue
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize() if df.index.tz is not None else pd.to_datetime(df.index)
    c, h, l = df.Close.to_numpy(float), df.High.to_numpy(float), df.Low.to_numpy(float)
    L = len(c) - 1; s = pd.Series(c)
    m25, m75, m200 = (s.rolling(n).mean().iloc[-1] for n in (25, 75, 200)); sd = s.rolling(25).std(ddof=0).iloc[-1]
    hi = h[-250:].max(); hid = df.index[-250:][int(np.argmax(h[-250:]))].date(); lo = l[-250:].min()
    d_ = s.diff(); up = d_.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).iloc[-1]
    m75s = s.rolling(75).mean()
    v = val.get(code, {})
    cost = float(cost)
    print(f"===== {code} {names.get(code)}  買値 {cost:,.0f}円")
    print(f"終値 {c[-1]:,.1f}円（{df.index[-1].date()}）  損益 {(c[-1] / cost - 1) * 100:+.1f}%")
    print(f"25日線 {m25:,.0f} ／ 75日線 {m75:,.0f}（20日前 {m75s.iloc[-21]:,.0f}）／ 200日線 {m200:,.0f}")
    print(f"ボリンジャー（25日）: +2σ {m25 + 2 * sd:,.0f} ／ +1σ {m25 + sd:,.0f} ／ −1σ {m25 - sd:,.0f} ／ −2σ {m25 - 2 * sd:,.0f}")
    print(f"52週高値 {hi:,.0f}（{hid}）・高値から {(c[-1] / hi - 1) * 100:+.1f}% ／ 52週安値 {lo:,.0f}・安値から {(c[-1] / lo - 1) * 100:+.1f}%")
    print(f"RSI {rsi:.0f}・5日 {(c[-1] / c[-6] - 1) * 100:+.1f}%・20日 {(c[-1] / c[-21] - 1) * 100:+.1f}%・60日 {(c[-1] / c[-61] - 1) * 100:+.1f}%")
    tl = tpx.reindex(df.index).ffill().to_numpy()
    print(f"TOPIXとの差（20日）{((c[-1] / c[-21]) - (tl[-1] / tl[-21])) * 100:+.1f}%・相対的株価強度 {rs.get(code, float('nan')):.0f}")
    print(f"ランク {ranks.get(code)}・PER業種比 {v.get('per_rel')}・PBR {v.get('pbr')}・配当利回り {v.get('dy')}%・ROE {v.get('roe')}・割高点数 {v.get('fs')}")
    print(f"業種の空売り比率 {sr_map.get(sec33.get(code))}・買い残 {mg.get(code)}")
    print(f"市場全体の新高値ブレイク件数 {nb}")
