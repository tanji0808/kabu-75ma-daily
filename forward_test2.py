"""
フォワードテスト2（買い＋空売り・点数で金額を変える）と「今日の売買（買い＋空売り）」ページ

採用ルール（10年のバックテスト・その時点の決算・上場廃止銘柄を含めた検証で選んだもの）
  買い   : 75日線ブレイク（本日突破）＋5日+3%以下＋RSI60以下＋ファンダメンタルランク6以上＋ROE4%以上。翌営業日の始値で買う。
           金額 ＝ ランク（6:50万・7:100万・8:150万・9〜10:200万）×点数の倍率（0点×1・1点×1.5・2点以上×2）
           点数：20日でTOPIXより8%以上強い／半年高値から45%以上下落／決算から10日以内／TOPIXが20日で3%以上下落／
                 直近の決算の増益率が−30%以下
           売り方：引き上げ式ロスカット（forward_test.py と同じ）。同時10銘柄・1,000万円まで（同じ日は増益率の高い順）
  空売り : 終値で+1σを上から割った日（その前の10日のうち5日以上は+1σより上）・出来高20万株以上・
           10日でTOPIXより5%以上弱い・割高点数1以上・買い残÷発行済株式1.0%以上・買い残の4週間の増加2割未満・
           直近90日に業績予想の上方修正なし。翌営業日の始値で空売り。
           割高点数：PERが業種の中央値の1.39倍以上で+1・2.57倍以上でさらに+1／PBR2.08倍以上で+1／配当利回り0.91%未満で+1／
                     PERが業種の0.87倍未満（黒字）で−1／配当利回り1.93%以上で−1
           金額 ＝ 点数で 0点50万・1点100万・2点150万・3点以上200万
           点数：10日でTOPIXより8%以上弱い／下方修正あり／25日線より3%以上上で割った／買い残2.5%以上／
                 直近1週間に「信用取引の注意・日々公表」に載った（4点以上も200万）
           貸借銘柄だけ（制度信用で空売りできる銘柄。信用銘柄は除く）
           見送り：海外投資家の直近4週の差引（東証・名証）が約3,240億円以上の売り越しの時は空売りしない
           買い戻し：25日線逆指値（終値が−1σを割ったら、25日線に買い戻しの逆指値）＋終値が−2σ割れで利確／+1σ超えで損切り。
           同時10銘柄・建玉1,000万円まで（同じ日は点数の高い順）。
  新高値ブレイク（2026-11-23 採用）: 終値が52週高値を初めて上に抜けた日（その高値は5営業日以上前・20日平均の売買代金3,000万円以上）。
           100点満点の点数60点以上：PBR1.07倍未満20／ランク6以上20／TOPIXも52週高値圏15／配当利回り2.3%以上10／
           PERが業種の0.95倍未満10／業種の空売り比率40.6%未満10／決算から10日以内5／買い残が発行済の0.36%以上5／52週安値から60%以上上5。
           さらに：業績予想の下方修正なし・相対的株価強度50以上・空売り残高報告の合計4%未満・信用取引の注意に載っていない。
           翌営業日の始値で買い、−6%で損切り・+25%で利確・120営業日で手じまい。金額は全銘柄同じ（新高値の枠÷12）。同じ日は点数の高い順。
           市場全体の新高値ブレイクの件数（買った日が30〜6日前）が102件以下なら金額を半分、430件超なら見送り。
  資金の分け方 : 現物は 75日線の買い4割・新高値6割。空売りは信用の建玉で別枠。
信用残・決算データは J-Quants（data/jq/ と API）。信用残が取れない日は空売りの判定ができないので「判定できず」と記録する。
台帳 data/forward2.json には合図だけを保存し、損益は毎回株価から計算し直す。
"""

from __future__ import annotations

import bisect
import html as h
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

LEDGER = os.environ.get("FORWARD2_PATH", "data/forward2.json")
RULE_ID = "combo_v4"   # v2: 空売りはランク5以下→割高点数1以上、買いはROE4%未満を除く。v3: 空売りに海外投資家の見送り・信用取引の注意の加点・貸借銘柄だけ。v4: 新高値ブレイクを追加・現物を買い4割／新高値6割に（2026-11-23から）
LOT_BUY = {6: 0.5, 7: 1.0, 8: 1.5}
CAPITAL = float(os.environ.get("FORWARD2_CAPITAL", "10000000"))   # 最初の資金
BASE = 13_000_000      # 元のルール（買い1,000万円＋空売りの保証金300万円）の規模。資金がこの何倍かで金額を変える
CAP_PER = 3_000_000    # 1回の金額の上限
COST = 0.002           # 往復の売買コスト（値段のずれ・手数料）
W_SHORT = [0.5, 1, 1.5, 2, 2]
# 資金の分け方（2026-11-23 採用）：現物は 75日線の買い 4割・新高値ブレイク 6割。空売りは信用の建玉で別枠
ALLOC_BUY = 0.4
ALLOC_NH = 0.6
NH_MAX = 12            # 新高値の同時保有数（1回 ＝ 新高値の枠 ÷ 12）
NH_STOP, NH_TP = 0.06, 0.25
NB_HOT, NB_WEAK = 430, 102   # 市場全体の新高値ブレイクの件数（買った日が30〜6日前のもの）：430件超は過熱で見送り・102件以下は弱い相場で半分
NH_ITEMS = [("PBR1.07倍未満", 20), ("ランク6以上", 20), ("TOPIXも52週高値圏", 15), ("配当利回り2.3%以上", 10),
            ("PERが業種の0.95倍未満", 10), ("業種の空売り比率が低い", 10), ("決算から10日以内", 5),
            ("買い残が発行済の0.36%以上", 5), ("52週安値から60%以上上", 5)]


# ------------------------------------------------------------------ データ
def topix() -> pd.Series | None:
    try:
        import yfinance as yf
        b = yf.download("1306.T", period="2y", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
        return b
    except Exception as e:
        print(f"  TOPIX連動ETFを取得できませんでした（{e}）")
        return None


def margin_latest() -> tuple[dict, str | None]:
    """銘柄ごとの直近の信用残（買い残）と4週間前の買い残。J-Quants の過去データ＋APIで最新の週を足す。"""
    hist = {}
    p = "data/jq/margin_interest.csv.gz"
    if os.path.exists(p):
        m = pd.read_csv(p, dtype=str)
        m["LongVol"] = pd.to_numeric(m["LongVol"], errors="coerce")
        for c, g in m.sort_values("Date").groupby(m["Code"].str[:4]):
            hist[c] = list(zip(g["Date"], g["LongVol"]))
    key = os.environ.get("JQUANTS_API_KEY", "").strip()
    if key:
        try:
            import requests
            have = max((v[-1][0] for v in hist.values() if v), default="2000-01-01")
            fridays = []
            d = date.today()
            while len(fridays) < 4:                        # 直近4週の金曜日のうち、まだ持っていない週を取る
                d -= timedelta(days=1)
                if d.weekday() == 4 and d.isoformat() > have:
                    fridays.append(d)
                if d.isoformat() <= have:
                    break
            for d in sorted(fridays):
                rows, pk = [], None
                while True:
                    prm = {"date": d.strftime("%Y%m%d")}
                    if pk:
                        prm["pagination_key"] = pk
                    r = requests.get("https://api.jquants.com/v2/markets/margin-interest", headers={"x-api-key": key},
                                     params=prm, timeout=60)
                    if r.status_code != 200:
                        print(f"  信用残（{d}）を取得できませんでした（HTTP {r.status_code}）")
                        break
                    j = r.json()
                    rows += j.get("data", [])
                    pk = j.get("pagination_key")
                    if not pk:
                        break
                for x in rows:
                    hist.setdefault(x["Code"][:4], []).append((x["Date"], float(x.get("LongVol") or 0)))
                if rows:
                    print(f"  信用残（{d}）: {len(rows)}銘柄")
        except Exception as e:
            print(f"  信用残の取得に失敗（{e}）")
    jp = "data/jpx/margin.csv.gz"                          # JPXの公開PDFから毎日ためた分（J-Quantsが無くても使える）
    if os.path.exists(jp):
        m = pd.read_csv(jp, dtype=str)
        m["LongVol"] = pd.to_numeric(m["LongVol"], errors="coerce")
        for c, g in m.groupby(m["Code"].str[:4]):
            hist.setdefault(c, []).extend(zip(g["Date"], g["LongVol"]))
    latest = max((max(x[0] for x in v) for v in hist.values() if v), default=None)
    out = {}
    for c, v in hist.items():
        wk = {}
        for d_, lv in sorted(set(v)):                      # 週ごと（その週の最後の日）にそろえる。バックテストと同じ週次
            if lv == lv:
                wk[date.fromisoformat(d_[:10]).isocalendar()[:2]] = (d_, lv)
        v = sorted(wk.values())
        if not v:
            continue
        lv = v[-1][1]
        prev = v[-5][1] if len(v) >= 5 else None
        out[c] = {"date": v[-1][0], "long": lv, "lchg": (lv / prev - 1) if prev else None}
    return out, latest


def fins_recent() -> dict:
    """決算の概要（data/jq/fins_summary.csv.gz）を銘柄ごとに。"""
    p = "data/jq/fins_summary.csv.gz"
    if not os.path.exists(p):
        return {}
    import short_rel_pit as sp
    return sp.load_fins()


def valuation(master: pd.DataFrame, prices: dict, fins: dict) -> dict:
    """銘柄ごとの割高点数・ROE（その日までの決算。J-Quants がなければ data/fundamentals.json で補う）。"""
    import statistics
    import kabu_75ma_search as k
    import short_rel_pit as sp
    fnow = {}
    if os.path.exists("data/fundamentals.json"):
        fnow = json.load(open("data/fundamentals.json", encoding="utf-8"))
    sec_of = dict(zip(master["コード"], master["業種"]))
    met = {}
    for code, df in prices.items():
        if not len(df):
            continue
        recs = fins.get(code)
        f = sp.pit_fund(recs) if recs else fnow.get(code)
        if not f:
            continue
        met[code] = k._fund_metrics(f, float(df["Close"].iloc[-1]))
    med = {}
    for sec in set(sec_of.get(c) for c in met):
        v = [met[c]["per"] for c in met if sec_of.get(c) == sec and met[c]["per"] is not None and 0 < met[c]["per"] < 200]
        med[sec] = statistics.median(v) if v else None
    out = {}
    for c, m in met.items():
        mp = med.get(sec_of.get(c))
        per_rel = m["per"] / mp if (m["per"] is not None and mp) else None
        dy, pbr = m["dy"], m["pbr"]
        fs = ((per_rel is not None and per_rel >= 1.39) + (per_rel is not None and per_rel >= 2.57) + (pbr is not None and pbr >= 2.08)
              + (dy is not None and dy < 0.91) - (per_rel is not None and per_rel < 0.87 and not m["perNeg"]) - (dy is not None and dy >= 1.93))
        out[c] = {"fs": int(fs), "per_rel": per_rel, "pbr": pbr, "dy": dy, "roe": m["roe"]}
    return out


# ------------------------------------------------------------------ 今日の候補
def _jq(path: str, **params) -> tuple[int, list]:
    """J-Quants API（ページ送りあり）。キーが無い・失敗したら (0, [])。"""
    key = os.environ.get("JQUANTS_API_KEY", "").strip()
    if not key:
        return 0, []
    try:
        import requests
        rows, pk = [], None
        while True:
            p = dict(params)
            if pk:
                p["pagination_key"] = pk
            r = requests.get("https://api.jquants.com/v2" + path, headers={"x-api-key": key}, params=p, timeout=60)
            if r.status_code != 200:
                return r.status_code, rows
            j = r.json()
            rows += j.get("data", [])
            pk = j.get("pagination_key")
            if not pk:
                return 200, rows
    except Exception as e:
        print(f"  J-Quants {path} の取得に失敗（{e}）")
        return 0, []


FRGN_LIMIT = -3.24   # 海外投資家の直近4週の差引（億円単位の内部値。約3,240億円の売り越し）以下なら空売りを見送る


def foreign_flow() -> tuple[float | None, str | None]:
    """海外投資家の直近4週の差引の合計（東証・名証合計）と、その最新の公表日。"""
    p = "data/jq/investor_types.csv.gz"
    it = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["PubDate", "Section", "FrgnBal"])
    st, rows = _jq("/equities/investor-types", **{"from": (date.today() - timedelta(days=70)).strftime("%Y%m%d"),
                                                  "to": date.today().strftime("%Y%m%d")})
    if rows:
        it = pd.concat([it, pd.DataFrame(rows)], ignore_index=True)
    elif st not in (0, 200):
        print(f"  投資部門別（海外投資家）を取得できませんでした（HTTP {st}）")
    it = it[it["Section"] == "TokyoNagoya"].copy()
    jp = "data/jpx/investor.csv.gz"                       # JPXの公開Excelの分（同じ千円単位）
    if os.path.exists(jp):
        x = pd.read_csv(jp, dtype=str)
        x = x[~x["EnDate"].isin(set(it.get("EnDate", [])))]
        it = pd.concat([it, x.assign(Section="TokyoNagoya", PubDate=x["EnDate"])], ignore_index=True)
    if it.empty:
        return None, None
    it["FrgnBal"] = pd.to_numeric(it["FrgnBal"], errors="coerce")
    it = it.drop_duplicates(["EnDate"] if "EnDate" in it else ["PubDate"], keep="first").sort_values("EnDate" if "EnDate" in it else "PubDate")
    last = it.tail(4)
    if len(last) < 4:
        return None, None
    return float(last["FrgnBal"].sum() / 1e8), str(last["EnDate"].iloc[-1] if "EnDate" in last else last["PubDate"].iloc[-1])


def margin_alert_recent() -> set:
    """直近1週間に「信用取引の注意・日々公表」（J-Quants margin-alert）に載った銘柄。"""
    out = set()
    d = date.today()
    for i in range(8):
        x = d - timedelta(days=i)
        if x.weekday() >= 5:
            continue
        st, rows = _jq("/markets/margin-alert", date=x.strftime("%Y%m%d"))
        out |= {str(r.get("Code", ""))[:4] for r in rows}
    p = "data/jpx/alert.csv.gz"                          # JPXの公開Excel（申込日）の分
    if os.path.exists(p):
        a = pd.read_csv(p, dtype=str).fillna("")
        a = a[(a["Date"] >= (d - timedelta(days=8)).isoformat()) & (a["Code"] != "")]
        out |= {c[:4] for c in a["Code"]}
    out.discard("")
    return out


def loan_kind() -> dict:
    """銘柄ごとの区分（貸=貸借銘柄・制=制度信用銘柄・他）。JPXの毎日のPDF → なければ J-Quants の銘柄一覧。"""
    out = {}
    p = "data/jq/master.csv"
    if os.path.exists(p):
        m = pd.read_csv(p, dtype=str)
        out = {c[:4]: {"貸借": "貸", "信用": "制"}.get(n, "他") for c, n in zip(m["Code"], m["MrgnNm"])}
    p = "data/jpx/margin.csv.gz"
    if os.path.exists(p):
        m = pd.read_csv(p, dtype=str)
        m = m[(m["Date"] == m["Date"].max()) & m["Kind"].isin(["貸", "制", "他"])]
        out.update({c[:4]: k for c, k in zip(m["Code"], m["Kind"])})
    return out


def sector_short_ratio() -> tuple[dict, str | None]:
    """業種（33業種コード）ごとの空売り比率（直近5日の平均、％）。J-Quants の過去データ＋APIで足りない日を取る。"""
    p = "data/jq/short_ratio.csv.gz"
    s = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["Date", "S33", "SellExShortVa", "ShrtWithResVa", "ShrtNoResVa"])
    have = set(s["Date"].astype(str))
    new = []
    for i in range(0, 12):
        d = date.today() - timedelta(days=i)
        if d.weekday() >= 5 or d.isoformat() in have:
            continue
        st, rows = _jq("/markets/short-ratio", date=d.strftime("%Y%m%d"))
        new += rows
    if new:
        s = pd.concat([s, pd.DataFrame(new)], ignore_index=True)
    if s.empty:
        return {}, None
    for k in ("SellExShortVa", "ShrtWithResVa", "ShrtNoResVa"):
        s[k] = pd.to_numeric(s[k], errors="coerce")
    s["S33"] = pd.to_numeric(s["S33"], errors="coerce")
    s["ratio"] = (s.ShrtWithResVa + s.ShrtNoResVa) / (s.SellExShortVa + s.ShrtWithResVa + s.ShrtNoResVa) * 100
    p_ = s.pivot_table(index="Date", columns="S33", values="ratio").sort_index()
    last = p_.tail(5).mean()
    return {int(k): float(v) for k, v in last.items() if v == v}, str(p_.index[-1])


def short_report_sum() -> tuple[dict, str | None]:
    """銘柄ごとの空売り残高報告（0.5%以上の機関の合計、％）。直近1年以内の、機関ごとの最新の報告を合計する。"""
    p = "data/jq/short_sale_report.csv.gz"
    if not os.path.exists(p):
        return {}, None
    r = pd.read_csv(p, dtype={"Code": str}, usecols=["DiscDate", "Code", "SSName", "FundName", "ShrtPosToSO"])
    last = r["DiscDate"].max()
    new = []
    d = date.fromisoformat(last) + timedelta(days=1)
    while d <= date.today():
        if d.weekday() < 5:
            st, rows = _jq("/markets/short-sale-report", disc_date=d.strftime("%Y%m%d"))
            new += [{k: x.get(k) for k in ("DiscDate", "Code", "SSName", "FundName", "ShrtPosToSO")} for x in rows]
        d += timedelta(days=1)
    if new:
        r = pd.concat([r, pd.DataFrame(new)], ignore_index=True)
    r["ShrtPosToSO"] = pd.to_numeric(r["ShrtPosToSO"], errors="coerce")
    lo = (date.today() - timedelta(days=365)).isoformat()
    r = r[r["DiscDate"] >= lo].sort_values("DiscDate")
    r["h"] = r["SSName"].astype(str) + "|" + r["FundName"].astype(str)
    r = r.drop_duplicates(["Code", "h"], keep="last")
    r = r[r["ShrtPosToSO"] >= 0.005]
    return (r.groupby(r["Code"].astype(str).str[:4])["ShrtPosToSO"].sum() * 100).to_dict(), str(r["DiscDate"].max())


def nh_breakout(c, hi, v, L) -> tuple[bool, float, int]:
    """新高値ブレイクの合図：終値が52週高値（前日までの250営業日）を初めて上に抜け、その高値は5営業日以上前、20日平均の売買代金3,000万円以上。"""
    piv = hi[L - 250:L].max()
    base_len = L - (L - 250 + int(np.argmax(hi[L - 250:L])))
    ok = bool(c[L] > piv and c[L - 1] <= hi[L - 251:L - 1].max() and np.mean(c[L - 19:L + 1] * v[L - 19:L + 1]) >= 3e7 and base_len >= 5)
    return ok, float(piv), base_len


def nb_count(prices: dict, today_=None) -> int:
    """市場の幅：新高値ブレイクの件数。バックテストと同じく、買った日（合図の翌営業日）が today_ の30日前〜6日前のもの。"""
    if today_ is None:
        today_ = max(pd.Timestamp(df.index[-1]) for df in prices.values())
    today_ = pd.Timestamp(today_)
    today_ = (today_.tz_localize(None) if today_.tzinfo else today_).normalize()
    n = 0
    for code, df in prices.items():
        if len(df) < 280:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        cc, hh, vv = (df[x].to_numpy(float) for x in ("Close", "High", "Volume"))
        end = int(idx.searchsorted(today_, side="right")) - 1
        for j in range(max(260, end - 26), end):
            ent = idx[j + 1]
            if today_ - pd.Timedelta(days=30) <= ent <= today_ - pd.Timedelta(days=6) and nh_breakout(cc, hh, vv, j)[0]:
                n += 1
    return n


def rs_table(prices: dict) -> dict:
    """相対的株価強度（3か月×2＋6・9・12か月の騰落率）の全銘柄の中での順位（0〜100）。"""
    raw = {}
    for code, df in prices.items():
        c = df["Close"].to_numpy(float)
        if len(c) < 253 or not c[-1] > 0:
            continue
        g = lambda n: c[-1] / c[-1 - n] - 1 if c[-1 - n] > 0 else np.nan
        raw[code] = 0.4 * g(63) + 0.2 * g(126) + 0.2 * g(189) + 0.2 * g(252)
    sr = pd.Series(raw).dropna()
    return (sr.rank(pct=True) * 100).to_dict()


def candidates(master: pd.DataFrame, prices: dict, ranks: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    import backtest as bt
    import short_candidates as sc
    import short_rel_pit as sp
    tpx = topix()
    mg, mg_date = margin_latest()
    fins = fins_recent()
    names = dict(zip(master["コード"], master["銘柄名"]))
    val = valuation(master, prices, fins)
    kind = loan_kind()
    frgn4, frgn_date = foreign_flow()
    alert = margin_alert_recent()
    frgn_stale = frgn_date is None or (date.today() - date.fromisoformat(frgn_date[:10])).days > 17
    info = {"frgn4": frgn4, "frgn_date": frgn_date, "frgn_ok": not frgn_stale, "alert_n": len(alert),
            "short_paused": (not frgn_stale) and frgn4 <= FRGN_LIMIT,
            "margin_date": mg_date, "fins_last": max((v[-1]["DiscDate"] for v in fins.values() if v), default=None)}
    sr_map, sr_date = sector_short_ratio()
    ss_map, ss_date = short_report_sum()
    rs = rs_table(prices)
    sec33 = {}
    if os.path.exists("data/jq/master.csv"):
        jm = pd.read_csv("data/jq/master.csv", dtype=str)
        sec33 = {c_[:4]: int(x) for c_, x in zip(jm["Code"], jm["S33"]) if str(x).isdigit()}
    info.update(sr_date=sr_date, ss_date=ss_date)
    # 市場の幅：直近の新高値ブレイクの件数（バックテストと同じ数え方：買った日＝合図の翌営業日が、今日の30日前〜6日前）
    nb30 = nb_count(prices)
    nb_state = "過熱（見送り）" if nb30 > NB_HOT else ("弱い（半分）" if nb30 <= NB_WEAK else "普通")
    info.update(nb30=nb30, nb_state=nb_state)
    nb_mult = 0.0 if nb30 > NB_HOT else (0.5 if nb30 <= NB_WEAK else 1.0)
    buys, shorts, nhs = [], [], []
    for code, df in prices.items():
        if len(df) < 260:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        L = len(df) - 1
        sd_ = df.index[L].strftime("%Y-%m-%d")
        c, hi, lo, v = (df[x].to_numpy(float) for x in ("Close", "High", "Low", "Volume"))
        t = tpx.reindex(df.index).ffill().to_numpy() if tpx is not None else None
        recs = [r for r in fins.get(code, []) if r["DiscDate"] <= sd_]
        f = sp.pit_fund(recs) if recs else {}
        rk = ranks.get(code)
        # ---- 買い
        sig = bt.signal_days(df, "calm5+rsi60")
        roe = (val.get(code) or {}).get("roe")
        if len(sig) and sig[-1] == L and rk is not None and rk >= 6 and not (roe is not None and roe < 4):
            stm = [r for r in recs if "FinancialStatements" in str(r["DocType"])]
            dsince = (df.index[L] - pd.Timestamp(stm[-1]["DiscDate"])).days if stm else None
            drop = (1 - lo[L - 120:L + 1].min() / hi[L - 120:L + 1].max()) * 100
            rel20 = (c[L] / c[L - 20] - 1 - (t[L] / t[L - 20] - 1)) * 100 if t is not None else None
            tpx20 = (t[L] / t[L - 20] - 1) * 100 if t is not None else None
            eg = f.get("eg")
            pts = [("20日でTOPIXより8%以上強い", rel20 is not None and rel20 >= 8), ("半年高値から45%以上下落", drop >= 45),
                   ("決算から10日以内", dsince is not None and dsince <= 10), ("TOPIXが20日で3%以上下落", tpx20 is not None and tpx20 <= -3),
                   ("直近の増益率が−30%以下", eg is not None and eg <= -0.3)]
            score = sum(ok for _, ok in pts)
            lot = LOT_BUY.get(int(rk), 2.0) * [1, 1.5, 2][min(score, 2)]
            buys.append({"date": sd_, "side": "buy", "code": code, "name": names.get(code, code), "rank": int(rk),
                         "score": score, "lot": lot, "reasons": [n for n, ok in pts if ok], "eg": eg, "close": round(float(c[L]), 1)})
        # ---- 新高値ブレイク
        brk, piv, base_len = nh_breakout(c, hi, v, L)
        if brk:
            vv = val.get(code) or {}
            rev = sc.rev_flag(recs, sd_) if recs else 0
            rsv = rs.get(code)
            ssv = ss_map.get(code, 0.0)
            if rev != -1 and rsv is not None and rsv >= 50 and ssv < 4 and code not in alert:
                stm = [r for r in recs if "FinancialStatements" in str(r["DocType"])]
                dsince = (df.index[L] - pd.Timestamp(stm[-1]["DiscDate"])).days if stm else None
                sh = None
                for r in reversed(recs):
                    if r["ShOutFY"] == r["ShOutFY"] and r["ShOutFY"]:
                        sh = r["ShOutFY"] - (r["TrShFY"] if r["TrShFY"] == r["TrShFY"] else 0)
                        break
                g = mg.get(code)
                lsh = g["long"] / sh * 100 if g and sh else None
                tpx_new = t is not None and t[L] >= np.nanmax(t[L - 249:L + 1]) * 0.97
                srv = sr_map.get(sec33.get(code))
                above_low = (c[L] / lo[L - 249:L + 1].min() - 1) * 100
                ok = [vv.get("pbr") is not None and vv["pbr"] < 1.07, rk is not None and rk >= 6, bool(tpx_new),
                      vv.get("dy") is not None and vv["dy"] >= 2.31, vv.get("per_rel") is not None and vv["per_rel"] < 0.95,
                      srv is not None and srv < 40.61, dsince is not None and dsince <= 9, lsh is not None and lsh >= 0.36,
                      above_low >= 60.19]
                score = sum(p_ for (n_, p_), o_ in zip(NH_ITEMS, ok) if o_)
                if score >= 60:
                    nhs.append({"date": sd_, "side": "newhigh", "code": code, "name": names.get(code, code), "rank": rk,
                                "score": score, "lot": 1.0, "mult": nb_mult, "rs": round(rsv, 1), "reasons": [n_ for (n_, p_), o_ in zip(NH_ITEMS, ok) if o_],
                                "close": round(float(c[L]), 1), "pivot": round(float(piv), 1)})
        # ---- 空売り
        s = pd.Series(c)
        m25 = s.rolling(25).mean()
        sdv = s.rolling(25).std(ddof=0)
        u1 = (m25 + sdv).to_numpy()
        if not (c[L - 1] > u1[L - 1] and c[L] < u1[L]) or v[L] < 200_000:
            continue
        walk = int(sum(c[j] > u1[j] for j in range(L - 10, L)))
        if walk < 5 or t is None:
            continue
        rel10 = (c[L] / c[L - 10] - 1 - (t[L] / t[L - 10] - 1)) * 100
        vv = val.get(code)
        if rel10 > -5 or vv is None or vv["fs"] < 1:
            continue
        rev = sc.rev_flag(recs, sd_) if recs else 0
        if rev == 1:
            continue
        sh = None
        for r in reversed(recs):
            if r["ShOutFY"] == r["ShOutFY"] and r["ShOutFY"]:
                sh = r["ShOutFY"] - (r["TrShFY"] if r["TrShFY"] == r["TrShFY"] else 0)
                break
        g = mg.get(code)
        lsh = g["long"] / sh * 100 if g and sh else None
        lchg = g["lchg"] if g else None
        fresh = g is not None and mg_date is not None and (date.fromisoformat(mg_date[:10]) - date.fromisoformat(g["date"][:10])).days <= 10
        ok_margin = lsh is not None and lchg is not None and fresh            # 信用残が古い銘柄は判定しない
        if ok_margin and not (lsh >= 1.0 and lchg < 0.2):
            continue
        if kind.get(code) != "貸":
            continue                                       # 貸借銘柄でないと制度信用で空売りできないので除く
        dev25 = (c[L] / m25.iloc[L] - 1) * 100
        pts = [("10日でTOPIXより8%以上弱い", rel10 <= -8), ("業績予想の下方修正あり", rev == -1),
               ("25日線より3%以上上で割った", dev25 >= 3), ("買い残2.5%以上", lsh is not None and lsh >= 2.5)]
        pts.append(("信用取引の注意・日々公表に載った", code in alert))
        score = sum(ok for _, ok in pts)
        shorts.append({"date": sd_, "side": "short", "code": code, "name": names.get(code, code), "rank": rk, "fs": vv["fs"],
                       "score": score, "lot": W_SHORT[min(score, 4)], "reasons": [n for n, ok in pts if ok], "close": round(float(c[L]), 1),
                       "lsh": None if lsh is None else round(lsh, 2), "lchg": None if lchg is None else round(lchg, 3),
                       "margin_ok": ok_margin and info["frgn_ok"], "paused": info["short_paused"]})
    return buys, shorts, nhs, info


# ------------------------------------------------------------------ 台帳と成績
def load_ledger() -> dict:
    if os.path.exists(LEDGER):
        led = json.load(open(LEDGER, encoding="utf-8"))
        if led.get("rule_id") == RULE_ID:
            return led
    return {"start": datetime.now().strftime("%Y-%m-%d"), "rule_id": RULE_ID, "signals": []}


def evaluate(ledger: dict, prices: dict) -> list[dict]:
    import buy_candidates as bc
    import short_bw2_backtest as b2
    out = []
    for s in ledger["signals"]:
        df = prices.get(s["code"])
        r = dict(s)
        if df is None:
            r["status"] = "株価なし"
            out.append(r)
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        i = bisect.bisect_left(dates, s["date"])
        if i >= len(dates) or dates[i] != s["date"]:
            r["status"] = "株価なし"
            out.append(r)
            continue
        if i + 1 >= len(dates):
            r["status"] = "明日の寄りで" + ("空売り" if s["side"] == "short" else "買い")
            out.append(r)
            continue
        if s["side"] == "newhigh":
            import newhigh_candidates as nh
            a = {"o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float), "l": df["Low"].to_numpy(float),
                 "c": df["Close"].to_numpy(float), "pivot": np.full(len(df), np.nan), "dates": dates}
            x = nh.sim(a, i, NH_STOP, tp=NH_TP)
            entry = a["o"][i + 1]
            r.update(entry=round(entry, 1), entry_date=dates[i + 1], stop_px=round(entry * (1 - NH_STOP), 1),
                     tp_px=round(entry * (1 + NH_TP), 1))
            if x is None:
                last = a["c"][-1]
                r.update(status="保有中", now=round(last, 1), ret=round((last / entry - 1) * 100, 2))
            else:
                r.update(status="決済済み", ret=float(x["ret"]), exit_date=x["exit"],
                         why={"stop": "損切り（−6%）", "tp": "利確（+25%）", "time": "120営業日で手じまい"}.get(x["reason"], x["reason"]))
            amt = s.get("amount") or 0
            r["pnl"] = round(amt * r["ret"] / 100 - amt * COST)
        elif s["side"] == "buy":
            cs = pd.Series(df["Close"].to_numpy(float))
            m25, sd = cs.rolling(25).mean(), cs.rolling(25).std(ddof=0)
            a = {"o": df["Open"].to_numpy(float), "c": df["Close"].to_numpy(float), "l": df["Low"].to_numpy(float),
                 "d1": (m25 - sd).to_numpy(), "u1": (m25 + sd).to_numpy(), "m25": m25.to_numpy(), "dates": dates}
            x = bc.sim(a, i, None, False, "u1")
            entry = a["o"][i + 1]
            if x is None:     # まだ持っている
                last = a["c"][-1]
                r.update(status="保有中", entry=round(entry, 1), now=round(last, 1), ret=round((last / entry - 1) * 100, 2),
                         entry_date=dates[i + 1])
            else:
                r.update(status="決済済み", entry=round(entry, 1), ret=x["ret"], exit_date=x["exit"], entry_date=x["entry"],
                         why=x["reason"])
            amt = s.get("amount") or s["lot"] * 1e6
            r["pnl"] = round(amt * r["ret"] / 100 - amt * COST)
        else:
            x = b2.simulate(s["code"], df, i, "lk25c", {})
            closed = not x["why"].startswith("保有中")
            r.update(status="決済済み" if closed else "保有中", entry=x["entry"], ret=x["ret_pct"], entry_date=x["entry_date"],
                     exit_date=x["exit_date"] if closed else None, why=x["why"])
            amt = s.get("amount") or s["lot"] * 1e6
            r["pnl"] = round(amt * x["pnl"] / max(x["notional"], 1) - amt * COST)
        out.append(r)
    # 同時10銘柄・1,000万円まで（買いは増益率の高い順、空売りは点数の高い順）
    for side in ("buy", "short", "newhigh"):
        mx = NH_MAX if side == "newhigh" else 10
        rows = [r for r in out if r["side"] == side and r.get("entry_date")]
        rows.sort(key=lambda r: (r["entry_date"], -(r.get("eg") or -9) if side == "buy" else -r["score"]))
        open_ = []
        for r in rows:
            open_ = [x for x in open_ if (x.get("exit_date") or "9999") > r["entry_date"]]
            amt = lambda x: x.get("amount") or x["lot"] * 1e6
            if len(open_) >= mx or sum(amt(x) for x in open_) + amt(r) > r.get("cap", 1e7) * 1.0001 or any(x["code"] == r["code"] for x in open_):
                r["status"] = "見送り（枠がいっぱい）"
                continue
            open_.append(r)
    return out


def stats(rows: list[dict], side: str) -> dict:
    d = [r for r in rows if r["side"] == side and r["status"] in ("保有中", "決済済み")]
    cl = [r for r in d if r["status"] == "決済済み"]
    return {"n": len(d), "closed": len(cl), "win": round(sum(r["pnl"] > 0 for r in cl) / len(cl) * 100, 1) if cl else None,
            "pnl_closed": sum(r["pnl"] for r in cl), "pnl_open": sum(r["pnl"] for r in d if r["status"] == "保有中")}


# ------------------------------------------------------------------ ページ
def build_page(ledger: dict, rows: list[dict], today_buy: list, today_short: list, info: dict, data_date: str, path: str,
               today_nh: list | None = None) -> None:
    today_nh = today_nh or []
    def yen(v):
        return f"{v / 1e4:+,.1f}万円"

    def lot(v):
        return f"{v * 100:.0f}万円"

    def table(items, cols):
        if not items:
            return '<p class="none">なし</p>'
        head = "".join(f"<th>{h.escape(c)}</th>" for c, _ in cols)
        body = "".join("<tr>" + "".join(f"<td>{f(x)}</td>" for _, f in cols) + "</tr>" for x in items)
        return f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>'

    link = lambda x: f'<a href="https://finance.yahoo.co.jp/quote/{x["code"]}.T/chart" target="_blank" rel="noopener">{x["code"]} {h.escape(x["name"])}</a>'
    man = lambda v: f"{v / 1e4:,.1f}万円"
    today_cols = [("銘柄", link), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'), ("購入金額", lambda x: man(x.get("amount", 0))),
                  ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
                  ("ランク", lambda x: "—" if x.get("rank") is None else x["rank"]), ("点数", lambda x: x["score"]),
                  ("点数の理由", lambda x: h.escape("・".join(x["reasons"]) or "—"))]
    sh_cols = today_cols + [("割高点数", lambda x: x.get("fs", "—")),
                            ("買い残", lambda x: "—" if x.get("lsh") is None else f'{x["lsh"]}%（4週{x["lchg"] * 100:+.0f}%）')]
    nh_cols = [("銘柄", link), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'), ("購入金額", lambda x: man(x.get("amount", 0))),
               ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
               ("点数", lambda x: f'{x["score"]}点'), ("損切り目安", lambda x: f'{x["close"] * (1 - NH_STOP):,.0f}円'),
               ("利確目安", lambda x: f'{x["close"] * (1 + NH_TP):,.0f}円'), ("点数の理由", lambda x: h.escape("・".join(x["reasons"]) or "—"))]
    hold = [r for r in rows if r["status"] == "保有中"]
    closed = sorted([r for r in rows if r["status"] == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    SIDE = {"buy": "買い（75日線）", "newhigh": "買い（新高値）", "short": "空売り"}
    pos_cols = [("売買", lambda x: SIDE.get(x["side"], x["side"])), ("銘柄", link), ("合図", lambda x: x["date"]),
                ("金額", lambda x: man(x["amount"]) if x.get("amount") else lot(x["lot"])), ("株数", lambda x: f'{x["shares"]:,}株' if x.get("shares") else "—"),
                ("建値", lambda x: x.get("entry")), ("損益", lambda x: f'{x["ret"]:+.2f}%（{yen(x["pnl"])}）')]
    hold_cols = pos_cols + [("損切り／利確（新高値）", lambda x: f'{x["stop_px"]:,.0f}円／{x["tp_px"]:,.0f}円' if x.get("stop_px") else "—")]
    cl_cols = pos_cols + [("決済", lambda x: x.get("exit_date") or ""), ("理由", lambda x: h.escape(x.get("why") or ""))]
    sb_, ss_, sn_ = stats(rows, "buy"), stats(rows, "short"), stats(rows, "newhigh")
    warn = ""
    if not info.get("margin_date") or (datetime.now() - pd.Timestamp(info["margin_date"])).days > 14:
        warn = ('<p class="warn">信用残のデータが古いか、取れていません（最新：' + h.escape(str(info.get("margin_date"))) +
                '）。空売りの「買い残」の条件が判定できないため、空売りの候補は参考です。</p>')
    f4 = info.get("frgn4")
    if not info.get("frgn_ok"):
        warn += ('<p class="warn">海外投資家の売買（週ごと）のデータが古いか、取れていません（最新：' + h.escape(str(info.get("frgn_date"))) +
                 '）。判定できないため、空売りは記録しません。</p>')
    elif info.get("short_paused"):
        warn += (f'<p class="warn">海外投資家が直近4週で大きく売り越し中（約{-f4 * 1000:,.0f}億円・{h.escape(str(info.get("frgn_date")))}までの週）。'
                 '下げ余地が小さい時期なので、空売りは見送ります。</p>')
    frgn_txt = "—" if f4 is None else (f"{f4 * 1000:+,.0f}億円（{'見送り中' if info.get('short_paused') else '空売りOK'}）")
    page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>今日の売買（買い＋新高値＋空売り）</title><style>
body{{font-family:'Noto Sans JP',system-ui,sans-serif;margin:0;background:#f5f4f0;color:#1d1d1b;font-size:14px;line-height:1.55}}
main{{max-width:1100px;margin:0 auto;padding:16px}}h1{{font-size:18px;margin:4px 0 2px}}h2{{font-size:15px;margin:22px 0 8px}}
.sub{{color:#6b6a65;font-size:12px}}.card{{background:#fff;border:1px solid #e3e1db;border-radius:10px;padding:12px 14px;margin-top:10px}}
.tw{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px 8px;border-bottom:1px solid #efede7;text-align:left;white-space:nowrap}}
th{{color:#6b6a65;font-weight:500;font-size:12px}}.none{{color:#6b6a65}}.warn{{background:#fff8ec;border:1px solid #f0d9b0;color:#b26b00;padding:8px 10px;border-radius:6px;font-size:12.5px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}}.stat{{background:#faf9f6;border:1px solid #efede7;border-radius:8px;padding:8px 10px}}
.stat .k{{font-size:11px;color:#6b6a65}}.stat .v{{font-size:16px;font-weight:700}}a{{color:#1f4e79}}</style></head><body><main>
<a href="./" class="sub">← スクリーナーに戻る</a>
<h1>今日の売買（75日線の買い＋新高値ブレイク＋空売り）</h1>
<div class="card"><div class="stats">
<div class="stat"><div class="k">今の資金（最初 {CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{info.get('equity', CAPITAL) / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">規模（元のルールの1,300万円に対して）</div><div class="v">{info.get('equity', CAPITAL) / BASE * 100:.0f}%</div></div>
<div class="stat"><div class="k">1回の上限</div><div class="v">{CAP_PER / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">75日線の買いの枠（資金の{ALLOC_BUY * 100:.0f}%）</div><div class="v">10銘柄・{info.get('equity', CAPITAL) * ALLOC_BUY / 1e4:,.0f}万円まで</div></div>
<div class="stat"><div class="k">新高値の枠（資金の{ALLOC_NH * 100:.0f}%）</div><div class="v">{NH_MAX}銘柄・1回{info.get('equity', CAPITAL) * ALLOC_NH / NH_MAX / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">空売りの枠（信用の建玉）</div><div class="v">10銘柄・{info.get('equity', CAPITAL) / BASE * 1000:,.0f}万円まで</div></div></div>
<div class="sub" style="margin-top:6px">75日線の買い ＝ 点数で決まる金額 × 資金の{ALLOC_BUY * 100:.0f}%の規模。新高値 ＝ 資金の{ALLOC_NH * 100:.0f}%を{NH_MAX}等分した同じ金額。空売り ＝ 点数で決まる金額 × 今の資金の規模。1回{CAP_PER / 1e4:,.0f}万円まで。株数は購入金額 ÷ 終値を100株単位で切り捨て。
明日の始値で買う（空売りする）ので、実際の金額は少し変わります。損益は往復{COST * 100:.1f}%の費用を引いた額。</div></div>
<div class="sub">{h.escape(data_date)} 終値時点。記録開始 {h.escape(ledger['start'])}。信用残 {h.escape(str(info.get('margin_date')))}・海外投資家の直近4週 {frgn_txt}（{h.escape(str(info.get('frgn_date')))}までの週）・信用取引の注意 {info.get('alert_n', 0)}銘柄・決算データ {h.escape(str(info.get('fins_last')))} まで。</div>
{warn}
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補（{len(today_buy)}件）</h2>{table(today_buy, today_cols)}</div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・新高値ブレイク（{len(today_nh)}件）</h2>{table(today_nh, nh_cols)}
<p class="sub">市場全体の新高値ブレイクの件数（30〜6日前に買った分）：<b>{info.get('nb30', '—')}件・{h.escape(str(info.get('nb_state', '—')))}</b>
（{NB_WEAK}件以下は弱い相場なので金額を半分、{NB_HOT}件超は過熱なので見送り）。購入金額はこれを反映済み。<br>損切り・利確の目安は今日の終値から計算。実際は明日の始値（買値）の−6%で損切り・+25%で利確（逆指値・指値を入れておく）。120営業日で手じまい。業種の空売り比率 {h.escape(str(info.get('sr_date')))}・空売り残高報告 {h.escape(str(info.get('ss_date')))} まで。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで空売りする候補（{len(today_short)}件）</h2>{table(today_short, sh_cols)}</div>
<div class="card"><h2 style="margin-top:0">これまでの成績（フォワードテスト）</h2><div class="stats">
<div class="stat"><div class="k">買い：合図／決済済み</div><div class="v">{sb_['n']}／{sb_['closed']}件</div></div>
<div class="stat"><div class="k">買い：決済済みの損益（勝率）</div><div class="v">{yen(sb_['pnl_closed'])}（{sb_['win'] if sb_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">買い：保有中の含み損益</div><div class="v">{yen(sb_['pnl_open'])}</div></div>
<div class="stat"><div class="k">新高値：合図／決済済み</div><div class="v">{sn_['n']}／{sn_['closed']}件</div></div>
<div class="stat"><div class="k">新高値：決済済みの損益（勝率）</div><div class="v">{yen(sn_['pnl_closed'])}（{sn_['win'] if sn_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">新高値：保有中の含み損益</div><div class="v">{yen(sn_['pnl_open'])}</div></div>
<div class="stat"><div class="k">空売り：合図／決済済み</div><div class="v">{ss_['n']}／{ss_['closed']}件</div></div>
<div class="stat"><div class="k">空売り：決済済みの損益（勝率）</div><div class="v">{yen(ss_['pnl_closed'])}（{ss_['win'] if ss_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">空売り：保有中の含み損益</div><div class="v">{yen(ss_['pnl_open'])}</div></div></div></div>
<div class="card"><h2 style="margin-top:0">保有中（{len(hold)}件）</h2>{table(hold, hold_cols)}</div>
<div class="card"><h2 style="margin-top:0">決済済み（{len(closed)}件）</h2>{table(closed, cl_cols)}</div>
<p class="sub">ルール：{h.escape(__doc__.split('採用ルール')[1].split('信用残・決算データ')[0].strip()).replace(chr(10), '<br>')}</p>
<p class="sub">条件に機械的に当てはまる銘柄を表示するもので、売買のおすすめではありません。空売りは損失に上限がなく、逆日歩・貸株料・配当相当額がかかることがあります。</p>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def equity(rows: list[dict]) -> float:
    """今の資金 ＝ 最初の資金 ＋ 決済済みの損益（費用を引いた額）。"""
    return CAPITAL + sum(r["pnl"] for r in rows if r["status"] == "決済済み")


def size(cands: list[dict], rows: list[dict], eq: float, side: str) -> None:
    """今の資金に合わせて、購入金額・株数・枠の空きを付ける（点数の高い順に枠を使う）。"""
    f = eq / BASE
    cap = 10_000_000 * f
    mx = 10
    unit = lambda x: min(x["lot"] * 1e6 * f, CAP_PER)
    if side == "buy":                                      # 現物の4割（バックテスト：1,000万円で枠400万円・金額は元の0.4倍）
        cap = eq * ALLOC_BUY
        unit = lambda x: min(x["lot"] * 1e6 * ALLOC_BUY * eq / 1e7, CAP_PER)
    elif side == "newhigh":                                # 現物の6割を12等分（全銘柄同じ金額）
        cap, mx = eq * ALLOC_NH, NH_MAX
        unit = lambda x: min(eq * ALLOC_NH / NH_MAX, CAP_PER) * x.get("mult", 1.0)
    open_ = [r for r in rows if r["side"] == side and (r["status"] == "保有中" or r["status"].startswith("明日"))]
    used = sum(r.get("amount") or r["lot"] * 1e6 for r in open_)
    held = {r["code"] for r in open_}
    n = len(open_)
    for x in cands:
        want = unit(x)
        sh = int(want // x["close"] // 100 * 100) if x.get("close") else 0
        x["shares"], x["amount"], x["cap"] = sh, round(sh * x["close"]) if sh else 0, cap
        if x.get("paused"):
            x["slot"] = "見送り（海外投資家が売り越し中）"
        elif x.get("mult") == 0:
            x["slot"] = "見送り（新高値が多すぎる過熱相場）"
        elif not sh:
            x["slot"] = "100株に届かない"
        elif x["code"] in held:
            x["slot"] = "すでに保有中"
        elif n >= mx or used + x["amount"] > cap:
            x["slot"] = "枠がいっぱい（見送り）"
        else:
            x["slot"] = "枠あり"
            n += 1
            used += x["amount"]
            held.add(x["code"])


def run(master: pd.DataFrame, prices: dict, ranks: dict, out_dir: str) -> None:
    ledger = load_ledger()
    buys, shorts, nhs, info = candidates(master, prices, ranks)
    prev = evaluate(ledger, prices)
    eq = equity(prev)
    buys.sort(key=lambda x: -(x.get("eg") if x.get("eg") is not None else -9))   # バックテストと同じ：同じ日は増益率の高い順
    shorts.sort(key=lambda x: (-x["score"], x["code"]))
    size(buys, prev, eq, "buy")
    size(shorts, prev, eq, "short")
    nhs.sort(key=lambda x: (-x["score"], -x.get("rs", 0)))               # バックテストと同じ：点数の高い順→相対的な強さ
    size(nhs, prev, eq, "newhigh")
    info["equity"] = eq
    have = {(s["date"], s["side"], s["code"]) for s in ledger["signals"]}
    for s in buys + shorts + nhs:
        if s["side"] == "short" and (not s.get("margin_ok") or s.get("paused")):
            continue                                       # 海外投資家が大きく売り越し中は空売りを見送る                                       # 信用残が判定できない空売りは台帳に入れない
        if s.get("slot") != "枠あり":
            continue                                       # 実際に売買しないもの（枠いっぱい・100株に届かない）は記録しない
        if (s["date"], s["side"], s["code"]) not in have:
            ledger["signals"].append({k: v for k, v in s.items() if k not in ("eg", "slot")} | {"eg": s.get("eg")})
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    json.dump(ledger, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = evaluate(ledger, prices)
    data_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    build_page(ledger, rows, buys, shorts, info, data_date, os.path.join(out_dir, "forward2.html"), nhs)
    print(f"フォワードテスト2: 今日の買い候補 {len(buys)}件・新高値 {len(nhs)}件・空売り候補 {len(shorts)}件・台帳 {len(ledger['signals'])}件")
