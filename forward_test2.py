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
           買い戻し（2026-11-24 変更）：終値が−1σを割ったら、毎日「25日線に買い戻しの逆指値」と「−2σに買い戻しの指値」を入れる／それまでは終値が+1σ超えで翌朝損切り。
           同時10銘柄・建玉1,000万円まで（同じ日は点数の高い順）。
  新高値ブレイク（2026-11-23 採用）: 終値が52週高値を初めて上に抜けた日（その高値は5営業日以上前・20日平均の売買代金3,000万円以上）。
           100点満点の点数60点以上：PBR1.07倍未満20／ランク6以上20／TOPIXも52週高値圏15／配当利回り2.3%以上10／
           PERが業種の0.95倍未満10／業種の空売り比率40.6%未満10／決算から10日以内5／買い残が発行済の0.36%以上5／52週安値から60%以上上5。
           さらに：業績予想の下方修正なし・相対的株価強度50以上・空売り残高報告の合計4%未満・信用取引の注意に載っていない。
           翌営業日の始値で買い、−6%で損切り・+25%で利確・120営業日で手じまい。金額は全銘柄同じ（新高値の枠÷12）。同じ日は点数の高い順。
           市場全体の新高値ブレイクの件数（買った日が30〜6日前）が430件超なら見送り（2026-11-25：102件以下で半分のルールは廃止）。
  IPOセカンダリー（2026-11-23 採用）: 上場から250営業日以内に、終値が上場来高値を初めて上に抜けた日（上場から10営業日以上・その高値は5営業日以上前）。
           20日平均の売買代金1.43億円以上。4つのうち2つ以上：上場来高値からの下げが44%以内／決算から21日以内／上場来高値から21営業日以上／TOPIXが52週高値圏でない。
           さらに：信用取引の注意に載っていない・直近60日の上昇58%未満。翌営業日の始値で買い、−10%で損切り・+30%で利確・120営業日で手じまい。
  資金の分け方 : 現物の3つ（75日線の買い・新高値・IPO）は1つの枠（今の資金）を共有し、同じ日はIPO→新高値→75日線の買いの順。
           1回：75日線の買い＝上の金額×0.3×資金の規模／新高値＝資金の4%／IPO＝資金の5%。同時に持てる数は10・12・10銘柄。空売りは信用の建玉で別枠。
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
RULE_ID = "combo_v5"   # v2: 空売りはランク5以下→割高点数1以上、買いはROE4%未満を除く。v3: 空売りに海外投資家の見送り・信用取引の注意の加点・貸借銘柄だけ。v4: 新高値ブレイクを追加。v5: IPOを追加・現物3つで枠を共有（2026-11-23から）
LOT_BUY = {6: 0.5, 7: 1.0, 8: 1.5}
CAPITAL = float(os.environ.get("FORWARD2_CAPITAL", "10000000"))   # 最初の資金
BASE = 13_000_000      # 元のルール（買い1,000万円＋空売りの保証金300万円）の規模。資金がこの何倍かで金額を変える
CAP_PER = 3_000_000    # 1回の金額の上限
COST = 0.002           # 往復の売買コスト（値段のずれ・手数料）
W_SHORT = [0.5, 1, 1.5, 2, 2]
# 資金の分け方（2026-11-23 採用）：現物の3つ（75日線の買い・新高値・IPO）は1つの枠（今の資金）を共有して早い者勝ち。
# 1回の金額：75日線の買い＝ランクで決まる金額×0.3×（資金÷1,000万円）／新高値＝資金の4%／IPO＝資金の5%。空売りは信用の建玉で別枠
UNIT_BUY, UNIT_NH, UNIT_IPO = 0.3, 0.04, 0.05
MAX_N = {"buy": 10, "newhigh": 12, "ipo": 10}
LONG_ORDER = {"ipo": 0, "newhigh": 1, "buy": 2}    # 同じ日に枠が足りないときの順番（バックテストと同じ）
IPO_STOP, IPO_TP = 0.10, 0.30
NH_MAX = 12            # 新高値の同時保有数（1回 ＝ 新高値の枠 ÷ 12）
NH_STOP, NH_TP = 0.06, 0.25
NB_HOT = 430   # 市場全体の新高値ブレイクの件数（買った日が30〜6日前のもの）：430件超は過熱で見送り（102件以下で半分は 2026-11-25 に廃止：4つで枠を共有すると逆効果）
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


def ipo_list() -> pd.DataFrame:
    """直近約1年のIPO銘柄（JPXの新規上場会社情報のページ）。上場日・コード・市場。テクニカル上場（*）は除く。"""
    import re
    import requests
    rows = []
    for page in ("index.html", "00-archives-01.html"):
        try:
            t = requests.get("https://www.jpx.co.jp/listing/stocks/new/" + page, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
        except Exception as e:
            print(f"  新規上場のページを読めませんでした（{e}）")
            continue
        for tr in re.findall(r"<tr.*?</tr>", t, flags=re.S):
            txt = re.sub(r"<[^>]+>", " ", tr)
            txt = re.sub(r"\s+", " ", txt)
            m = re.search(r"(20\d\d)/(\d\d)/(\d\d)", txt)
            cd = re.search(r"\s([0-9][0-9A-Z]{2}[0-9A-Z])\s", txt)
            if m and cd:
                rows.append({"list_date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "code": cd.group(1), "tech": "*" in txt.split(cd.group(1))[0]})
    d = pd.DataFrame(rows).drop_duplicates("code") if rows else pd.DataFrame(columns=["list_date", "code", "tech"])
    print(f"  JPXの新規上場のページ: {len(d)}件（テクニカル上場 {int(d['tech'].sum()) if len(d) else 0}件）")
    # J-Quants で取った過去のIPO（data/jq/ipo_list）も足す（ページから外れた分の保険）
    if os.path.exists("data/jq/ipo_bars.csv.gz"):
        b = pd.read_csv("data/jq/ipo_bars.csv.gz", dtype={"Code": str}, usecols=["Date", "Code"])
        f = b.groupby("Code")["Date"].min().reset_index()
        f = pd.DataFrame({"list_date": f["Date"], "code": f["Code"].str[:4], "tech": False})
        d = pd.concat([d, f[~f["code"].isin(d["code"])]], ignore_index=True)
    lo = (date.today() - timedelta(days=380)).isoformat()
    d = d[(~d["tech"].astype(bool)) & (d["list_date"] >= lo) & (d["list_date"] <= date.today().isoformat())]
    return d.reset_index(drop=True)


def ipo_candidates(ipo_prices: dict, lst: pd.DataFrame, tpx, fins: dict, alert: set, names: dict) -> list[dict]:
    """IPOセカンダリー：上場から250営業日以内に、終値が上場来高値を初めて上に抜けた日（バックテスト ipo と同じ）。"""
    out = []
    ld = dict(zip(lst["code"], lst["list_date"]))
    for code, df in ipo_prices.items():
        if code not in ld or len(df) < 12:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        if abs((idx[0] - pd.Timestamp(ld[code])).days) > 5:
            continue                                       # 株価が上場日から始まっていない（初値が分からない）
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        L = len(c) - 1
        if L < 10 or L > 250:
            continue
        hm = np.maximum.accumulate(h)
        if not (c[L] > hm[L - 1] and c[L - 1] <= hm[L - 2]):
            continue
        pi = int(np.argmax(h[:L]))
        if L - pi < 5:
            continue
        tv20 = np.mean((c * v)[max(0, L - 19):L + 1])
        if tv20 < 1.43e8 or code in alert:
            continue
        ret60 = (c[L] / c[max(0, L - 60)] - 1) * 100
        if ret60 >= 58:
            continue
        sd_ = idx[L].strftime("%Y-%m-%d")
        recs = [r for r in fins.get(code, []) if r["DiscDate"] <= sd_]
        stm = [r for r in recs if "FinancialStatements" in str(r["DocType"])]
        dsince = (idx[L] - pd.Timestamp(stm[-1]["DiscDate"])).days if stm else None
        t = tpx.reindex(idx).ffill().to_numpy() if tpx is not None else None
        tpx_new = False
        if tpx is not None:
            tt = tpx[tpx.index <= idx[L]]
            tpx_new = bool(len(tt) and tt.iloc[-1] >= tt.tail(250).max() * 0.97)
        dd = (l[:L + 1].min() / hm[L] - 1) * 100
        pts = [("上場来高値からの下げが44%以内", dd > -43.8), ("決算から21日以内", dsince is not None and dsince <= 21),
               ("上場来高値から21営業日以上", L - pi >= 21), ("TOPIXが高値圏でない", not tpx_new)]
        p_ = sum(ok for _, ok in pts)
        if p_ >= 2:
            out.append({"date": sd_, "side": "ipo", "code": code, "name": names.get(code, code), "rank": None, "score": p_, "lot": 1.0,
                        "reasons": [n for n, ok in pts if ok], "close": round(float(c[L]), 1), "list_date": ld[code], "days": L})
    return out


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
    nb_state = "過熱（見送り）" if nb30 > NB_HOT else "普通"
    info.update(nb30=nb30, nb_state=nb_state)
    nb_mult = 0.0 if nb30 > NB_HOT else 1.0
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
        if s["side"] in ("newhigh", "ipo"):
            import newhigh_candidates as nh
            a = {"o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float), "l": df["Low"].to_numpy(float),
                 "c": df["Close"].to_numpy(float), "pivot": np.full(len(df), np.nan), "dates": dates}
            st_, tp_ = (NH_STOP, NH_TP) if s["side"] == "newhigh" else (IPO_STOP, IPO_TP)
            x = nh.sim(a, i, st_, tp=tp_)
            entry = a["o"][i + 1]
            r.update(entry=round(entry, 1), entry_date=dates[i + 1], stop_px=round(entry * (1 - st_), 1),
                     tp_px=round(entry * (1 + tp_), 1))
            if x is None:
                last = a["c"][-1]
                r.update(status="保有中", now=round(last, 1), ret=round((last / entry - 1) * 100, 2))
            else:
                r.update(status="決済済み", ret=float(x["ret"]), exit_date=x["exit"],
                         why={"stop": f"損切り（−{st_ * 100:.0f}%）", "tp": f"利確（+{tp_ * 100:.0f}%）", "time": "120営業日で手じまい"}.get(x["reason"], x["reason"]))
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
            x = b2.simulate(s["code"], df, i, "lk25", {})
            closed = not x["why"].startswith("保有中")
            r.update(status="決済済み" if closed else "保有中", entry=x["entry"], ret=x["ret_pct"], entry_date=x["entry_date"],
                     exit_date=x["exit_date"] if closed else None, why=x["why"])
            if not closed:   # 明日の注文（今日の終値で計算）
                cs = pd.Series(df["Close"].to_numpy(float)); m25, sd = cs.rolling(25).mean(), cs.rolling(25).std(ddof=0)
                armed = bool((cs.iloc[i + 1:] < (m25 - sd).iloc[i + 1:]).any())
                if armed:
                    r["order"] = f'逆指値 {m25.iloc[-1]:,.0f}円（25日線）／指値 {(m25 - 2 * sd).iloc[-1]:,.0f}円（−2σ）で買い戻し'
                else:
                    r["order"] = f'終値が{(m25 + sd).iloc[-1]:,.0f}円（+1σ）を超えたら翌朝買い戻し'
            amt = s.get("amount") or s["lot"] * 1e6
            r["pnl"] = round(amt * x["pnl"] / max(x["notional"], 1) - amt * COST)
        out.append(r)
    # 同時10銘柄・1,000万円まで（買いは増益率の高い順、空売りは点数の高い順）
    amt = lambda x: x.get("amount") or x["lot"] * 1e6
    # 空売り：自分の枠（10銘柄・建玉の上限）
    rows = sorted([r for r in out if r["side"] == "short" and r.get("entry_date")], key=lambda r: (r["entry_date"], -r["score"]))
    open_ = []
    for r in rows:
        open_ = [x for x in open_ if (x.get("exit_date") or "9999") > r["entry_date"]]
        if len(open_) >= 10 or sum(amt(x) for x in open_) + amt(r) > r.get("cap", 1e7) * 1.0001 or any(x["code"] == r["code"] for x in open_):
            r["status"] = "見送り（枠がいっぱい）"
            continue
        open_.append(r)
    # 現物の3つ：1つの枠を共有（手法ごとの銘柄数の上限＋金額の合計が資金まで。同じ銘柄は1つだけ）
    rows = [r for r in out if r["side"] in ("buy", "newhigh", "ipo") and r.get("entry_date")]
    rows.sort(key=lambda r: (r["entry_date"], LONG_ORDER.get(r["side"], 9), -(r.get("eg") or -9) if r["side"] == "buy" else -r["score"]))
    open_ = []
    for r in rows:
        open_ = [x for x in open_ if (x.get("exit_date") or "9999") > r["entry_date"]]
        ns = sum(1 for x in open_ if x["side"] == r["side"])
        if ns >= MAX_N.get(r["side"], 10) or sum(amt(x) for x in open_) + amt(r) > r.get("cap", 1e7) * 1.0001 or any(x["code"] == r["code"] for x in open_):
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
def my_holdings(hold: list[dict]) -> str:
    """「自分の保有」：システムの保有中の記録から、買わなかったものを外し、買値を直せる欄（スマホのブラウザに保存）。"""
    import json as _j
    SIDE = {"buy": "買い（75日線）", "newhigh": "買い（新高値）", "ipo": "買い（IPO）", "short": "空売り"}
    items = []
    for r in hold:
        e = float(r.get("entry") or 0)
        ret = float(r.get("ret") or 0)
        now = e * (1 + ret / 100) if r["side"] != "short" else e * (1 - ret / 100)
        stop, tp = {"newhigh": (NH_STOP, NH_TP), "ipo": (IPO_STOP, IPO_TP)}.get(r["side"], (None, None))
        items.append({"id": f'{r["date"]}|{r["side"]}|{r["code"]}', "side": r["side"], "label": SIDE.get(r["side"], r["side"]),
                      "code": r["code"], "name": r.get("name", ""), "shares": int(r.get("shares") or 0), "entry": round(e, 1),
                      "now": round(now, 1), "entry_date": r.get("entry_date") or "", "stop": stop, "tp": tp})
    data = _j.dumps(items, ensure_ascii=False)
    return """<div class="card"><h2 style="margin-top:0">自分の保有</h2>
<p class="sub">システムの「保有中」から、<b>買わなかったものは「買わなかった」</b>に切り替え、買値が違うときは直してください（このスマホのブラウザに保存されます）。
損切り・利確は直した買値から計算します。75日線の買いと空売りは、毎日このページの判定（保有中の表）に従ってください。</p>
<div id="mysum" class="stats"></div><div class="tw"><table id="mytab"></table></div>
<p class="sub" id="mynote"></p></div>
<script>
(function(){
var D=""" + data + """;
var KEY='kabu_mypos_v1', st={};
try{st=JSON.parse(localStorage.getItem(KEY)||'{}')||{};}catch(e){st={};}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));}catch(e){}}
function yen(v){return (v>=0?'+':'')+Math.round(v).toLocaleString('ja-JP')+'円';}
function addBiz(d,n){var x=new Date(d+'T00:00:00');var k=0;while(k<n){x.setDate(x.getDate()+1);var w=x.getDay();if(w!==0&&w!==6)k++;}return x.toISOString().slice(0,10);}
function draw(){
  var tb=document.getElementById('mytab'); var h='<tr><th>買った？</th><th>売買</th><th>銘柄</th><th>株数</th><th>買値</th><th>今</th><th>損益</th><th>損切り／利確</th><th>手じまい予定</th></tr>';
  var tot=0,pl=0,n=0;
  D.forEach(function(r){
    var s=st[r.id]||{}; var skip=!!s.skip; var px=(s.px>0)?s.px:r.entry;
    var sign=r.side==='short'?-1:1; var p=(r.now/px-1)*sign; var v=p*px*r.shares;
    if(!skip){tot+=px*r.shares; pl+=v; n++;}
    var sl=r.stop?Math.round(px*(1-r.stop)).toLocaleString('ja-JP')+'円／'+Math.round(px*(1+r.tp)).toLocaleString('ja-JP')+'円':'ページの判定に従う';
    var end=r.entry_date?addBiz(r.entry_date,120):'';
    h+='<tr style="'+(skip?'opacity:.45':'')+'"><td><button data-id="'+r.id+'" class="tg" style="padding:4px 8px;border-radius:6px;border:1px solid #ccc;background:'+(skip?'#fff':'#e8f0fb')+'">'+(skip?'買わなかった':'買った')+'</button></td>'
      +'<td>'+r.label+'</td><td>'+r.code+' '+r.name+'</td><td>'+r.shares.toLocaleString('ja-JP')+'株</td>'
      +'<td><input data-id="'+r.id+'" class="px" type="number" inputmode="decimal" value="'+px+'" style="width:84px"></td>'
      +'<td>'+r.now.toLocaleString('ja-JP')+'</td><td>'+(p*100).toFixed(2)+'%（'+yen(v)+'）</td><td>'+sl+'</td><td>'+end+'</td></tr>';
  });
  tb.innerHTML=D.length?h:'<tr><td class="none">保有中の記録はまだありません</td></tr>';
  document.getElementById('mysum').innerHTML='<div class="stat"><div class="k">実際に持っている銘柄</div><div class="v">'+n+'銘柄</div></div>'
    +'<div class="stat"><div class="k">買った金額の合計</div><div class="v">'+Math.round(tot).toLocaleString('ja-JP')+'円</div></div>'
    +'<div class="stat"><div class="k">含み損益</div><div class="v">'+yen(pl)+'</div></div>';
  Array.prototype.forEach.call(document.querySelectorAll('#mytab .tg'),function(b){b.onclick=function(){var id=b.dataset.id;st[id]=st[id]||{};st[id].skip=!st[id].skip;save();draw();};});
  Array.prototype.forEach.call(document.querySelectorAll('#mytab .px'),function(i){i.onchange=function(){var id=i.dataset.id;var v=parseFloat(i.value);st[id]=st[id]||{};if(v>0)st[id].px=v;else delete st[id].px;save();draw();};});
}
draw();
})();
</script>"""


def build_page(ledger: dict, rows: list[dict], today_buy: list, today_short: list, info: dict, data_date: str, path: str,
               today_nh: list | None = None, today_ipo: list | None = None) -> None:
    today_nh = today_nh or []
    today_ipo = today_ipo or []
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
    # 補欠：条件はすべて満たしたが、金額・枠・相場の都合でルールでは見送ったもの（すでに保有中は除く）
    subs = []
    for side_, xs in (("buy", today_buy), ("newhigh", today_nh), ("ipo", today_ipo), ("short", today_short)):
        for x in xs:
            if x.get("slot") not in (None, "枠あり", "すでに保有中"):
                subs.append(dict(x, side_=side_))
    def sub_exit(x):
        if x["side_"] == "newhigh":
            return f'損切り {x["close"] * (1 - NH_STOP):,.0f}円／利確 {x["close"] * (1 + NH_TP):,.0f}円（終値で計算・買値で計算し直す）'
        if x["side_"] == "ipo":
            return f'損切り {x["close"] * (1 - IPO_STOP):,.0f}円／利確 {x["close"] * (1 + IPO_TP):,.0f}円（終値で計算・買値で計算し直す）'
        return "毎日の判定どおり（終値で−1σ割れなど）" if x["side_"] == "buy" else "毎日の判定どおり（+1σ超えで買い戻しなど）"
    sub_cols = [("手法", lambda x: SIDE_NAME[x["side_"]]), ("銘柄", link), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'),
                ("100株の金額", lambda x: man(x.get("close", 0) * 100)), ("ルールの1回の金額", lambda x: man(x.get("want", 0))),
                ("補欠になった理由", lambda x: h.escape(x.get("slot", "") + "：" + x.get("why", ""))),
                ("点数", lambda x: x.get("score", "—")), ("買うなら", lambda x: h.escape(sub_exit(x)))]
    sub_card = (f'''<div class="card"><h2 style="margin-top:0">補欠（条件は満たしたが、ルールでは見送り）（{len(subs)}件）</h2>{table(subs, sub_cols)}
<p class="sub">売買の条件はすべて満たしていて、金額・枠・相場の都合だけで見送った銘柄です。システムの記録（フォワードテスト）には入りません。
自分で買う場合は100株で、損切り・利確の注文を自分で入れてください。10年の検証では「100株に届かなくても資金の7%以内なら100株買う」を足しても成績はほぼ同じでした。</p></div>'''
                if subs else "")
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
    ipo_cols = [("銘柄", link), ("上場日", lambda x: x.get("list_date", "")), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'),
                ("購入金額", lambda x: man(x.get("amount", 0))), ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
                ("条件", lambda x: f'{x["score"]}/4'), ("損切り目安", lambda x: f'{x["close"] * (1 - IPO_STOP):,.0f}円'),
                ("利確目安", lambda x: f'{x["close"] * (1 + IPO_TP):,.0f}円'), ("当てはまった条件", lambda x: h.escape("・".join(x["reasons"]) or "—"))]
    hold = [r for r in rows if r["status"] == "保有中"]
    closed = sorted([r for r in rows if r["status"] == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    SIDE = {"buy": "買い（75日線）", "newhigh": "買い（新高値）", "ipo": "買い（IPO）", "short": "空売り"}
    pos_cols = [("売買", lambda x: SIDE.get(x["side"], x["side"])), ("銘柄", link), ("合図", lambda x: x["date"]),
                ("金額", lambda x: man(x["amount"]) if x.get("amount") else lot(x["lot"])), ("株数", lambda x: f'{x["shares"]:,}株' if x.get("shares") else "—"),
                ("建値", lambda x: x.get("entry")), ("損益", lambda x: f'{x["ret"]:+.2f}%（{yen(x["pnl"])}）')]
    hold_cols = pos_cols + [("明日の注文（損切り／利確）", lambda x: f'損切り {x["stop_px"]:,.0f}円／利確 {x["tp_px"]:,.0f}円' if x.get("stop_px")
                             else h.escape(x.get("order") or "毎日の判定に従う"))]
    cl_cols = pos_cols + [("決済", lambda x: x.get("exit_date") or ""), ("理由", lambda x: h.escape(x.get("why") or ""))]
    sb_, ss_, sn_, si_ = stats(rows, "buy"), stats(rows, "short"), stats(rows, "newhigh"), stats(rows, "ipo")
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
<title>今日の売買（買い＋新高値＋IPO＋空売り）</title><style>
body{{font-family:'Noto Sans JP',system-ui,sans-serif;margin:0;background:#f5f4f0;color:#1d1d1b;font-size:14px;line-height:1.55}}
main{{max-width:1100px;margin:0 auto;padding:16px}}h1{{font-size:18px;margin:4px 0 2px}}h2{{font-size:15px;margin:22px 0 8px}}
.sub{{color:#6b6a65;font-size:12px}}.card{{background:#fff;border:1px solid #e3e1db;border-radius:10px;padding:12px 14px;margin-top:10px}}
.tw{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px 8px;border-bottom:1px solid #efede7;text-align:left;white-space:nowrap}}
th{{color:#6b6a65;font-weight:500;font-size:12px}}.none{{color:#6b6a65}}.warn{{background:#fff8ec;border:1px solid #f0d9b0;color:#b26b00;padding:8px 10px;border-radius:6px;font-size:12.5px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}}.stat{{background:#faf9f6;border:1px solid #efede7;border-radius:8px;padding:8px 10px}}
.stat .k{{font-size:11px;color:#6b6a65}}.stat .v{{font-size:16px;font-weight:700}}a{{color:#1f4e79}}</style></head><body><main>
<a href="./" class="sub">← スクリーナーに戻る</a>　<a href="docs.html" class="sub">📘 まとめ資料（PDF）</a>
<h1>今日の売買（75日線の買い＋新高値ブレイク＋IPO＋空売り）</h1>
<div class="card"><div class="stats">
<div class="stat"><div class="k">今の資金（最初 {CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{info.get('equity', CAPITAL) / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">規模（元のルールの1,300万円に対して）</div><div class="v">{info.get('equity', CAPITAL) / BASE * 100:.0f}%</div></div>
<div class="stat"><div class="k">1回の上限</div><div class="v">{CAP_PER / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">現物の枠（3つで共有）</div><div class="v">{info.get('equity', CAPITAL) / 1e4:,.0f}万円まで</div></div>
<div class="stat"><div class="k">1回の金額：新高値／IPO</div><div class="v">{info.get('equity', CAPITAL) * UNIT_NH / 1e4:,.0f}万円／{info.get('equity', CAPITAL) * UNIT_IPO / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">空売りの枠（信用の建玉）</div><div class="v">10銘柄・{info.get('equity', CAPITAL) / BASE * 1000:,.0f}万円まで</div></div></div>
<div class="sub" style="margin-top:6px">現物の3つ（75日線の買い・新高値・IPO）は1つの枠を共有し、同じ日に枠が足りないときはIPO→新高値→75日線の買いの順。同時に持てる数は75日線の買い10・新高値12・IPO10銘柄まで。
75日線の買い ＝ 点数で決まる金額 × 0.3 × 資金の規模。新高値 ＝ 資金の4%（新高値が多すぎる過熱相場では見送り）。IPO ＝ 資金の5%。空売り ＝ 点数で決まる金額 × 今の資金の規模。1回{CAP_PER / 1e4:,.0f}万円まで。株数は購入金額 ÷ 終値を100株単位で切り捨て。
明日の始値で買う（空売りする）ので、実際の金額は少し変わります。損益は往復{COST * 100:.1f}%の費用を引いた額。</div></div>
<div class="sub">{h.escape(data_date)} 終値時点。記録開始 {h.escape(ledger['start'])}。信用残 {h.escape(str(info.get('margin_date')))}・海外投資家の直近4週 {frgn_txt}（{h.escape(str(info.get('frgn_date')))}までの週）・信用取引の注意 {info.get('alert_n', 0)}銘柄・決算データ {h.escape(str(info.get('fins_last')))} まで。</div>
{warn}
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補（{len(today_buy)}件）</h2>{table(today_buy, today_cols)}</div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・新高値ブレイク（{len(today_nh)}件）</h2>{table(today_nh, nh_cols)}
<p class="sub">市場全体の新高値ブレイクの件数（30〜6日前に買った分）：<b>{info.get('nb30', '—')}件・{h.escape(str(info.get('nb_state', '—')))}</b>
（{NB_HOT}件超は過熱なので見送り）。購入金額はこれを反映済み。<br>損切り・利確の目安は今日の終値から計算。実際は明日の始値（買値）の−6%で損切り・+25%で利確（逆指値・指値を入れておく）。120営業日で手じまい。業種の空売り比率 {h.escape(str(info.get('sr_date')))}・空売り残高報告 {h.escape(str(info.get('ss_date')))} まで。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・IPOセカンダリー（{len(today_ipo)}件）</h2>{table(today_ipo, ipo_cols)}
<p class="sub">直近1年のIPO {info.get('ipo_n', '—')}銘柄から判定。上場来高値を初めて抜け、20日平均の売買代金1.43億円以上、4つの条件のうち2つ以上、信用取引の注意なし、60日の上昇58%未満。
明日の始値（買値）の−10%で損切り・+30%で利確（OCO注文）。120営業日で手じまい。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで空売りする候補（{len(today_short)}件）</h2>{table(today_short, sh_cols)}
<p class="sub">買い戻し：終値が+1σを超えたら翌朝買い戻し（損切り）。終値が−1σを割ったら、その後は毎日「25日線に逆指値」と「−2σに指値」の買い戻し注文を入れる（値段は下の保有中の表に毎日出ます）。60営業日で手じまい。</p></div>
{sub_card}
<div class="card"><h2 style="margin-top:0">これまでの成績（フォワードテスト）</h2><div class="stats">
<div class="stat"><div class="k">買い：合図／決済済み</div><div class="v">{sb_['n']}／{sb_['closed']}件</div></div>
<div class="stat"><div class="k">買い：決済済みの損益（勝率）</div><div class="v">{yen(sb_['pnl_closed'])}（{sb_['win'] if sb_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">買い：保有中の含み損益</div><div class="v">{yen(sb_['pnl_open'])}</div></div>
<div class="stat"><div class="k">新高値：合図／決済済み</div><div class="v">{sn_['n']}／{sn_['closed']}件</div></div>
<div class="stat"><div class="k">新高値：決済済みの損益（勝率）</div><div class="v">{yen(sn_['pnl_closed'])}（{sn_['win'] if sn_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">新高値：保有中の含み損益</div><div class="v">{yen(sn_['pnl_open'])}</div></div>
<div class="stat"><div class="k">IPO：合図／決済済み</div><div class="v">{si_['n']}／{si_['closed']}件</div></div>
<div class="stat"><div class="k">IPO：決済済みの損益（勝率）</div><div class="v">{yen(si_['pnl_closed'])}（{si_['win'] if si_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">IPO：保有中の含み損益</div><div class="v">{yen(si_['pnl_open'])}</div></div>
<div class="stat"><div class="k">空売り：合図／決済済み</div><div class="v">{ss_['n']}／{ss_['closed']}件</div></div>
<div class="stat"><div class="k">空売り：決済済みの損益（勝率）</div><div class="v">{yen(ss_['pnl_closed'])}（{ss_['win'] if ss_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">空売り：保有中の含み損益</div><div class="v">{yen(ss_['pnl_open'])}</div></div></div></div>
{my_holdings(hold)}
<div class="card"><h2 style="margin-top:0">保有中（システムの記録・{len(hold)}件）</h2>{table(hold, hold_cols)}</div>
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
    open_ = [r for r in rows if r["side"] == side and (r["status"] == "保有中" or r["status"].startswith("明日"))]
    used = sum(r.get("amount") or r["lot"] * 1e6 for r in open_)
    held = {r["code"] for r in open_}
    n = len(open_)
    for x in cands:
        want = unit(x)
        sh = int(want // x["close"] // 100 * 100) if x.get("close") else 0
        x["shares"], x["amount"], x["cap"], x["want"] = sh, round(sh * x["close"]) if sh else 0, cap, want
        if x.get("paused"):
            x["slot"] = "見送り（海外投資家が売り越し中）"
            x["why"] = "海外投資家の直近4週の売り越しが大きく、空売りを止める期間"
        elif x.get("mult") == 0:
            x["slot"] = "見送り（新高値が多すぎる過熱相場）"
        elif not sh:
            x["slot"] = "100株に届かない"
            x["why"] = f'1回の上限 {want / 1e4:,.1f}万円に対して、100株で {x["close"] * 100 / 1e4:,.1f}万円かかる'
        elif x["code"] in held:
            x["slot"] = "すでに保有中"
        elif n >= mx or used + x["amount"] > cap:
            x["slot"] = "枠がいっぱい（見送り）"
            x["why"] = (f"空売りの同時保有の上限（{mx}銘柄）に達している" if n >= mx
                        else f"空売りの建玉の枠（{cap / 1e4:,.0f}万円）の残り {max(cap - used, 0) / 1e4:,.1f}万円では足りない")
        else:
            x["slot"] = "枠あり"
            n += 1
            used += x["amount"]
            held.add(x["code"])


SIDE_NAME = {"buy": "75日線の買い", "newhigh": "新高値", "ipo": "IPO", "short": "空売り"}


def size_long(cands: list[dict], rows: list[dict], eq: float) -> None:
    """現物の3つ（買い・新高値・IPO）を1つの枠（今の資金）で共有。cands はすでに枠を使う順に並んでいること。"""
    open_ = [r for r in rows if r["side"] in ("buy", "newhigh", "ipo") and (r["status"] == "保有中" or r["status"].startswith("明日"))]
    used = sum(r.get("amount") or 0 for r in open_)
    held = {r["code"] for r in open_}
    n = {k: sum(1 for r in open_ if r["side"] == k) for k in MAX_N}
    for x in cands:
        if x["side"] == "buy":
            want = x["lot"] * 1e6 * UNIT_BUY * eq / 1e7
        elif x["side"] == "newhigh":
            want = eq * UNIT_NH * x.get("mult", 1.0)
        else:
            want = eq * UNIT_IPO
        want = min(want, CAP_PER)
        sh = int(want // x["close"] // 100 * 100) if x.get("close") else 0
        x["shares"], x["amount"], x["cap"], x["want"] = sh, round(sh * x["close"]) if sh else 0, eq, want
        if x.get("mult") == 0:
            x["slot"] = "見送り（新高値が多すぎる過熱相場）"
            x["why"] = f"市場全体の新高値ブレイクが多すぎる（{NB_HOT}件超）過熱相場なので、新高値は買わない"
        elif not sh:
            x["slot"] = "100株に届かない"
            x["why"] = f'1回の上限 {want / 1e4:,.1f}万円に対して、100株で {x["close"] * 100 / 1e4:,.1f}万円かかる'
        elif x["code"] in held:
            x["slot"] = "すでに保有中"
        elif n[x["side"]] >= MAX_N[x["side"]] or used + x["amount"] > eq:
            x["slot"] = "枠がいっぱい（見送り）"
            x["why"] = (f'{SIDE_NAME[x["side"]]}の同時保有の上限（{MAX_N[x["side"]]}銘柄）に達している' if n[x["side"]] >= MAX_N[x["side"]]
                        else f"資金の枠の残り {max(eq - used, 0) / 1e4:,.1f}万円では、{x['amount'] / 1e4:,.1f}万円が足りない")
        else:
            x["slot"] = "枠あり"
            n[x["side"]] += 1
            used += x["amount"]
            held.add(x["code"])


def run(master: pd.DataFrame, prices: dict, ranks: dict, out_dir: str) -> None:
    ledger = load_ledger()
    buys, shorts, nhs, info = candidates(master, prices, ranks)
    # IPO：直近1年のIPO銘柄の株価（全市場）を取って判定
    ipos, ipo_prices = [], {}
    try:
        import kabu_75ma_search as k
        lst = ipo_list()
        info["ipo_n"] = len(lst)
        print(f"  IPO銘柄（直近1年）: {len(lst)}件・最新 {lst.sort_values('list_date').tail(3)[['list_date', 'code']].values.tolist()}")
        if len(lst):
            ipo_prices = k.download_prices(sorted(lst["code"].unique()), period="2y")
            ipo_prices = {c: d for c, d in ipo_prices.items() if len(d)}
            names = dict(zip(master["コード"], master["銘柄名"]))
            if os.path.exists("data/jq/master.csv"):
                jm = pd.read_csv("data/jq/master.csv", dtype=str)
                names.update({c[:4]: n for c, n in zip(jm["Code"], jm["CoName"]) if c[:4] not in names})
            ipos = ipo_candidates(ipo_prices, lst, topix(), fins_recent(), margin_alert_recent(), names)
    except Exception as e:
        print(f"  IPOの判定に失敗（{e}）")
    allp = {**ipo_prices, **prices}
    prev = evaluate(ledger, allp)
    eq = equity(prev)
    buys.sort(key=lambda x: -(x.get("eg") if x.get("eg") is not None else -9))   # バックテストと同じ：同じ日は増益率の高い順
    shorts.sort(key=lambda x: (-x["score"], x["code"]))
    size(shorts, prev, eq, "short")
    nhs.sort(key=lambda x: (-x["score"], -x.get("rs", 0)))               # バックテストと同じ：点数の高い順→相対的な強さ
    ipos.sort(key=lambda x: -x["score"])
    size_long(ipos + nhs + buys, prev, eq)                              # 枠を使う順：IPO→新高値→75日線の買い
    info["equity"] = eq
    have = {(s["date"], s["side"], s["code"]) for s in ledger["signals"]}
    for s in buys + shorts + nhs + ipos:
        if s["side"] == "short" and (not s.get("margin_ok") or s.get("paused")):
            continue                                       # 海外投資家が大きく売り越し中は空売りを見送る                                       # 信用残が判定できない空売りは台帳に入れない
        if s.get("slot") != "枠あり":
            continue                                       # 実際に売買しないもの（枠いっぱい・100株に届かない）は記録しない
        if (s["date"], s["side"], s["code"]) not in have:
            ledger["signals"].append({k: v for k, v in s.items() if k not in ("eg", "slot")} | {"eg": s.get("eg")})
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    json.dump(ledger, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = evaluate(ledger, allp)
    data_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    build_page(ledger, rows, buys, shorts, info, data_date, os.path.join(out_dir, "forward2.html"), nhs, ipos)
    print(f"フォワードテスト2: 今日の買い候補 {len(buys)}件・新高値 {len(nhs)}件・IPO {len(ipos)}件・空売り候補 {len(shorts)}件・台帳 {len(ledger['signals'])}件")
    for lab, xs in (("75日線", buys), ("新高値", nhs), ("IPO", ipos), ("空売り", shorts)):
        for x in xs:   # 候補の中身をログにも残す（ページを見られない時の確認用）
            print("  候補", lab, {kk: x.get(kk) for kk in ("code", "name", "signal", "close", "score", "slot", "shares", "yen", "amount", "lot") if kk in x})
