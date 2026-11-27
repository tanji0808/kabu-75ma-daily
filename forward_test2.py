"""
フォワードテスト2（買い＋空売り・点数で金額を変える）と「今日の売買（買い＋空売り）」ページ

採用ルール（10年のバックテスト・その時点の決算・上場廃止銘柄を含めた検証で選んだもの）
  買い   : 75日線ブレイク（本日突破）＋5日+3%以下＋RSI60以下＋ファンダメンタルランク6以上＋ROE4%以上。翌営業日の始値で買う。
           市場全体の信用倍率（週次）が3.67倍未満の時期は見送り（2026-11-27 追加）。
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
           翌営業日の始値で買い、−6%で損切り。終値で+20%に届いたら、そこからは終値の最高値の−20%（最低でも買値）で追いかけ、利確はしない（2026-11-27 変更）。120営業日で手じまい。金額は全銘柄同じ（新高値の枠÷12）。同じ日は点数の高い順。
           市場全体の新高値ブレイクの件数（買った日が30〜6日前）が430件超なら見送り（2026-11-25：102件以下で半分のルールは廃止）。
           米S&P500の前日終値が200日線より下の時も見送り（2026-11-27 追加）。
  IPOセカンダリー（2026-11-23 採用）: 上場から250営業日以内に、終値が上場来高値を初めて上に抜けた日（上場から10営業日以上・その高値は5営業日以上前）。
           20日平均の売買代金1.43億円以上。4つのうち2つ以上：上場来高値からの下げが44%以内／決算から21日以内／上場来高値から21営業日以上／TOPIXが52週高値圏でない。
           さらに：信用取引の注意に載っていない・直近60日の上昇58%未満。翌営業日の始値で買い、−10%で損切り・+30%で利確・120営業日で手じまい。
  資金の分け方 : 現物の7つは1つの枠（今の資金）を共有し、同じ日は窓あけ下げ→権利取り前→リバウンド→IPO→新高値→IR→75日線の買いの順（2026-11-27 時点）。
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
RULE_ID = "combo_v13"   # v2: 空売りはランク5以下→割高点数1以上、買いはROE4%未満を除く。v3: 空売りに海外投資家の見送り・信用取引の注意の加点・貸借銘柄だけ。v4: 新高値ブレイクを追加。v5: IPOを追加・現物3つで枠を共有（2026-11-23から）。v6: IRトレードを追加・現物4つで枠を共有・資金1,500万円（2026-11-27から）。v7: リバウンドと権利取り前を追加（2026-11-27から）。v8: 外部要因の再チェック：IRは営業利益率4.6%以上・新高値はS&P500が200日線より上・75日線の買いは市場の信用倍率3.67倍以上（2026-11-27から）。v9: 年末の反発（別枠1,000万円）を追加。v10: 窓あけ下げ（急落の2日目から）を追加。v11: 新高値の追いかけを+20%・−20%に。v12: リバウンドに25日線−10%での利確を追加。v13: 窓あけ下げは急落の朝のあと5営業日、毎朝買う（今朝の数の確認をやめた）
LOT_BUY = {6: 0.5, 7: 1.0, 8: 1.5}
CAPITAL = float(os.environ.get("FORWARD2_CAPITAL", "15000000"))   # 最初の資金
BASE = 13_000_000      # 元のルール（買い1,000万円＋空売りの保証金300万円）の規模。資金がこの何倍かで金額を変える
CAP_PER = 3_000_000    # 1回の金額の上限
COST = 0.002           # 往復の売買コスト（値段のずれ・手数料）
W_SHORT = [0.5, 1, 1.5, 2, 2]
# 資金の分け方（2026-11-23 採用）：現物の3つ（75日線の買い・新高値・IPO）は1つの枠（今の資金）を共有して早い者勝ち。
# 1回の金額：75日線の買い＝ランクで決まる金額×0.3×（資金÷1,000万円）／新高値＝資金の4%／IPO＝資金の5%。空売りは信用の建玉で別枠
UNIT_BUY, UNIT_NH, UNIT_IPO, UNIT_IR, UNIT_RB, UNIT_KR = 0.3, 0.04, 0.05, 0.05, 0.05, 0.20
MAX_N = {"buy": 10, "newhigh": 12, "ipo": 10, "ir": 20, "rebound": 10, "kenri": 1, "gapdown": 10}
LONG_ORDER = {"gapdown": -3, "kenri": -2, "rebound": -1, "ipo": 0, "newhigh": 1, "ir": 2, "buy": 3}
RB_TP_DEV = -0.10   # リバウンドの利確：合図の日の25日線の−10%の値段（2026-11-27 追加）
RB_LIMIT, RB_STOP, RB_HOLD = 0.03, 0.10, 3   # リバウンド（2026-11-27 追加）：前日終値−3%に指値、買値−10%で損切り、買った日を1日目として3日目の引けで売る
YE_CAP, YE_N, YE_YTD, YE_TV = 10_000_000, 20, -0.40, 3e7   # 年末の反発（2026-11-27 追加・別枠1,000万円）：その年に40%以上下げた銘柄（売買代金3,000万円以上）を下げの大きい順に20銘柄
YE_BUY, YE_SELL = 5, 5   # 大納会の5営業日前の引けで買い、1月の5営業日目の引けで売る
UNIT_GD, GD_GAP, GD_DEV, GD_TV, GD_TP, GD_STOP = 0.05, 0.05, 0.05, 1e8, 0.10, 0.10   # 窓あけ下げ（2026-11-27 追加）
GD_CRASH, GD_WIN = 20, 5   # 監視リストのうち20銘柄以上が−5%以下で始まった朝＝「急落の朝」。そのあと5営業日のあいだ買う（11/27変更）
KR_DAYS = 10                                   # 権利取り前（2026-11-27 追加）：3月・9月の権利付き最終日の10営業日前の引けでTOPIX連動ETF（1306）を資金の20%買い、権利付き最終日の引けで売る
IR_HOLD, IR_STOP = 60, 0.10   # IRトレード：買値−10%に損切りの逆指値（2026-11-27 追加）。当たらなければ反応の日から60営業日目の引けで売る。利確なし    # 同じ日に枠が足りないときの順番（バックテストと同じ）
IPO_STOP, IPO_TP = 0.10, 0.30
NH_MAX = 12            # 新高値の同時保有数（1回 ＝ 新高値の枠 ÷ 12）
NH_STOP, NH_TP = 0.06, None          # 新高値：−6%損切り・利確なし
NH_ARM, NH_TRAIL = 0.20, 0.20        # 終値で+20%に届いたら、終値の最高値−20%（最低でも買値）で追いかける（2026-11-27：+25%・−15%から変更）
IR_OM_MIN = 0.046   # IR：営業利益率（直近の通期）4.6%未満は買わない（2026-11-27 外部要因の再チェックで追加）
MKT_MRATIO_MIN = 3.67   # 75日線の買い：市場全体の信用倍率（週次）3.67倍未満の時期は買わない（同上）
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


def yf_df(code: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        d = yf.download(f"{code}.T", period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d.dropna()
    except Exception as e:
        print(f"  {code} の株価を取得できませんでした（{e}）")
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


def ir_candidates(master: pd.DataFrame, prices: dict, fins: dict, ledger: dict) -> list[dict]:
    """IRトレード：今日が「発表を受けた最初の日」の良いニュースで、強く上がって引けた銘柄（10年の検証と同じ条件）。"""
    import ir_rules
    import short_rel_pit as sp
    tpx = topix()
    out = []
    names = dict(zip(master["コード"], master["銘柄名"]))
    recent = {}
    for s_ in ledger.get("signals", []):
        if s_["side"] == "ir":
            recent[s_["code"]] = max(recent.get(s_["code"], ""), s_["date"])
    for code, recs in fins.items():
        df = prices.get(code)
        if df is None or len(df) < 60 or not recs or recs[-1]["DiscDate"] < (date.today() - timedelta(days=10)).isoformat():
            continue
        d_ = pd.to_datetime(df.index).tz_localize(None).normalize()
        dates = [x.strftime("%Y-%m-%d") for x in d_]
        today_ = dates[-1]
        c, o, h_, l_, v = (df[x].to_numpy(float) for x in ("Close", "Open", "High", "Low", "Volume"))
        for ev in ir_rules.events_from_recs(recs):
            if ev["date"] < dates[-6]:
                continue
            i0 = bisect.bisect_left(dates, ev["date"])
            if i0 >= len(dates):
                continue
            rr = i0 if (dates[i0] > ev["date"] or ev["time"] < "15:00") else i0 + 1
            if rr != len(dates) - 1:
                continue                                     # 今日が反応の日のものだけ
            news = ir_rules.good_news(ev)
            if not news:
                continue
            day = c[rr] / c[rr - 1] - 1
            clv = (c[rr] - l_[rr]) / (h_[rr] - l_[rr]) if h_[rr] > l_[rr] else 0
            pre20 = c[rr - 1] / c[rr - 21] - 1
            tv20 = float(np.mean(c[rr - 20:rr] * v[rr - 20:rr]))
            n_ = bisect.bisect_left([r["DiscDate"] for r in recs], today_)
            f = sp.pit_fund(recs[:n_]) if n_ else {}
            sh = None
            for r in reversed(recs[:n_]):
                if r["ShOutFY"] == r["ShOutFY"] and r["ShOutFY"]:
                    sh = r["ShOutFY"] - (r["TrShFY"] if r["TrShFY"] == r["TrShFY"] else 0)
                    break
            cap = sh * c[rr - 1] / 1e8 if sh else None
            om = f.get("om")
            up50 = None
            if tpx is not None and len(tpx) > 60:
                tp_ = tpx[tpx.index < pd.Timestamp(today_)]
                up50 = bool(tp_.iloc[-1] > tp_.tail(50).mean())
            ng = []
            if day < 0.05: ng.append(f"その日の上げ {day * 100:+.1f}%（+5%以上が条件）")
            if clv < 0.7: ng.append("高値近くで引けていない")
            if pre20 > 0.30: ng.append(f"発表前20日で {pre20 * 100:+.0f}% 上がっていた")
            if tv20 < 3e7: ng.append("売買代金が少ない")
            if cap is not None and cap > 700: ng.append(f"時価総額 {cap:,.0f}億円（700億円以下が条件）")
            if om is not None and om < IR_OM_MIN: ng.append(f"営業利益率 {om * 100:.1f}%（{IR_OM_MIN * 100:.1f}%以上が条件）")
            if up50 is False: ng.append("TOPIXが50日線より下")
            if code in recent and (pd.Timestamp(today_) - pd.Timestamp(recent[code])).days < 90:
                ng.append("90日以内に同じ銘柄でIRの合図あり")
            if ng:
                print(f"  IRの候補になりかけ {code} {news}：{'・'.join(ng)}")
                continue
            out.append({"date": today_, "side": "ir", "code": code, "name": names.get(code, ""), "close": float(c[rr]),
                        "score": round(day * 100, 1), "lot": 1.0, "news": news, "day": round(day * 100, 1),
                        "mcap": round(cap) if cap else None, "om": round(om * 100, 1) if om is not None else None,
                        "reasons": [news, f"その日 {day * 100:+.1f}%・高値近くで引け", f"発表前20日 {pre20 * 100:+.0f}%"]})
            break
    return out


def jp_bdays(start: str, end: str) -> list[str]:
    """東証の営業日（土日・祝日・年末年始を除く）。jpholiday が無ければ土日だけ除く。"""
    days = pd.bdate_range(start, end)
    try:
        import jpholiday
        days = [d for d in days if not jpholiday.is_holiday(d.date())]
    except Exception:  # noqa
        days = list(days)
    return [d.strftime("%Y-%m-%d") for d in days if not ((d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3))]


def kenri_window(today_: str) -> tuple[str, str] | None:
    """今日以降で一番近い3月・9月の (買う日, 売る日)。売る日＝権利付き最終日（月末の2営業日前）、買う日＝その10営業日前。"""
    y, m = int(today_[:4]), int(today_[5:7])
    for yy, mm in [(y, 3), (y, 9), (y + 1, 3), (y + 1, 9)]:
        days = jp_bdays(f"{yy}-{mm:02d}-01", f"{yy}-{mm:02d}-28") + jp_bdays(f"{yy}-{mm:02d}-29", (pd.Timestamp(yy, mm, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"))
        last = days[-3]                                         # 月末の営業日から数えて3つ目＝2営業日前
        before = jp_bdays((pd.Timestamp(last) - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), last)
        buy = before[-1 - KR_DAYS]
        if last >= today_:
            return buy, last
    return None


def gd_watch(prices: dict, fins: dict, upto: int = 0) -> tuple[list[dict], dict]:
    """窓あけ下げの監視リスト（前日の終値で判定）と、直近6営業日それぞれの「−5%以下で始まった数」。
    監視リスト：前日終値が25日線の−5%以下・直前20日の売買代金平均1億円以上・決算発表から3日以内でない。"""
    stm = {}
    for c_, recs in fins.items():
        stm[c_] = sorted(str(r["DiscDate"])[:10] for r in recs if "FinancialStatements" in str(r.get("DocType")))
    counts, today_hits, tomorrow = {}, [], []
    for code, df in prices.items():
        if len(df) < 60:
            continue
        idx = pd.to_datetime(df.index).tz_localize(None).normalize()
        dates = [d.strftime("%Y-%m-%d") for d in idx]
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        m25 = pd.Series(c).rolling(25).mean().to_numpy()
        tv = pd.Series(c * v).rolling(20).mean().to_numpy()
        ds = stm.get(code, [])
        n = len(c)
        def ok(t):   # t の日の朝に監視リストに入っていたか（t-1 の終値で判定）
            if t < 26 or not (c[t - 1] / m25[t - 1] - 1 <= -GD_DEV and tv[t - 1] >= GD_TV):
                return False
            j = bisect.bisect_left(ds, dates[t]) if t < n else len(ds)
            return not (j and (pd.Timestamp(dates[t] if t < n else date.today().isoformat()) - pd.Timestamp(ds[j - 1])).days < 3)
        for t in range(max(26, n - 6), n):
            if ok(t) and o[t] / c[t - 1] - 1 <= -GD_GAP:
                counts[dates[t]] = counts.get(dates[t], 0) + 1
                if t == n - 1:
                    today_hits.append({"code": code, "gap": o[t] / c[t - 1] - 1, "open": float(o[t]), "prev": float(c[t - 1])})
        # 明日の監視リスト（今日の終値で判定。決算は今日までの発表で判定）
        dev = c[-1] / m25[-1] - 1
        if dev <= -GD_DEV and tv[-1] >= GD_TV:
            j = len(ds)
            nxt = (pd.Timestamp(dates[-1]) + pd.offsets.BDay(1))
            if not (j and (nxt - pd.Timestamp(ds[j - 1])).days < 3):
                tomorrow.append({"code": code, "close": float(c[-1]), "dev": round(dev * 100, 1), "tv20": float(tv[-1])})
    return today_hits, {"counts": counts, "tomorrow": tomorrow}


def ye_window(today_: str) -> tuple[str, str] | None:
    """今日以降で一番近い年末の反発の (買う日, 売る日)。買う日＝大納会の5営業日前（12月の営業日の後ろから5番目）、売る日＝1月の5営業日目。"""
    y = int(today_[:4])
    for yy in (y - 1, y):
        dec = jp_bdays(f"{yy}-12-01", f"{yy}-12-31")
        jan = jp_bdays(f"{yy + 1}-01-01", f"{yy + 1}-01-31")
        if len(dec) >= YE_BUY and len(jan) >= YE_SELL and jan[YE_SELL - 1] >= today_:
            return dec[-YE_BUY], jan[YE_SELL - 1]
    return None


def ye_candidates(master: pd.DataFrame, prices: dict, data_date: str) -> list[dict]:
    """年末の反発：今年の最初の終値から今日まで40%以上下げた銘柄（直前20日の売買代金の平均3,000万円以上）を、下げの大きい順に20銘柄。"""
    names = dict(zip(master["コード"], master["銘柄名"]))
    out = []
    for code, df in prices.items():
        if len(df) < 30:
            continue
        idx = pd.to_datetime(df.index).tz_localize(None)
        c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
        yr = np.where(idx.year == int(data_date[:4]))[0]
        if not len(yr) or idx[-1].strftime("%Y-%m-%d") != data_date:
            continue
        ytd = c[-1] / c[yr[0]] - 1
        tv20 = float(np.mean(c[-21:-1] * v[-21:-1]))
        if ytd <= YE_YTD and tv20 >= YE_TV:
            out.append({"date": data_date, "side": "yearend", "code": code, "name": names.get(code, code), "close": float(c[-1]),
                        "ytd": round(ytd * 100, 1), "score": round(-ytd * 100, 1), "lot": 1.0, "tv20": tv20,
                        "reasons": [f"今年 {ytd * 100:+.0f}%"]})
    out.sort(key=lambda x: x["ytd"])
    return out[:YE_N]


def brk5_count(prices: dict) -> int:
    """市場の幅（リバウンドの判定用）：直近5営業日（今日を含む）に新高値ブレイクの合図が出た数。"""
    n = 0
    for code, df in prices.items():
        if len(df) < 260:
            continue
        cc, hh, vv = (df[x].to_numpy(float) for x in ("Close", "High", "Volume"))
        for j in range(len(cc) - 5, len(cc)):
            if nh_breakout(cc, hh, vv, j)[0]:
                n += 1
    return n


def rb_candidates(master: pd.DataFrame, prices: dict, tpx, info: dict) -> list[dict]:
    """リバウンド：25日線から−20%以上下・5日で−20%以上の急落・安値近くで引けた（引けの位置30%未満）・売買代金1億円以上。
    市場の条件：直近5日の新高値ブレイク2件以上、その日のTOPIXが−1%以上（全体の暴落の日は見送る）。"""
    names = dict(zip(master["コード"], master["銘柄名"]))
    t1 = None
    if tpx is not None and len(tpx) > 2:
        t1 = float(tpx.iloc[-1] / tpx.iloc[-2] - 1)
    b5 = brk5_count(prices)
    info["rb_brk5"], info["rb_t1"] = b5, t1
    out = []
    for code, df in prices.items():
        if len(df) < 60:
            continue
        c, h_, l_, v = (df[x].to_numpy(float) for x in ("Close", "High", "Low", "Volume"))
        m25 = c[-25:].mean()
        dev, drop5 = c[-1] / m25 - 1, c[-1] / c[-6] - 1
        clv = (c[-1] - l_[-1]) / (h_[-1] - l_[-1]) if h_[-1] > l_[-1] else 1
        tv20 = float(np.mean(c[-20:] * v[-20:]))
        if dev > -0.20 or drop5 > -0.20 or clv >= 0.3 or tv20 < 1e8:
            continue
        ng = []
        if b5 < 2: ng.append(f"直近5日の新高値ブレイク {b5}件（2件以上が条件）")
        if t1 is not None and t1 < -0.01: ng.append(f"今日のTOPIX {t1 * 100:+.1f}%（−1%以上が条件）")
        lim = c[-1] * (1 - RB_LIMIT)
        x = {"date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"), "side": "rebound", "code": code, "name": names.get(code, ""),
             "close": float(lim), "last_close": float(c[-1]), "limit": round(lim, 1), "score": round(-dev * 100, 1), "lot": 1.0,
             "dev": round(dev * 100, 1), "drop5": round(drop5 * 100, 1),
             "reasons": [f"25日線から {dev * 100:+.0f}%", f"5日で {drop5 * 100:+.0f}%", "安値近くで引け"]}
        if ng:
            x["mult"] = 0
            x["why_mkt"] = "・".join(ng)
        out.append(x)
    return out


def spx_above200() -> tuple[bool | None, str | None]:
    """米S&P500の前日終値が200日線より上か（日本の引け時点で分かっている分。バックテストと同じ）。"""
    s = None
    try:
        import yfinance as yf
        d = yf.download("^GSPC", period="2y", progress=False, auto_adjust=False)
        if d is not None and len(d):
            s = d["Close"].squeeze().dropna()
    except Exception as e:
        print(f"  S&P500の取得に失敗（{e}）")
    if (s is None or len(s) < 210) and os.path.exists("data/macro.csv.gz"):
        s = pd.read_csv("data/macro.csv.gz", index_col=0, parse_dates=True)["spx"].dropna()
    if s is None or len(s) < 210:
        return None, None
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    ma = s.rolling(200).mean()
    s, ma = s[s.index < pd.Timestamp(date.today())], ma[ma.index < pd.Timestamp(date.today())]
    return bool(s.iloc[-1] > ma.iloc[-1]), f"{s.index[-1]:%Y-%m-%d} 終値 {s.iloc[-1]:,.0f}／200日線 {ma.iloc[-1]:,.0f}"


def mkt_margin_ratio() -> tuple[float | None, str | None]:
    """市場全体の信用倍率（買い残÷売り残の合計、週次）。バックテストと同じく、週の日付の5日後から使う。"""
    p = "data/jq/margin_interest.csv.gz"
    if not os.path.exists(p):
        return None, None
    m = pd.read_csv(p, dtype=str, usecols=["Date", "LongVol", "ShrtVol"])
    for k in ("LongVol", "ShrtVol"):
        m[k] = pd.to_numeric(m[k], errors="coerce")
    w = m.groupby("Date")[["LongVol", "ShrtVol"]].sum()
    w = w[[pd.Timestamp(d) + pd.Timedelta(days=5) <= pd.Timestamp(date.today()) for d in w.index]]
    if not len(w) or not w.ShrtVol.iloc[-1]:
        return None, None
    return float(w.LongVol.iloc[-1] / w.ShrtVol.iloc[-1]), w.index[-1]


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
    spx_up, spx_info = spx_above200()
    mratio, mratio_date = mkt_margin_ratio()
    info.update(spx_up=spx_up, spx_info=spx_info, mratio=mratio, mratio_date=mratio_date)
    nh_why = []
    if nb30 > NB_HOT: nh_why.append(f"市場全体の新高値ブレイクが多すぎる（{NB_HOT}件超）過熱相場")
    if spx_up is False: nh_why.append(f"米S&P500が200日線より下（{spx_info}）")
    if nh_why: nb_mult = 0.0
    buy_why = f"市場全体の信用倍率 {mratio:.2f}倍（{MKT_MRATIO_MIN}倍以上が条件・{mratio_date}週）" if mratio is not None and mratio < MKT_MRATIO_MIN else ""
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
            if buy_why:
                buys[-1].update(mult=0, why_mkt=buy_why)
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
                                "score": score, "lot": 1.0, "mult": nb_mult, "why_mkt": "・".join(nh_why), "rs": round(rsv, 1), "reasons": [n_ for (n_, p_), o_ in zip(NH_ITEMS, ok) if o_],
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
        if led.get("rule_id") != RULE_ID:                 # ルールを変えても、これまでの記録（保有中の銘柄）は消さない
            led.setdefault("rule_history", []).append({"rule_id": led.get("rule_id"), "until": datetime.now().strftime("%Y-%m-%d")})
            led["rule_id"] = RULE_ID
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
        if s.get("entry_override"):                          # 実際に買った値段（手で入れたもの）：その日の始値をこの値段に置きかえて計算する
            j_ = i if s["side"] == "gapdown" else i + 1
            if j_ < len(df):
                df.iloc[j_, df.columns.get_loc("Open")] = float(s["entry_override"])
                df.iloc[j_, df.columns.get_loc("High")] = max(float(df["High"].iloc[j_]), float(s["entry_override"]))
                df.iloc[j_, df.columns.get_loc("Low")] = min(float(df["Low"].iloc[j_]), float(s["entry_override"]))
        if s["side"] == "gapdown":
            oo_, hh_, ll_, cc_ = (df[x_].to_numpy(float) for x_ in ("Open", "High", "Low", "Close"))
            entry = float(oo_[i]); tp = entry * (1 + GD_TP); sl = entry * (1 - GD_STOP)
            hit = None
            for j in range(i, min(len(dates), i + 2)):
                if j > i and oo_[j] <= sl:
                    hit = (j, oo_[j], f"損切り（−{GD_STOP * 100:.0f}%）"); break
                if ll_[j] <= sl:
                    hit = (j, sl, f"損切り（−{GD_STOP * 100:.0f}%）"); break
                if j > i and oo_[j] >= tp:
                    hit = (j, oo_[j], f"利確（+{GD_TP * 100:.0f}%）"); break
                if hh_[j] >= tp:
                    hit = (j, tp, f"利確（+{GD_TP * 100:.0f}%）"); break
                if j == i + 1:
                    hit = (j, cc_[j], "翌日の引けで売り"); break
            r["stop_px"], r["tp_px"] = round(sl, 1), round(tp, 1)
            if hit:
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[i], exit_date=dates[hit[0]], ret=round((hit[1] / entry - 1) * 100, 2), why=hit[2])
            else:
                r.update(status="保有中", entry=round(entry, 1), entry_date=dates[i], now=round(cc_[-1], 1), ret=round((cc_[-1] / entry - 1) * 100, 2),
                         order=f"利確 {tp:,.0f}円・損切り {sl:,.0f}円（返済予約）。届かなければ明日の引けで売る")
            amt = s.get("amount") or 0
            r["pnl"] = round(amt * r["ret"] / 100 - amt * COST)
            out.append(r)
            continue
        if i + 1 >= len(dates):
            r["status"] = {"rebound": f"明日の朝に指値 {s.get('limit', 0):,.0f}円で買い", "kenri": f"明日（{s.get('buy_date', '')}）の引けで買い", "yearend": f"明日（{s.get('buy_date', '')}）の引けで買い"}.get(
                s["side"], "明日の寄りで" + ("空売り" if s["side"] == "short" else "買い"))
            out.append(r)
            continue
        if s["side"] in ("newhigh", "ipo"):
            import newhigh_candidates as nh
            a = {"o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float), "l": df["Low"].to_numpy(float),
                 "c": df["Close"].to_numpy(float), "pivot": np.full(len(df), np.nan), "dates": dates}
            st_, tp_ = (NH_STOP, NH_TP) if s["side"] == "newhigh" else (IPO_STOP, IPO_TP)
            kw = dict(arm=NH_ARM, armlock=0.0, armtrail=NH_TRAIL) if s["side"] == "newhigh" else {}
            x = nh.sim(a, i, st_, tp=tp_, **kw)
            entry = a["o"][i + 1]
            sl = entry * (1 - st_)
            if kw:   # 今日までの追いかけのライン（明日の逆指値）
                cc = a["c"][i + 1:]
                if len(cc) and cc.max() >= entry * (1 + NH_ARM):
                    sl = max(entry, cc.max() * (1 - NH_TRAIL))
            r.update(entry=round(entry, 1), entry_date=dates[i + 1], stop_px=round(sl, 1),
                     tp_px=round(entry * (1 + tp_), 1) if tp_ else None, trailing=bool(kw and sl > entry * (1 - st_)))
            if x is None:
                last = a["c"][-1]
                r.update(status="保有中", now=round(last, 1), ret=round((last / entry - 1) * 100, 2))
            else:
                r.update(status="決済済み", ret=float(x["ret"]), exit_date=x["exit"],
                         why={"stop": ("追いかけのラインで売り" if kw and x["ret"] > -st_ * 100 + 0.5 else f"損切り（−{st_ * 100:.0f}%）"),
                              "tp": f"利確（+{(tp_ or 0) * 100:.0f}%）", "time": "120営業日で手じまい"}.get(x["reason"], x["reason"]))
            amt = s.get("amount") or 0
            r["pnl"] = round(amt * r["ret"] / 100 - amt * COST)
        elif s["side"] == "rebound":
            oo_, hh_, ll_, cc_ = (df[x_].to_numpy(float) for x_ in ("Open", "High", "Low", "Close"))
            L_ = s.get("limit") or s["last_close"] * (1 - RB_LIMIT)
            j0 = i + 1
            if oo_[j0] <= L_:
                entry, atopen = oo_[j0], True
            elif ll_[j0] <= L_:
                entry, atopen = L_, False
            else:
                r.update(status="約定せず", ret=0.0, pnl=0, why="指値まで下がらなかった", entry_date=dates[j0], exit_date=dates[j0])
                out.append(r)
                continue
            sl = entry * (1 - RB_STOP); hit = None
            m25_ = float(np.mean(cc_[max(0, i - 24):i + 1]))                   # 合図の日の25日線
            tp_ = max(m25_ * (1 + RB_TP_DEV), entry * 1.001)                   # 25日線−10%まで戻ったら利確（B・N・F流の戻り売り。2026-11-27 追加）
            r["tp_px"] = round(tp_, 1)
            for kk in range(RB_HOLD):
                j = j0 + kk
                if j >= len(dates):
                    break
                if kk > 0 and oo_[j] <= sl:
                    hit = (j, oo_[j], f"損切り（−{RB_STOP * 100:.0f}%・寄りで）"); break
                if (kk > 0 or atopen) and ll_[j] <= sl:
                    hit = (j, sl, f"損切り（−{RB_STOP * 100:.0f}%）"); break
                if kk > 0 and oo_[j] >= tp_:
                    hit = (j, oo_[j], "25日線−10%まで戻ったので利確（寄りで）"); break
                if (kk > 0 or atopen) and hh_[j] >= tp_:
                    hit = (j, tp_, "25日線−10%まで戻ったので利確"); break
                if kk == RB_HOLD - 1:
                    hit = (j, cc_[j], f"{RB_HOLD}日目の引けで売り")
            sh_ = int(s.get("amount", 0) / max(s.get("limit", L_), 1) // 100 * 100) if s.get("amount") else 0
            amt = sh_ * entry if sh_ else s.get("amount", 0)
            r["stop_px"] = round(sl, 1)
            if hit:
                ret_ = (hit[1] / entry - 1) * 100
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[j0], exit_date=dates[hit[0]], ret=round(ret_, 2), why=hit[2])
            else:
                left = RB_HOLD - (len(dates) - j0)
                plan = np.busday_offset(np.datetime64(dates[-1]), left, roll="forward")
                r.update(status="保有中", entry=round(entry, 1), entry_date=dates[j0], now=round(cc_[-1], 1), ret=round((cc_[-1] / entry - 1) * 100, 2),
                         order=f"損切り {sl:,.0f}円（逆指値）・利確 {tp_:,.0f}円（指値・25日線−10%）のOCO。届かなければ{str(plan)}の引けで売る")
            r["pnl"] = round(amt * r["ret"] / 100 - amt * COST)
        elif s["side"] in ("kenri", "yearend"):
            cc_ = df["Close"].to_numpy(float)
            entry = cc_[i + 1]
            sell = s.get("sell_date", "")
            js = bisect.bisect_left(dates, sell)
            if js < len(dates) and dates[js] == sell:
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[i + 1], exit_date=sell, ret=round((cc_[js] / entry - 1) * 100, 2),
                         why="権利付き最終日の引けで売り" if s["side"] == "kenri" else "1月の5営業日目の引けで売り")
            else:
                r.update(status="保有中", entry=round(entry, 1), entry_date=dates[i + 1], now=round(cc_[-1], 1), ret=round((cc_[-1] / entry - 1) * 100, 2),
                         order=f"{sell}（権利付き最終日）の引けで売る" if s["side"] == "kenri" else f"{sell}の引けで売る（別枠）")
            amt = s.get("amount") or 0
            r["pnl"] = round(amt * r["ret"] / 100 - amt * (0.001 if s["side"] == "kenri" else COST))
        elif s["side"] == "ir":
            cc_, oo_, ll_ = (df[x_].to_numpy(float) for x_ in ("Close", "Open", "Low"))
            entry = float(oo_[i + 1]); sl = entry * (1 - IR_STOP)
            left = IR_HOLD - (len(dates) - 1 - i)
            hit = None
            for j in range(i + 1, min(len(dates), i + IR_HOLD + 1)):
                if j > i + 1 and oo_[j] <= sl:
                    hit = (j, oo_[j]); break
                if ll_[j] <= sl:
                    hit = (j, sl); break
            r["stop_px"] = round(sl, 1)
            if hit:
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[i + 1], exit_date=dates[hit[0]],
                         ret=round((hit[1] / entry - 1) * 100, 2), why=f"損切り（−{IR_STOP * 100:.0f}%）")
            elif left <= 0:
                xi = i + IR_HOLD
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[i + 1], exit_date=dates[xi],
                         ret=round((cc_[xi] / entry - 1) * 100, 2), why=f"{IR_HOLD}営業日目の引けで売り")
            else:
                plan = np.busday_offset(np.datetime64(dates[-1]), left, roll="forward")
                r.update(status="保有中", entry=round(entry, 1), entry_date=dates[i + 1], now=round(cc_[-1], 1),
                         ret=round((cc_[-1] / entry - 1) * 100, 2),
                         order=f"損切り {sl:,.0f}円（逆指値）。あと{left}営業日・{str(plan)}ごろの引けで売る")
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
    rows = [r for r in out if r["side"] in ("buy", "newhigh", "ipo", "ir", "rebound", "kenri", "gapdown") and r.get("entry_date") and r["status"] != "約定せず"]
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
    SIDE = {"buy": "買い（75日線）", "newhigh": "買い（新高値）", "ipo": "買い（IPO）", "ir": "買い（IR）", "rebound": "買い（リバウンド）", "kenri": "買い（権利取り）", "yearend": "買い（年末の反発・別枠）", "gapdown": "買い（窓あけ下げ）", "short": "空売り"}
    items = []
    for r in hold:
        e = float(r.get("entry") or 0)
        ret = float(r.get("ret") or 0)
        now = e * (1 + ret / 100) if r["side"] != "short" else e * (1 - ret / 100)
        stop, tp = {"newhigh": (NH_STOP, NH_TP), "ipo": (IPO_STOP, IPO_TP), "ir": (IR_STOP, None)}.get(r["side"], (None, None))
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
    var sl=r.stop?(Math.round(px*(1-r.stop)).toLocaleString('ja-JP')+'円／'+(r.tp?Math.round(px*(1+r.tp)).toLocaleString('ja-JP')+'円':(r.side=='ir'?'利確なし（60営業日目の引けで売る）':'利確なし（+20%から追いかけ・保有中の表を見る）'))):'ページの判定に従う';
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
               today_nh: list | None = None, today_ipo: list | None = None, today_ir: list | None = None,
               today_rb: list | None = None, today_kr: list | None = None) -> None:
    today_rb = today_rb or []
    today_kr = today_kr or []
    today_nh = today_nh or []
    today_ipo = today_ipo or []
    today_ir = today_ir or []
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
    for side_, xs in (("buy", today_buy), ("newhigh", today_nh), ("ipo", today_ipo), ("ir", today_ir), ("rebound", today_rb), ("short", today_short)):
        for x in xs:
            if x.get("slot") not in (None, "枠あり", "すでに保有中"):
                subs.append(dict(x, side_=side_))
    def sub_exit(x):
        if x["side_"] == "newhigh":
            return f'損切り {x["close"] * (1 - NH_STOP):,.0f}円（買値−6%）。利確なし：終値で+20%に届いたら最高値−20%で追いかける'
        if x["side_"] == "rebound":
            return f'翌朝 {x.get("limit", 0):,.0f}円に指値。買えたら−10%に損切り、3日目の引けで売る'
        if x["side_"] == "ir":
            return f'損切り {x["close"] * (1 - IR_STOP):,.0f}円（買値−10%・買値で計算し直す）。利確なし。{IR_HOLD}営業日目の引けで売る'
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
               ("利確", lambda x: f'なし（+20%{x["close"] * 1.20:,.0f}円から追いかけ）'), ("点数の理由", lambda x: h.escape("・".join(x["reasons"]) or "—"))]
    ipo_cols = [("銘柄", link), ("上場日", lambda x: x.get("list_date", "")), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'),
                ("購入金額", lambda x: man(x.get("amount", 0))), ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
                ("条件", lambda x: f'{x["score"]}/4'), ("損切り目安", lambda x: f'{x["close"] * (1 - IPO_STOP):,.0f}円'),
                ("利確目安", lambda x: f'{x["close"] * (1 + IPO_TP):,.0f}円'), ("当てはまった条件", lambda x: h.escape("・".join(x["reasons"]) or "—"))]
    ir_cols = [("銘柄", link), ("終値", lambda x: f'{x.get("close", 0):,.1f}円'), ("購入金額", lambda x: man(x.get("amount", 0))),
               ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
               ("良いニュース", lambda x: h.escape(x.get("news", ""))), ("その日の上げ", lambda x: f'{x.get("day", 0):+.1f}%'),
               ("時価総額", lambda x: f'{x["mcap"]:,}億円' if x.get("mcap") else "—"), ("営業利益率", lambda x: f'{x["om"]}%' if x.get("om") is not None else "—"),
               ("損切り目安", lambda x: f'{x["close"] * (1 - IR_STOP):,.0f}円'), ("売り", lambda x: f"利確なし・{IR_HOLD}営業日目の引け")]
    rb_cols = [("銘柄", link), ("今日の終値", lambda x: f'{x.get("last_close", 0):,.1f}円'), ("明日の指値", lambda x: f'{x.get("limit", 0):,.0f}円'),
               ("購入金額", lambda x: man(x.get("amount", 0))), ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("枠", lambda x: h.escape(x.get("slot", ""))),
               ("25日線から", lambda x: f'{x.get("dev", 0):+.0f}%'), ("5日で", lambda x: f'{x.get("drop5", 0):+.0f}%'),
               ("損切り目安", lambda x: f'{x.get("limit", 0) * (1 - RB_STOP):,.0f}円'),
               ("利確（25日線−10%）", lambda x: f'{x.get("last_close", 0) / (1 + x.get("dev", 0) / 100) * (1 + RB_TP_DEV):,.0f}円'), ("売り", lambda x: "届かなければ3日目の引け")]
    kw_ = info.get("kenri")
    kr_line = (f"次の権利取り前：<b>{kw_[0]}の引け</b>でTOPIX連動ETF（1306）を資金の20%買い、<b>{kw_[1]}（権利付き最終日）の引け</b>で売る。"
               if kw_ else "次の権利取り前の日付を計算できませんでした。")
    if today_kr:
        k0 = today_kr[0]
        kr_line = (f"<b>明日（{k0['buy_date']}）の引けで</b>TOPIX連動ETF（1306）を {k0.get('shares', 0):,}口（約{man(k0.get('amount', 0))}）買う"
                   f"（引け成行）。{k0['sell_date']}（権利付き最終日）の引けで売る。枠：{h.escape(k0.get('slot', ''))}")
    yw_ = info.get("ye")
    ye_plan = info.get("ye_plan") or []
    ye_line = (f"次の年末の反発：<b>{yw_[0]}（大納会の5営業日前）の引け</b>で買い、<b>{yw_[1]}（1月の5営業日目）の引け</b>で売る。"
               if yw_ else "次の年末の反発の日付を計算できませんでした。")
    ye_cols = [("銘柄", link), ("今日の終値", lambda x: f'{x.get("close", 0):,.1f}円'), ("今年の下げ", lambda x: f'{x.get("ytd", 0):+.1f}%'),
               ("株数（50万円分）", lambda x: f'{int(YE_CAP / YE_N // x["close"] // 100 * 100):,}株' if x.get("close") else "—"),
               ("金額", lambda x: man(int(YE_CAP / YE_N // x["close"] // 100 * 100) * x["close"]) if x.get("close") else "—")]
    ye_tbl = table(ye_plan, ye_cols) if ye_plan else '<p class="sub">候補は12月に入ると毎日ここに出ます（買う日の前日の候補で確定）。</p>'
    sy_ = stats(rows, "yearend")
    gi = info.get("gd") or {}
    cnt_txt = "・".join(f"{d_[5:]} {n_}銘柄" for d_, n_ in (gi.get("counts") or {}).items()) or "—"
    if gi.get("active_tomorrow"):
        gd_head = (f'<p class="warn">直近5営業日に「急落の朝」がありました（{"・".join(gi.get("crash", []))}）。<b>明日の朝も大きく下げて始まりそうなら、窓あけ下げを実行</b>：'
                   f'8:55ごろ気配を見て、下の一覧のうち「寄りの気配が−5%の値段以下」の銘柄を、下げの大きい順に10銘柄まで寄りで買う（「−5%の値段」で寄付の指値＝寄指にすると、ロックされる資金が少なくて済む）。'
                   f'注文の時に返済予約で「利確 +{GD_TP * 100:.0f}%の円幅・損切り −{GD_STOP * 100:.0f}%の円幅」を付ける。届かなければ翌日の引けで売る。'
                   f'対象期間中は毎朝この形（その朝が急落でなくても、−5%以下で始まった銘柄があれば買う）。</p>')
        tw = sorted(gi.get("tomorrow") or [], key=lambda z: -z["tv20"])[:60]
        unit_ = info.get("equity", CAPITAL) * UNIT_GD
        gd_cols = [("銘柄", link), ("今日の終値", lambda x: f'{x["close"]:,.1f}円'), ("寄りがこれ以下なら買う", lambda x: f'{x["close"] * (1 - GD_GAP):,.0f}円'),
                   ("25日線から", lambda x: f'{x["dev"]:+.0f}%'), ("株数の目安", lambda x: f'{int(unit_ // (x["close"] * (1 - GD_GAP)) // 100 * 100):,}株'),
                   ("利確の円幅", lambda x: f'+{x["close"] * (1 - GD_GAP) * GD_TP:,.0f}円'), ("損切りの円幅", lambda x: f'−{x["close"] * (1 - GD_GAP) * GD_STOP:,.0f}円')]
        gd_tbl = table([dict(x, name=x.get("name", x["code"])) for x in tw], gd_cols) + (f'<p class="sub">売買代金の多い順に60銘柄まで表示（全部で{len(gi.get("tomorrow") or [])}銘柄）。円幅は「−5%の値段」で買えた場合の目安。</p>')
    else:
        gd_head = '<p>今は見送り（直近5営業日に「急落の朝」がない）。急落の朝が来ると、その翌日から5営業日だけ、ここに明日の監視リストが出ます。</p>'
        gd_tbl = ""
    hold = [r for r in rows if r["status"] == "保有中"]
    closed = sorted([r for r in rows if r["status"] == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    SIDE = {"buy": "買い（75日線）", "newhigh": "買い（新高値）", "ipo": "買い（IPO）", "ir": "買い（IR）", "rebound": "買い（リバウンド）", "kenri": "買い（権利取り）", "yearend": "買い（年末の反発・別枠）", "gapdown": "買い（窓あけ下げ）", "short": "空売り"}
    pos_cols = [("売買", lambda x: SIDE.get(x["side"], x["side"])), ("銘柄", link), ("合図", lambda x: x["date"]),
                ("金額", lambda x: man(x["amount"]) if x.get("amount") else lot(x["lot"])), ("株数", lambda x: f'{x["shares"]:,}株' if x.get("shares") else "—"),
                ("建値", lambda x: x.get("entry")), ("損益", lambda x: f'{x["ret"]:+.2f}%（{yen(x["pnl"])}）')]
    hold_cols = pos_cols + [("明日の注文（損切り／利確）", lambda x: (f'{"追いかけ" if x.get("trailing") else "損切り"} {x["stop_px"]:,.0f}円' + (f'／利確 {x["tp_px"]:,.0f}円' if x.get("tp_px") else "（利確なし）")) if x.get("stop_px") and x["side"] not in ("ir", "rebound")
                             else h.escape(x.get("order") or "毎日の判定に従う"))]
    cl_cols = pos_cols + [("決済", lambda x: x.get("exit_date") or ""), ("理由", lambda x: h.escape(x.get("why") or ""))]
    sb_, ss_, sn_, si_ = stats(rows, "buy"), stats(rows, "short"), stats(rows, "newhigh"), stats(rows, "ipo")
    sr_ = stats(rows, "ir")
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
<a href="./" class="sub">← スクリーナーに戻る</a>　<a href="docs.html" class="sub">📘 まとめ資料（PDF）</a>　<a href="houjin.html" class="sub">🏢 法人の売買</a>　<a href="mini.html" class="sub">🪙 ミニ株の売買</a>
<h1>今日の売買（8つの手法：75日線の買い・新高値・IPO・IR・リバウンド・窓あけ下げ・権利取り前・空売り＋別枠の年末の反発）</h1>
<div class="card"><div class="stats">
<div class="stat"><div class="k">今の資金（最初 {CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{info.get('equity', CAPITAL) / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">規模（元のルールの1,300万円に対して）</div><div class="v">{info.get('equity', CAPITAL) / BASE * 100:.0f}%</div></div>
<div class="stat"><div class="k">1回の上限</div><div class="v">{CAP_PER / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">現物の枠（7つで共有）</div><div class="v">{info.get('equity', CAPITAL) / 1e4:,.0f}万円まで</div></div>
<div class="stat"><div class="k">1回の金額：新高値／IPO</div><div class="v">{info.get('equity', CAPITAL) * UNIT_NH / 1e4:,.0f}万円／{info.get('equity', CAPITAL) * UNIT_IPO / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">空売りの枠（信用の建玉）</div><div class="v">10銘柄・{info.get('equity', CAPITAL) / BASE * 1000:,.0f}万円まで</div></div></div>
<div class="sub" style="margin-top:6px">現物の7つは1つの枠を共有し、同じ日に枠が足りないときは窓あけ下げ→権利取り前→リバウンド→IPO→新高値→IR→75日線の買いの順。同時に持てる数は75日線の買い10・新高値12・IPO10・IR20・リバウンド10・窓あけ下げ10銘柄まで。年末の反発は別枠（1,000万円）。
75日線の買い ＝ 点数で決まる金額 × 0.3 × 資金の規模。新高値 ＝ 資金の4%。IPO・IR・リバウンド・窓あけ下げ ＝ 資金の5%。権利取り前 ＝ 資金の20%。空売り ＝ 点数で決まる金額 × 今の資金の規模。1回{CAP_PER / 1e4:,.0f}万円まで。株数は購入金額 ÷ 終値を100株単位で切り捨て。
明日の始値で買う（空売りする）ので、実際の金額は少し変わります。損益は往復{COST * 100:.1f}%の費用を引いた額。</div></div>
<div class="sub">{h.escape(data_date)} 終値時点。記録開始 {h.escape(ledger['start'])}。信用残 {h.escape(str(info.get('margin_date')))}・海外投資家の直近4週 {frgn_txt}（{h.escape(str(info.get('frgn_date')))}までの週）・信用取引の注意 {info.get('alert_n', 0)}銘柄・決算データ {h.escape(str(info.get('fins_last')))} まで。</div>
{warn}
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補（{len(today_buy)}件）</h2>{table(today_buy, today_cols)}
<p class="sub">市場全体の信用倍率：<b>{"—" if info.get("mratio") is None else f"{info['mratio']:.2f}倍"}</b>（{h.escape(str(info.get("mratio_date") or "—"))}週）。{MKT_MRATIO_MIN}倍未満の時期は75日線の買いを見送り。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・新高値ブレイク（{len(today_nh)}件）</h2>{table(today_nh, nh_cols)}
<p class="sub">市場全体の新高値ブレイクの件数（30〜6日前に買った分）：<b>{info.get('nb30', '—')}件・{h.escape(str(info.get('nb_state', '—')))}</b>
（{NB_HOT}件超は過熱なので見送り）。米S&P500と200日線：<b>{h.escape(str(info.get('spx_info') or '—'))}・{'上（買える）' if info.get('spx_up') else ('下（見送り）' if info.get('spx_up') is False else '—')}</b>（200日線より下の時は新高値を買わない）。購入金額はこれを反映済み。<br>損切り・利確の目安は今日の終値から計算。実際は明日の始値（買値）の−6%に損切りの逆指値。利確はしない：終値で+20%に届いたら、毎日「終値の最高値の−20%（最低でも買値）」まで逆指値を引き上げる（値段は保有中の表）。120営業日で手じまい。業種の空売り比率 {h.escape(str(info.get('sr_date')))}・空売り残高報告 {h.escape(str(info.get('ss_date')))} まで。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・IPOセカンダリー（{len(today_ipo)}件）</h2>{table(today_ipo, ipo_cols)}
<p class="sub">直近1年のIPO {info.get('ipo_n', '—')}銘柄から判定。上場来高値を初めて抜け、20日平均の売買代金1.43億円以上、4つの条件のうち2つ以上、信用取引の注意なし、60日の上昇58%未満。
明日の始値（買値）の−10%で損切り・+30%で利確（OCO注文）。120営業日で手じまい。</p></div>
<div class="card"><h2 style="margin-top:0">明日の寄りで買う候補・IRトレード（{len(today_ir)}件）</h2>{table(today_ir, ir_cols)}
<p class="sub">今日が「発表を受けた最初の日」の決算・業績予想の修正のうち、良いニュース（上方修正+15%以上／決算で今期予想を+5%以上引き上げ／本決算の来期予想が前期実績の+20%以上）で、
今日+5%以上上がって高値近くで引けたもの。TOPIXが50日線より上・発表前20日の上昇30%以下・時価総額700億円以下・営業利益率（直近の通期）4.6%以上・売買代金3,000万円以上・同じ銘柄は90日に1回。
明日の寄りで買い、買値の−10%に損切りの逆指値を入れる。当たらなければ{IR_HOLD}営業日目の引けで売る（利確はしない）。1回は資金の5%、同時に20銘柄まで。</p></div>
<div class="card"><h2 style="margin-top:0">明日の朝に指値で買う候補・リバウンド（{len(today_rb)}件）</h2>{table(today_rb, rb_cols)}
<p class="sub">25日線から−20%以上下・5日で−20%以上の急落・安値近くで引けた銘柄（売買代金1億円以上）。直近5日に新高値ブレイクが2件以上あり、その日のTOPIXが−1%以上の時だけ
（今日：新高値ブレイク {info.get("rb_brk5", "—")}件・TOPIX {"—" if info.get("rb_t1") is None else f'{info["rb_t1"] * 100:+.1f}%'}）。
明日の朝、今日の終値の−3%に指値で買う（届かなければ買わない）。買えたら−10%に損切りの逆指値と、今日の25日線の−10%の値段に利確の指値（OCO）。どちらにも届かなければ、買った日を1日目として3日目の引けで売る。1回は資金の5%、同時に10銘柄まで。</p></div>
<div class="card"><h2 style="margin-top:0">権利取り前（3月・9月）</h2><p>{kr_line}</p>
<p class="sub">3月・9月の権利付き最終日の前は、配当目当ての買いで市場全体が上がりやすい（10年で20回中15回プラス）。資金の20%で、年2回・約2週間だけ持つ。</p></div>
<div class="card"><h2 style="margin-top:0">窓あけ下げ（急落の朝のあと5営業日・寄りで買う）</h2>{gd_head}{gd_tbl}
<p class="sub">監視リスト：前日終値が25日線の−5%以下・売買代金1億円以上・決算発表から3日以内でない銘柄。この中で−5%以下で始まった数（直近）：{h.escape(cnt_txt)}。
{GD_CRASH}銘柄以上の朝が「急落の朝」。暴落の初日は負けやすいので見送り、そのあと5営業日のあいだだけ、−5%以下で始まった銘柄を下げの大きい順に買う。1回は資金の{UNIT_GD * 100:.0f}%、10銘柄まで。
外部要因41種類も確認済み（追加で効く条件はなし）。10年で258回・勝率70%・+462万円（1,000万円の場合）。</p></div>
<div class="card"><h2 style="margin-top:0">年末の反発（別枠1,000万円・12月〜1月）</h2><p>{ye_line}</p>{ye_tbl}
<p class="sub">その年に40%以上下げた銘柄（直前20日の売買代金の平均3,000万円以上）を、下げの大きい順に20銘柄・1銘柄50万円ずつ。年末の節税売りが終わったあとの反発をねらう。
損切り・利確なしで、決まった日の引けで売るだけ。10年（10回）で+415万円・負けた年1回（1銘柄ごとの勝率 約8割）。この枠の損益は、上の「今の資金」には入れない。
別枠の成績：合図 {sy_['n']}件・決済済み {sy_['closed']}件・損益 {yen(sy_['pnl_closed'] + sy_['pnl_open'])}。</p></div>
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
<div class="stat"><div class="k">IR：合図／決済済み</div><div class="v">{sr_['n']}／{sr_['closed']}件</div></div>
<div class="stat"><div class="k">IR：決済済みの損益（勝率）</div><div class="v">{yen(sr_['pnl_closed'])}（{sr_['win'] if sr_['win'] is not None else '—'}%）</div></div>
<div class="stat"><div class="k">IR：保有中の含み損益</div><div class="v">{yen(sr_['pnl_open'])}</div></div>
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
    return CAPITAL + sum(r["pnl"] for r in rows if r["status"] == "決済済み" and r["side"] != "yearend")   # 年末の反発は別枠


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


SIDE_NAME = {"buy": "75日線の買い", "newhigh": "新高値", "ipo": "IPO", "ir": "IRトレード", "rebound": "リバウンド", "kenri": "権利取り前", "yearend": "年末の反発", "gapdown": "窓あけ下げ", "short": "空売り"}


def size_long(cands: list[dict], rows: list[dict], eq: float) -> None:
    """現物の7つを1つの枠（今の資金）で共有。cands はすでに枠を使う順に並んでいること。"""
    open_ = [r for r in rows if r["side"] in ("buy", "newhigh", "ipo", "ir", "rebound", "kenri", "gapdown") and (r["status"] == "保有中" or r["status"].startswith("明日"))]
    used = sum(r.get("amount") or 0 for r in open_)
    held = {r["code"] for r in open_}
    n = {k: sum(1 for r in open_ if r["side"] == k) for k in MAX_N}
    for x in cands:
        if x["side"] == "buy":
            want = x["lot"] * 1e6 * UNIT_BUY * eq / 1e7
        elif x["side"] == "newhigh":
            want = eq * UNIT_NH * x.get("mult", 1.0)
        elif x["side"] == "ir":
            want = eq * UNIT_IR
        elif x["side"] == "rebound":
            want = eq * UNIT_RB
        elif x["side"] == "kenri":
            want = eq * UNIT_KR
        elif x["side"] == "gapdown":
            want = eq * UNIT_GD
        else:
            want = eq * UNIT_IPO
        want = min(want, CAP_PER)
        sh = int(want // x["close"] // 100 * 100) if x.get("close") else 0
        x["shares"], x["amount"], x["cap"], x["want"] = sh, round(sh * x["close"]) if sh else 0, eq, want
        if x.get("mult") == 0:
            x["slot"] = "見送り（相場の条件）"
            x["why"] = x.get("why_mkt", "") or "相場の条件を満たさない"
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
    irs = []
    try:
        irs = ir_candidates(master, prices, fins_recent(), ledger)
    except Exception as e:
        print(f"  IRトレードの判定に失敗（{e}）")
    # リバウンド・権利取り前（2026-11-27 追加）
    rbs, krs = [], []
    tpx = topix()
    data_date = max(pd.Timestamp(d.index[-1]) for d in prices.values()).strftime("%Y-%m-%d")
    try:
        rbs = rb_candidates(master, prices, tpx, info)
    except Exception as e:
        print(f"  リバウンドの判定に失敗（{e}）")
    try:
        win = kenri_window(data_date)
        info["kenri"] = win
        nxt = jp_bdays((pd.Timestamp(data_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (pd.Timestamp(data_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
        if win and nxt and nxt[0] == win[0] and tpx is not None:
            krs = [{"date": data_date, "side": "kenri", "code": "1306", "name": "TOPIX連動ETF（1306）", "close": float(tpx.iloc[-1]), "score": 0, "lot": 1.0,
                    "buy_date": win[0], "sell_date": win[1], "reasons": [f"{win[1]}が権利付き最終日"]}]
        if tpx is not None:
            et = yf_df("1306")
            if et is not None:
                allp["1306"] = et
    except Exception as e:
        print(f"  権利取り前の判定に失敗（{e}）")
    gds = []
    try:
        hits, gw = gd_watch(prices, fins_recent())
        cnt = gw["counts"]
        nm_ = dict(zip(master["コード"], master["銘柄名"]))
        for x in gw["tomorrow"]:
            x["name"] = nm_.get(x["code"], x["code"])
        last6 = [d_ for d_ in sorted({x.strftime("%Y-%m-%d") for d_ in prices.values() for x in pd.to_datetime(d_.index[-6:])})][-6:]
        crash = [d_ for d_ in last6 if cnt.get(d_, 0) >= GD_CRASH]
        today_crash = cnt.get(data_date, 0) >= GD_CRASH
        prev_crash = [d_ for d_ in last6[:-1] if cnt.get(d_, 0) >= GD_CRASH]
        info["gd"] = {"counts": {d_: cnt.get(d_, 0) for d_ in last6}, "crash": crash, "today": today_crash,
                      "active_tomorrow": bool([d_ for d_ in last6[-5:] if cnt.get(d_, 0) >= GD_CRASH]), "tomorrow": gw["tomorrow"]}
        if prev_crash:                                                     # 直前5営業日に急落の朝があった（対象期間）：−5%以下で始まった銘柄を寄りで買ったことにして記録（11/27：今朝の数の確認はやめた）
            names_ = dict(zip(master["コード"], master["銘柄名"]))
            for x in sorted(hits, key=lambda z: z["gap"]):
                gds.append({"date": data_date, "side": "gapdown", "code": x["code"], "name": names_.get(x["code"], x["code"]), "close": x["open"],
                            "gap": round(x["gap"] * 100, 1), "score": round(-x["gap"] * 100, 1), "lot": 1.0, "reasons": [f"寄り {x['gap'] * 100:+.1f}%"]})
    except Exception as e:
        print(f"  窓あけ下げの判定に失敗（{e}）")
    yes, ye_plan = [], []
    try:
        yw = ye_window(data_date)
        info["ye"] = yw
        if yw and data_date[5:7] == "12" and data_date < yw[0]:
            ye_plan = ye_candidates(master, prices, data_date)             # 12月：今の時点の候補（予定）
            nxt = jp_bdays((pd.Timestamp(data_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (pd.Timestamp(data_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
            if nxt and nxt[0] == yw[0]:                                    # 明日が買う日：台帳に入れる
                for x in ye_plan:
                    want = YE_CAP / YE_N
                    sh = int(want // x["close"] // 100 * 100)
                    x.update(buy_date=yw[0], sell_date=yw[1], want=want, shares=sh, amount=round(sh * x["close"]),
                             slot="枠あり" if sh else "100株に届かない",
                             why="" if sh else f'1銘柄 {want / 1e4:,.0f}万円に対して、100株で {x["close"] * 100 / 1e4:,.1f}万円かかる')
                yes = ye_plan
    except Exception as e:
        print(f"  年末の反発の判定に失敗（{e}）")
    prev = evaluate(ledger, allp)
    eq = equity(prev)
    buys.sort(key=lambda x: -(x.get("eg") if x.get("eg") is not None else -9))   # バックテストと同じ：同じ日は増益率の高い順
    shorts.sort(key=lambda x: (-x["score"], x["code"]))
    size(shorts, prev, eq, "short")
    nhs.sort(key=lambda x: (-x["score"], -x.get("rs", 0)))               # バックテストと同じ：点数の高い順→相対的な強さ
    ipos.sort(key=lambda x: -x["score"])
    irs.sort(key=lambda x: -x["score"])                                  # バックテストと同じ：その日の上げが大きい順
    rbs.sort(key=lambda x: -x["score"])                                  # バックテストと同じ：25日線から離れている順
    size_long(gds + krs + rbs + ipos + nhs + irs + buys, prev, eq)            # 枠を使う順：権利取り→リバウンド→IPO→新高値→IR→75日線の買い
    info["equity"] = eq
    have = {(s["date"], s["side"], s["code"]) for s in ledger["signals"]}
    for s in buys + shorts + nhs + ipos + irs + rbs + krs + yes + gds:
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
    info["ye_plan"] = ye_plan
    build_page(ledger, rows, buys, shorts, info, data_date, os.path.join(out_dir, "forward2.html"), nhs, ipos, irs, rbs, krs)
    try:                                                   # 法人用のページ（少ない手間版：IR・IPO・新高値・権利取り前・年末の反発）
        import corp_page
        corp_page.run_corp(allp, {"ir": irs, "ipo": ipos, "newhigh": nhs, "kenri": krs, "yearend": yes}, info, data_date, out_dir)
    except Exception as e:
        print(f"  法人ページの作成に失敗（{e}）")
    try:                                                   # ミニ株版のページ（注文は寄り付きだけ：逆指値なし・空売りなし）
        import mini_page
        mini_page.run_mini(allp, {"buy": buys, "newhigh": nhs, "ipo": ipos, "ir": irs, "rebound": rbs, "gapdown": gds, "kenri": krs}, info, data_date, out_dir)
    except Exception as e:
        print(f"  ミニ株ページの作成に失敗（{e}）")
    print(f"フォワードテスト2: 今日の買い候補 {len(buys)}件・新高値 {len(nhs)}件・IPO {len(ipos)}件・IR {len(irs)}件・リバウンド {len(rbs)}件・権利取り {len(krs)}件・年末の反発 {len(yes)}件（予定 {len(ye_plan)}件）・窓あけ下げ {len(gds)}件（急落の朝の数 {info.get("gd", {}).get("counts")}）・空売り候補 {len(shorts)}件・台帳 {len(ledger['signals'])}件")
    for lab, xs in (("75日線", buys), ("新高値", nhs), ("IPO", ipos), ("IR", irs), ("リバウンド", rbs), ("権利取り", krs), ("年末の反発", ye_plan), ("窓あけ下げ", gds), ("空売り", shorts)):
        for x in xs:   # 候補の中身をログにも残す（ページを見られない時の確認用）
            print("  候補", lab, {kk: x.get(kk) for kk in ("code", "name", "signal", "close", "score", "slot", "shares", "yen", "amount", "lot") if kk in x})
