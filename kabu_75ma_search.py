#!/usr/bin/env python3
"""
東証プライム 75日線ブレイク候補 検索スクリプト
================================================

『世界一やさしい 株の教科書 1年生』の考え方（6ヶ月チャートで、長く下落したあとに
75日移動平均線を下から上へ突き抜ける＝上昇トレンドへの転換）を、条件として数値化して
東証プライム全銘柄から探します。

  条件（初期値。下の「設定」で変えられます）
    1. 東証プライム銘柄
    2. 出来高 50万株以上（直近の営業日）
    3. 3ヶ月〜半年 下落が続いている
         - 75日線が約3ヶ月前（60営業日前）より 2% 以上下がっている
         - 直近3ヶ月の 75% 以上の日で、終値が75日線の下にある
         - 過去6ヶ月の高値から 15% 以上下げた
    4. 75日線を下から突き抜けそう／突き抜けた直後
         - 突破目前：終値が75日線の 5% 下〜75日線の間 で、直近5日で上昇中
         - 突破直後：直近5営業日以内に終値が75日線を下から上に抜け、上抜け幅 3% 以内
    5. 買いの絞り込み（上抜けた日、突破目前は当日に判定）
         - 25日線が5日前より上（上向き）
         - 出来高が過去20日平均の1.5倍以上

使い方（Windows / Mac）
    1) Python 3.9 以上をインストール
    2) ターミナル（コマンドプロンプト）で:
         pip install yfinance pandas xlrd
    3) このファイルを保存したフォルダで:
         python kabu_75ma_search.py
    4) 数分〜十数分で結果が表示され、同じフォルダに次のファイルができます
         - 75日線ブレイク候補_YYYYMMDD.html  … 候補をチャート付きで一覧できるページ（ブラウザで開く）
         - 75日線ブレイク候補_YYYYMMDD.csv   … 候補一覧（Excelで開けます。Yahoo!ファイナンスのリンク付き）
         - スクリーナー読込用_YYYYMMDD.csv   … 候補銘柄の日足。スクリーナーの「CSVを読み込む」でチャート確認用

注意: 株価は Yahoo Finance（yfinance）から取得します。非公式の取得方法のため、
混雑時は一部の銘柄が取れないことがあります。そのときは時間をおいて再実行してください。
結果は売買のおすすめではありません。最終判断はチャートを見てご自身で行ってください。
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

# ============================== 設定 ==============================
MIN_VOLUME = 500_000          # 出来高の下限（株）
VOLUME_MODE = "last"          # "last"=直近の営業日 / "avg5"=直近5日平均 / "avg20"=直近20日平均

MA = 75                       # 基準の移動平均（日）
DOWN_DAYS = 60                # 下落を見る期間（営業日）。60≒3ヶ月、120≒半年
MA_DECLINE_MIN = 0.02         # この期間で75日線が何％以上下がっていること
BELOW_RATIO_MIN = 0.75        # この期間で終値が75日線の下にあった日の割合
DROP_LOOKBACK = 120           # 高値からの下落率を見る期間（営業日）≒半年
DROP_MIN = 0.15               # その期間の高値から何％以上下げたか

APPROACH_GAP = 0.05           # 「突破目前」：75日線の何％下まで対象にするか
CROSS_WITHIN = 5              # 「突破直後」：何営業日以内の上抜けを対象にするか
CROSS_GAP_MAX = 0.03          # 「突破直後」：75日線からの上抜け幅の上限
RISING_DAYS = 5               # 「突破目前」：何日前より終値が上がっていること

# 買いの絞り込み（バックテストで成績が良くなった条件）。上抜けた日（突破目前は当日）に判定する
USE_FILTER = True
FILTER_MA = 25                # この移動平均が FILTER_RISE_DAYS 日前より上（上向き）
FILTER_RISE_DAYS = 5
FILTER_VOL_X = 1.5            # 出来高が過去20日平均の何倍以上か

JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_LIST_URL = ("https://www.jpx.co.jp/markets/statistics-equities/misc/"
                "tvdivq0000001vg2-att/data_j.xls")   # 一覧ページから見つからなかったときの予備
PRIME_LABEL = "プライム（内国株式）"
CHUNK = 150                   # 一度にダウンロードする銘柄数
# ==================================================================


def load_prime_list() -> pd.DataFrame:
    """JPXの上場銘柄一覧からプライム（内国株式）を取り出す。"""
    import re
    from urllib.parse import urljoin
    print("東証の上場銘柄一覧を取得しています…")
    ua = {"User-Agent": "Mozilla/5.0"}
    urls = []
    try:   # 一覧ページに載っている最新のExcelファイルの場所を探す（場所は時々変わる）
        with urllib.request.urlopen(urllib.request.Request(JPX_LIST_PAGE, headers=ua), timeout=60) as r:
            page = r.read().decode("utf-8", "ignore")
        links = re.findall(r'href="([^"]+\.xlsx?)"', page)
        links.sort(key=lambda u: "data_j" not in u)
        urls += [urljoin(JPX_LIST_PAGE, u) for u in links]
    except Exception as e:
        print(f"  一覧ページを読めませんでした（{e}）")
    urls.append(JPX_LIST_URL)
    raw = None
    for u in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60) as r:
                raw = r.read()
            print(f"  取得元: {u}")
            break
        except Exception as e:
            print(f"  取得できませんでした: {u}（{e}）")
    if raw is None:
        sys.exit("東証の上場銘柄一覧を取得できませんでした。")
    df = pd.read_excel(io.BytesIO(raw), dtype=str)
    df = df[df["市場・商品区分"] == PRIME_LABEL]
    out = pd.DataFrame({
        "コード": df["コード"].str.strip(),
        "銘柄名": df["銘柄名"].str.strip(),
        "業種": df["33業種区分"].str.strip(),
    }).reset_index(drop=True)
    print(f"  プライム銘柄: {len(out)}")
    if os.environ.get("K_DELISTED") == "1":
        out = pd.concat([out, delisted_list()], ignore_index=True).drop_duplicates("コード")
        print(f"  上場廃止銘柄を加えて: {len(out)}")
    return out


DELISTED_DIR = "data/jq"


def delisted_list() -> pd.DataFrame:
    """東証一部・プライムだったことがあり、今は上場していない銘柄（J-Quants の過去の上場銘柄一覧から）。
    バックテストで「今ある銘柄だけ」で検証する偏りを直すために使う（環境変数 K_DELISTED=1 のとき）。"""
    h = pd.read_csv(os.path.join(DELISTED_DIR, "master_history.csv.gz"), dtype=str)
    now = set(pd.read_csv(os.path.join(DELISTED_DIR, "master.csv"), dtype=str)["Code"])
    big = h[h["MktNm"].fillna("").str.contains("一部|プライム")]
    gone = big[~big["Code"].isin(now)].sort_values("Date").groupby("Code").last().reset_index()
    return pd.DataFrame({"コード": gone["Code"].str[:4], "銘柄名": gone["CoName"], "業種": gone["S33Nm"]})


def delisted_prices(codes: list[str], period: str) -> dict[str, pd.DataFrame]:
    """上場廃止銘柄の日足（J-Quants の分割調整済みの値）を、yfinance と同じ形にして返す。"""
    p = os.path.join(DELISTED_DIR, "delisted_bars.csv.gz")
    if not os.path.exists(p):
        return {}
    b = pd.read_csv(p, dtype={"Code": str})
    b["c4"] = b["Code"].str[:4]
    b = b[b["c4"].isin(set(codes))]
    years = int("".join(ch for ch in period if ch.isdigit()) or 1)
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    out = {}
    for c, g in b.groupby("c4"):
        if g["Code"].nunique() > 1:      # 同じ4桁に複数のコードがあるときは、データの多い方を使う
            g = g[g["Code"] == g["Code"].value_counts().idxmax()]
        df = pd.DataFrame({"Open": g["AdjO"].values, "High": g["AdjH"].values, "Low": g["AdjL"].values,
                           "Close": g["AdjC"].values, "Volume": g["AdjVo"].values},
                          index=pd.DatetimeIndex(pd.to_datetime(g["Date"].values)))
        df = df.sort_index().dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")]
        df = df[(df.index >= since) & (df["Close"] > 0)]
        if len(df):
            out[c] = df
    return out


def download_prices(codes: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """yfinanceで約1年分の日足を取得する。"""
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    total = len(codes)
    for i in range(0, total, CHUNK):
        part = codes[i:i + CHUNK]
        tickers = [f"{c}.T" for c in part]
        print(f"  株価を取得中 {i + 1}〜{min(i + CHUNK, total)} / {total}")
        for attempt in range(3):
            try:
                data = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                                   auto_adjust=os.environ.get("K_ADJUST") == "1", threads=True, progress=False)
                break
            except Exception as e:  # 一時的な失敗は少し待って再試行
                print(f"    取得に失敗（{e}）。再試行します…")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for c, t in zip(part, tickers):
            try:
                df = data[t] if len(tickers) > 1 else data
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df):
                    result[c] = df
            except KeyError:
                pass
        time.sleep(1)
    if os.environ.get("K_DELISTED") == "1":
        miss = [c for c in codes if c not in result]
        add = delisted_prices(miss, period)
        result.update(add)
        print(f"  上場廃止銘柄の株価をJ-Quantsのデータで補いました: {len(add)}銘柄")
    # 取引時間中（15:45より前）に実行したときは、まだ確定していない当日分を使わない
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Tokyo"))
        if now.hour * 60 + now.minute < 15 * 60 + 45:
            today = pd.Timestamp(now.date())
            cut = 0
            for c, df in list(result.items()):
                if len(df) and df.index[-1].normalize() == today:
                    result[c] = df.iloc[:-1]
                    cut += 1
            if cut:
                print(f"  取引時間中のため、当日の途中データを除きました（{cut}銘柄）")
    except Exception:
        pass
    print(f"  取得できた銘柄: {len(result)} / {total}")
    return result


def screen(df: pd.DataFrame) -> dict | None:
    """1銘柄を判定する。条件を満たせば情報を、満たさなければ None を返す。"""
    need = max(MA + DOWN_DAYS, DROP_LOOKBACK) + 5
    if len(df) < need:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma = c.rolling(MA).mean()
    L = len(df) - 1

    # 2. 出来高
    vol = {"last": v.iloc[L], "avg5": v.iloc[-5:].mean(), "avg20": v.iloc[-20:].mean()}[VOLUME_MODE]
    if vol < MIN_VOLUME:
        return None

    # 3. 3ヶ月〜半年の下落
    ma_then, ma_now = ma.iloc[L - DOWN_DAYS], ma.iloc[L]
    ma_decline = ma_then / ma_now - 1
    if ma_decline < MA_DECLINE_MIN:
        return None
    win = slice(L - DOWN_DAYS, L - CROSS_WITHIN)       # 直近の上抜け部分は除いて数える
    below_ratio = (c.iloc[win] < ma.iloc[win]).mean()
    if below_ratio < BELOW_RATIO_MIN:
        return None
    hi = h.iloc[L - DROP_LOOKBACK:L + 1].max()
    lo = l.iloc[L - DROP_LOOKBACK:L + 1].min()
    drop = 1 - lo / hi
    if drop < DROP_MIN:
        return None

    # 4. 75日線への接近・突破
    gap = c.iloc[L] / ma_now - 1
    crossed_days_ago = None
    for k in range(CROSS_WITHIN):
        i = L - k
        if c.iloc[i - 1] <= ma.iloc[i - 1] and c.iloc[i] > ma.iloc[i]:
            crossed_days_ago = k
            break
    if crossed_days_ago is not None and 0 < gap <= CROSS_GAP_MAX:
        status = "突破直後" if crossed_days_ago > 0 else "本日突破"
    elif -APPROACH_GAP <= gap <= 0 and c.iloc[L] > c.iloc[L - RISING_DAYS]:
        status = "突破目前"
    else:
        return None

    # 買いの絞り込み：25日線が上向き かつ 出来高が過去20日平均の1.5倍以上
    ref = L - crossed_days_ago if crossed_days_ago is not None and status != "突破目前" else L
    ma_f = c.rolling(FILTER_MA).mean()
    vol_x = v.iloc[ref] / v.iloc[ref - 20:ref].mean() if v.iloc[ref - 20:ref].mean() > 0 else 0
    ma_up = ma_f.iloc[ref] > ma_f.iloc[ref - FILTER_RISE_DAYS]
    if USE_FILTER and not (ma_up and vol_x >= FILTER_VOL_X):
        return None

    return {
        "状態": status,
        "株価": round(float(c.iloc[L]), 1),
        "75日線": round(float(ma_now), 1),
        "75日線との差(%)": round(gap * 100, 2),
        "上抜け(何日前)": "" if crossed_days_ago is None else crossed_days_ago,
        "5日騰落(%)": round((c.iloc[L] / c.iloc[L - RISING_DAYS] - 1) * 100, 2),
        "75日線の3ヶ月下落(%)": round(-ma_decline / (1 + ma_decline) * 100, 2),
        "75日線より下の日の割合(%)": round(below_ratio * 100, 1),
        "半年高値からの下落(%)": round(-drop * 100, 1),
        "出来高": int(vol),
        "判定日の出来高倍率": round(float(vol_x), 2),
        "日付": df.index[L].strftime("%Y-%m-%d"),
    }


FIN_SECTORS = {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}


def _fund_metrics(f: dict, close: float) -> dict:
    """スクリーナー画面の fund() と同じ計算（終値は0.1円単位）。"""
    c = float(np.round(close * 10) / 10)
    pos = lambda v: v is not None and v > 0
    eps, feps, bps = f.get("eps"), f.get("feps"), f.get("bps")
    div = f.get("div") if f.get("div") is not None else f.get("tdiv")
    pct = lambda key: f[key] * 100 if f.get(key) is not None else None
    return {"per": c / eps if pos(eps) else None, "perNeg": eps is not None and eps <= 0,
            "fper": c / feps if pos(feps) else None, "pbr": c / bps if pos(bps) else None,
            "dy": div / c * 100 if div is not None else None, "roe": pct("roe"), "om": pct("om"),
            "rg": pct("rg"), "eg": pct("eg"), "de": f.get("de"), "eps": eps, "feps": feps}


def fund_ranks(master: pd.DataFrame, prices: dict, fund: dict, close_map: dict | None = None) -> dict:
    """ファンダメンタル10段階ランク（スクリーナー画面の fundScore() と同じ採点）。
    close_map を渡すと、その終値（例：過去のある日の終値）で PER・PBR などを計算する。"""
    import statistics
    met, sec_of = {}, {}
    for _, m in master.iterrows():
        if close_map is not None:
            px = close_map.get(m["コード"])
            if px is None or px != px or px <= 0:
                continue
        else:
            df = prices.get(m["コード"])
            if df is None or not len(df):
                continue
            px = df["Close"].iloc[-1]
        sec_of[m["コード"]] = m["業種"]
        met[m["コード"]] = _fund_metrics(fund.get(m["コード"], {}), px)
    stats = {}
    for sec in set(sec_of.values()):
        arr = [met[c] for c in met if sec_of[c] == sec]
        def med(key, lo, hi):
            v = [x[key] for x in arr if x[key] is not None and lo < x[key] < hi]
            return statistics.median(v) if v else None
        stats[sec] = {"per": med("per", 0, 200), "pbr": med("pbr", 0, 30), "om": med("om", -100, 100)}
    out = {}
    for code, F in met.items():
        if code not in fund:
            out[code] = None
            continue
        S, sec = stats[sec_of[code]], sec_of[code]
        mx = raw = 0.0
        def add(m, p):
            nonlocal mx, raw
            mx += m
            raw += p
        if F["perNeg"]:
            add(2, 0)
        elif F["per"] is not None and S["per"] is not None:
            r = F["per"] / S["per"]
            add(2, 2 if r < 0.7 else 1.5 if r < 0.9 else 1 if r < 1.1 else 0.5 if r < 1.5 else 0)
        if F["pbr"] is not None and S["pbr"] is not None:
            add(1, 1 if (F["pbr"] < 1 or F["pbr"] < S["pbr"] * 0.8) else 0.5 if F["pbr"] < S["pbr"] * 1.2 else 0)
        if F["roe"] is not None:
            x = F["roe"]
            add(2, 2 if x >= 15 else 1.5 if x >= 10 else 1 if x >= 8 else 0.5 if x >= 5 else 0)
        if F["om"] is not None and S["om"] is not None:
            add(1, 1 if (F["om"] >= S["om"] * 1.5 and F["om"] > 0) else 0.5 if F["om"] >= S["om"] else 0)
        a, b = F["rg"], F["eg"]
        if a is not None and b is not None:
            add(2, 2 if (a >= 10 and b >= 10) else 1.5 if (a >= 5 and b >= 5) else 1 if (a > 0 and b > 0)
                else 0.5 if (a > 0 or b > 0) else 0)
        elif a is not None or b is not None:
            x = a if a is not None else b
            add(2, 1.5 if x >= 10 else 1 if x >= 5 else 0.5 if x > 0 else 0)
        if F["eps"] is not None and F["feps"] is not None and F["eps"] > 0:
            g = F["feps"] / F["eps"] - 1
            add(1, 0.5 if abs(g) >= 0.4 else 1 if g >= 0.1 else 0.5 if g >= 0 else 0)
        if F["de"] is not None and sec not in FIN_SECTORS:
            add(0.5, 0.5 if F["de"] <= 50 else 0.25 if F["de"] <= 100 else 0)
        if F["dy"] is not None:
            add(0.5, 0.5 if F["dy"] >= 3 else 0.25 if F["dy"] >= 1.5 else 0)
        out[code] = int(np.floor(raw / mx * 10 + 0.5 + 1e-9)) if mx >= 6 else None
    return out


def rank_label(sc):
    return "データ不足" if sc is None else "優良" if sc >= 8 else "良好" if sc >= 6 else "普通" if sc >= 4 else "弱め" if sc >= 2 else "要注意"


def rank_color(sc):
    return "#8a8983" if sc is None else "#1f7a3a" if sc >= 8 else "#3f8f4f" if sc >= 6 else "#8a7a2a" if sc >= 4 else "#b26b00" if sc >= 2 else "#c8312b"


def _svg_chart(df: pd.DataFrame, days: int = 120) -> str:
    """約6ヶ月の線グラフに、25日線・75日線・ボリンジャーバンド(25日,±2σ)を重ねた小さなチャート。"""
    c = df["Close"]
    ma25, ma75 = c.rolling(25).mean(), c.rolling(MA).mean()
    sd = c.rolling(25).std(ddof=0)
    bu, bd, bu1, bd1 = ma25 + 2 * sd, ma25 - 2 * sd, ma25 + sd, ma25 - sd
    cl, m25, m75, u, d, u1, d1 = (x.iloc[-days:] for x in (c, ma25, ma75, bu, bd, bu1, bd1))
    vals = [v for ser in (cl, m25, m75, u, d) for v in ser if pd.notna(v)]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    W, H, n = 340, 130, len(cl)
    X = lambda i: 4 + i * (W - 8) / max(n - 1, 1)
    Y = lambda v: 4 + (hi - v) / (hi - lo) * (H - 8)
    pts = lambda ser: " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ser) if pd.notna(v))
    up_pts = [(X(i), Y(v)) for i, v in enumerate(u) if pd.notna(v)]
    dn_pts = [(X(i), Y(v)) for i, v in enumerate(d) if pd.notna(v)]
    band = " ".join(f"{x:.1f},{y:.1f}" for x, y in up_pts + dn_pts[::-1])
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="ch">'
            f'<polygon points="{band}" fill="rgba(154,166,178,.18)"/>'
            f'<polyline points="{pts(u)}" fill="none" stroke="#d9534f" stroke-width="1"/>'
            f'<polyline points="{pts(u1)}" fill="none" stroke="#f0a35e" stroke-width="1" stroke-dasharray="3 2"/>'
            f'<polyline points="{pts(d1)}" fill="none" stroke="#5fa8d8" stroke-width="1" stroke-dasharray="3 2"/>'
            f'<polyline points="{pts(d)}" fill="none" stroke="#2f6fc4" stroke-width="1"/>'
            f'<polyline points="{pts(m25)}" fill="none" stroke="#2a8a4a" stroke-width="1.6"/>'
            f'<polyline points="{pts(m75)}" fill="none" stroke="#7a4fbf" stroke-width="1.8"/>'
            f'<polyline points="{pts(cl)}" fill="none" stroke="#1d1d1b" stroke-width="1.5"/></svg>')


def build_page(rows: list[dict], prices: dict, data_date: str, checked: int, path: str) -> None:
    """結果をスマホでも見やすいHTMLページにする。"""
    import html as h
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = []
    for r in rows:
        tone = "brk" if r["状態"] != "突破目前" else "near"
        gap = r["75日線との差(%)"]
        cards.append(f"""<div class="item"><a class="card" href="index.html?c={h.escape(r['コード'])}&amp;v=book">
  <div class="top"><span class="st {tone}">{h.escape(r['状態'])}</span><span class="code">{h.escape(r['コード'])}</span>
  <span class="nm">{h.escape(r['銘柄名'])}</span>
  <span class="fr" style="background:{rank_color(r.get('Fランク'))}">F {r.get('Fランク') if r.get('Fランク') is not None else '—'}<small>/10</small></span></div>
  <div class="sub">{h.escape(str(r['業種']))}</div>
  {_svg_chart(prices[r['コード']])}
  <div class="kv"><div><b>{r['株価']:,.1f}</b><i>株価</i></div><div><b>{gap:+.2f}%</b><i>75日線との差</i></div>
  <div><b>{r['半年高値からの下落(%)']:.1f}%</b><i>半年高値から</i></div><div><b>{r['出来高']:,}</b><i>出来高</i></div></div>
</a>
<a class="ylink" href="{h.escape(r['Yahoo!ファイナンス'])}" target="_blank" rel="noopener">Yahoo!ファイナンスで見る ↗</a></div>""")
    body = "\n".join(cards) if cards else '<p class="none">今日は条件に合う銘柄がありませんでした。</p>'
    page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>75日線ブレイク候補 {data_date}</title>
<style>
body{{margin:0;background:#f5f4f0;color:#1d1d1b;font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}}
header{{background:#fff;border-bottom:1px solid #e3e1db;padding:16px}}
h1{{font-size:18px;margin:0 0 4px}} .meta{{font-size:12px;color:#6b6a65;line-height:1.6}}
.wrap{{max-width:760px;margin:0 auto;padding:12px;display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}
.card{{display:block;background:#fff;border:1px solid #e3e1db;border-radius:12px;padding:12px;color:inherit;text-decoration:none}}
.top{{display:flex;gap:8px;align-items:center}} .code{{font-family:ui-monospace,monospace;color:#6b6a65;font-size:13px}}
.nm{{font-weight:700;font-size:15px;flex:1}}
.fr{{color:#fff;border-radius:6px;padding:2px 7px;font-family:ui-monospace,monospace;font-size:12.5px;font-weight:700;white-space:nowrap}} .fr small{{font-weight:400;opacity:.8;font-size:10px}} .sub{{font-size:12px;color:#6b6a65;margin:2px 0 6px}}
.st{{font-size:11px;padding:2px 8px;border-radius:4px;font-weight:700;white-space:nowrap}}
.st.brk{{background:#fbe9e7;color:#c8312b}} .st.near{{background:#fff4e0;color:#b26b00}}
.ch{{width:100%;height:130px;display:block;background:#faf9f6;border-radius:6px}}
.kv{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:8px;text-align:center}}
.kv b{{display:block;font-size:13px;font-family:ui-monospace,monospace}} .kv i{{font-style:normal;font-size:10px;color:#6b6a65}}
.item{{display:flex;flex-direction:column}} .item .card{{border-radius:12px 12px 0 0;border-bottom:0}}
.ylink{{display:block;background:#fff;border:1px solid #e3e1db;border-top:1px dashed #e3e1db;border-radius:0 0 12px 12px;padding:9px 12px;font-size:12.5px;color:#1f4e79;text-decoration:none;text-align:right}}
.lg2{{display:inline-block;width:14px;height:2px;vertical-align:middle;margin:0 4px 0 8px}}
.none{{text-align:center;color:#6b6a65;padding:40px}} .lg{{display:inline-block;width:14px;height:2px;background:#7a4fbf;vertical-align:middle;margin-right:4px}}
footer{{max-width:760px;margin:0 auto;padding:8px 16px 30px;font-size:11px;color:#6b6a65;line-height:1.6}}
</style></head><body>
<header><h1>75日線ブレイク候補　{len(rows)}銘柄</h1>
<div class="meta">{data_date} 終値時点（{now} 更新）｜東証プライム {checked}銘柄から検索<br>
条件：3ヶ月以上の下落後、75日線を下から突き抜けそう／抜けた直後・出来高{MIN_VOLUME:,}株以上<br>
絞り込み：抜けた日（突破目前は当日）に25日線が上向き・出来高が20日平均の{FILTER_VOL_X}倍以上<br>
チャート（約6ヶ月）：<span class="lg2" style="background:#1d1d1b"></span>終値<span class="lg2" style="background:#2a8a4a"></span>25日線<span class="lg2" style="background:#7a4fbf"></span>75日線<span class="lg2" style="background:#d9534f"></span>+2σ<span class="lg2" style="background:#f0a35e"></span>+1σ<span class="lg2" style="background:#5fa8d8"></span>−1σ<span class="lg2" style="background:#2f6fc4"></span>−2σ（ボリンジャー25日）<br>
右上の「F」はファンダメンタルランク（10点満点）。カードをタップすると、同じ見方の大きいチャートとPER・PBRなどが開きます。<a href="index.html" style="color:#1f4e79">スクリーナーに戻る</a>　<a href="forward.html" style="color:#1f4e79;font-weight:700">今日の売買（採用ルール）</a></div></header>
<div class="wrap"><a href="forward.html" style="grid-column:1/-1;display:block;background:#1f4e79;color:#fff;border-radius:10px;padding:12px 14px;text-decoration:none;font-size:13.5px">▶ 採用ルール（急騰除外・ランク6以上・引き上げ式ロスカット）で「明日買う銘柄」と「持っている銘柄のロスカット」は <b>今日の売買</b> ページへ</a>{body}</div>
<footer>機械的に条件で拾った候補で、売買のおすすめではありません。最終判断はチャートを見てご自身で行ってください。データ：Yahoo Finance（yfinance）／東証上場銘柄一覧</footer>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def write_data_json(master: pd.DataFrame, prices: dict, path: str, days: int = 260) -> None:
    """全銘柄の日足を、Webのスクリーナーが読める軽いJSONにする。"""
    import json
    fund = {}
    fpath = os.environ.get("FUND_PATH", "data/fundamentals.json")
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            fund = json.load(f)
    dates = sorted(set().union(*[set(df.index) for df in prices.values()]))[-days:]
    idx = pd.DatetimeIndex(dates)
    stocks = []
    for _, m in master.iterrows():
        df = prices.get(m["コード"])
        if df is None:
            continue
        d = df.reindex(idx)
        d["Close"] = d["Close"].ffill()
        for col in ("Open", "High", "Low"):
            d[col] = d[col].fillna(d["Close"])
        d["Volume"] = d["Volume"].fillna(0)
        d = d.dropna(subset=["Close"])
        if len(d) < 30:
            continue
        # 容量を減らすため、価格は0.1円単位の整数にし、終値は前日との差、始値・高値・安値は終値との差で持つ
        c10 = (d["Close"] * 10).round().astype("int64")
        rel = lambda col: [int(x) for x in ((d[col] * 10).round().astype("int64") - c10)]
        stocks.append({"c": m["コード"], "n": m["銘柄名"], "s": m["業種"], "st": len(idx) - len(d),
                       "dc": [int(c10.iloc[0])] + [int(x) for x in c10.diff().iloc[1:]],
                       "o": rel("Open"), "h": rel("High"), "l": rel("Low"),
                       "v": [int(round(x / 100)) for x in d["Volume"]],
                       **({"f": fund[m["コード"]]} if m["コード"] in fund else {})})
    data = {"asof": idx[-1].strftime("%Y-%m-%d"), "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market": "プライム", "fmt": 2, "dates": [x.strftime("%Y-%m-%d") for x in idx], "stocks": stocks}
    try:   # 画面の「TOPIXより弱い」条件用に、TOPIX連動ETF（1306）の終値を日付をそろえて入れる
        import yfinance as yf
        t = yf.download("1306.T", period="2y", interval="1d", auto_adjust=False, progress=False)["Close"].squeeze()
        t.index = pd.DatetimeIndex(t.index).tz_localize(None).normalize()
        t = t[~t.index.duplicated()].reindex(idx).ffill()
        data["tpx"] = [None if x != x else round(float(x), 1) for x in t]
    except Exception as e:
        print(f"  TOPIX（1306）を取得できませんでした（{e}）。画面では全銘柄の平均で代わりに計算します")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"スクリーナー用データ: {path}（{len(stocks)}銘柄）")


def main() -> None:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        sys.exit("yfinance が入っていません。先に  pip install yfinance pandas xlrd  を実行してください。")

    out_dir = os.environ.get("KABU_OUTDIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    master = load_prime_list()
    prices = download_prices(master["コード"].tolist())
    if len(prices) < len(master) * 0.5:
        sys.exit("株価を半分以上取得できませんでした。時間をおいて再実行してください。")

    rows, ohlcv = [], []
    for _, m in master.iterrows():
        df = prices.get(m["コード"])
        if df is None:
            continue
        r = screen(df)
        if r:
            rows.append({"コード": m["コード"], "銘柄名": m["銘柄名"], "業種": m["業種"], **r,
                         "Yahoo!ファイナンス": f"https://finance.yahoo.co.jp/quote/{m['コード']}.T/chart"})
            d = df.reset_index()
            for _, b in d.iterrows():
                ohlcv.append([m["コード"], m["銘柄名"], "プライム", m["業種"], b.iloc[0].strftime("%Y-%m-%d"),
                              round(b["Open"], 1), round(b["High"], 1), round(b["Low"], 1), round(b["Close"], 1), int(b["Volume"])])

    data_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    fund = {}
    fpath = os.environ.get("FUND_PATH", "data/fundamentals.json")
    if os.path.exists(fpath):
        import json
        with open(fpath, encoding="utf-8") as f:
            fund = json.load(f)
    ranks = fund_ranks(master, prices, fund)
    from collections import Counter
    cnt = Counter(ranks.values())
    print("ファンダメンタルランクの分布: " + " ".join(f"{k}:{cnt.get(k, 0)}" for k in range(11)) + f" データ不足:{cnt.get(None, 0)}")
    for r in rows:
        r["Fランク"] = ranks.get(r["コード"])
    if os.environ.get("KABU_FORWARD") == "1":   # フォワードテスト（これからの成績の記録）
        try:
            import forward_test
            forward_test.run(master, prices, ranks, out_dir)
        except Exception as e:
            print(f"フォワードテストの更新に失敗しました（{e}）")
    order = {"本日突破": 0, "突破直後": 1, "突破目前": 2}
    rows.sort(key=lambda r: (order[r["状態"]], abs(r["75日線との差(%)"])))

    page = os.path.join(out_dir, os.environ.get("KABU_PAGE_NAME", f"75日線ブレイク候補_{today}.html"))
    build_page(rows, prices, data_date, len(prices), page)
    if os.environ.get("KABU_DATA_JSON"):
        write_data_json(master, prices, os.path.join(out_dir, os.environ["KABU_DATA_JSON"]))

    if not rows:
        print("\n条件に合う銘柄はありませんでした。設定（APPROACH_GAP や DROP_MIN など）をゆるめて再実行してください。")
    else:
        res = pd.DataFrame(rows)
        out1 = os.path.join(out_dir, f"75日線ブレイク候補_{today}.csv")
        out2 = os.path.join(out_dir, f"スクリーナー読込用_{today}.csv")
        res.to_csv(out1, index=False, encoding="utf-8-sig")
        pd.DataFrame(ohlcv, columns=["コード", "銘柄名", "市場", "業種", "日付", "始値", "高値", "安値", "終値", "出来高"]) \
            .to_csv(out2, index=False, encoding="utf-8-sig")
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.width", 200)
        print(f"\n=== 75日線ブレイク候補: {len(res)} 銘柄（{data_date} 終値時点） ===")
        print(res[["状態", "コード", "銘柄名", "株価", "75日線", "75日線との差(%)",
                   "半年高値からの下落(%)", "出来高"]].to_string(index=False))
        print(f"\n保存しました: {out1} / {out2}")
    print(f"結果ページ: {page}")
    if os.environ.get("KABU_OPEN") == "1":   # ダブルクリック起動のときは結果ページを開く
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(page))


if __name__ == "__main__":
    main()
