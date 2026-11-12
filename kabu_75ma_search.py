#!/usr/bin/env python3
"""
東証プライム 75日線ブレイク候補 検索スクリプト
================================================

『世界一やさしい 株の教科書 1年生』の考え方（6ヶ月チャートで、長く下落したあとに
75日移動平均線を下から上へ突き抜ける＝上昇トレンドへの転換）を、条件として数値化して
東証プライム全銘柄から探します。

  条件（初期値。下の「設定」で変えられます）
    1. 東証プライム銘柄
    2. 出来高 30万株以上（直近の営業日）
    3. 3ヶ月〜半年 下落が続いている
         - 75日線が約3ヶ月前（60営業日前）より 2% 以上下がっている
         - 直近3ヶ月の 75% 以上の日で、終値が75日線の下にある
         - 過去6ヶ月の高値から 15% 以上下げた
    4. 75日線を下から突き抜けそう／突き抜けた直後
         - 突破目前：終値が75日線の 5% 下〜75日線の間 で、直近5日で上昇中
         - 突破直後：直近5営業日以内に終値が75日線を下から上に抜け、上抜け幅 3% 以内

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

import pandas as pd

# ============================== 設定 ==============================
MIN_VOLUME = 300_000          # 出来高の下限（株）
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
    return out


def download_prices(codes: list[str]) -> dict[str, pd.DataFrame]:
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
                data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
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
        "日付": df.index[L].strftime("%Y-%m-%d"),
    }


def _svg_chart(df: pd.DataFrame, days: int = 120) -> str:
    """6ヶ月分の終値（ローソク足の代わりに線）と75日線の小さなチャート。"""
    ma = df["Close"].rolling(MA).mean()
    d = df.iloc[-days:]
    m = ma.iloc[-days:]
    vals = list(d["Low"]) + list(d["High"]) + [x for x in m if pd.notna(x)]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    W, H, n = 340, 130, len(d)
    X = lambda i: 4 + i * (W - 8) / max(n - 1, 1)
    Y = lambda v: 4 + (hi - v) / (hi - lo) * (H - 8)
    bars = []
    bw = max(1.0, (W - 8) / n * 0.6)
    for i, (_, b) in enumerate(d.iterrows()):
        col = "#c8312b" if b["Close"] >= b["Open"] else "#1f5fbf"
        y1, y2 = Y(max(b["Open"], b["Close"])), Y(min(b["Open"], b["Close"]))
        bars.append(f'<line x1="{X(i):.1f}" y1="{Y(b["High"]):.1f}" x2="{X(i):.1f}" y2="{Y(b["Low"]):.1f}" stroke="{col}" stroke-width=".7"/>'
                    f'<rect x="{X(i) - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{max(y2 - y1, .6):.1f}" fill="{col}"/>')
    ma_pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(m) if pd.notna(v))
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="ch">{"".join(bars)}'
            f'<polyline points="{ma_pts}" fill="none" stroke="#7a4fbf" stroke-width="1.8"/></svg>')


def build_page(rows: list[dict], prices: dict, data_date: str, checked: int, path: str) -> None:
    """結果をスマホでも見やすいHTMLページにする。"""
    import html as h
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = []
    for r in rows:
        tone = "brk" if r["状態"] != "突破目前" else "near"
        gap = r["75日線との差(%)"]
        cards.append(f"""<a class="card" href="{h.escape(r['Yahoo!ファイナンス'])}">
  <div class="top"><span class="st {tone}">{h.escape(r['状態'])}</span><span class="code">{h.escape(r['コード'])}</span>
  <span class="nm">{h.escape(r['銘柄名'])}</span></div>
  <div class="sub">{h.escape(str(r['業種']))}</div>
  {_svg_chart(prices[r['コード']])}
  <div class="kv"><div><b>{r['株価']:,.1f}</b><i>株価</i></div><div><b>{gap:+.2f}%</b><i>75日線との差</i></div>
  <div><b>{r['半年高値からの下落(%)']:.1f}%</b><i>半年高値から</i></div><div><b>{r['出来高']:,}</b><i>出来高</i></div></div>
</a>""")
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
.nm{{font-weight:700;font-size:15px}} .sub{{font-size:12px;color:#6b6a65;margin:2px 0 6px}}
.st{{font-size:11px;padding:2px 8px;border-radius:4px;font-weight:700;white-space:nowrap}}
.st.brk{{background:#fbe9e7;color:#c8312b}} .st.near{{background:#fff4e0;color:#b26b00}}
.ch{{width:100%;height:130px;display:block;background:#faf9f6;border-radius:6px}}
.kv{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:8px;text-align:center}}
.kv b{{display:block;font-size:13px;font-family:ui-monospace,monospace}} .kv i{{font-style:normal;font-size:10px;color:#6b6a65}}
.none{{text-align:center;color:#6b6a65;padding:40px}} .lg{{display:inline-block;width:14px;height:2px;background:#7a4fbf;vertical-align:middle;margin-right:4px}}
footer{{max-width:760px;margin:0 auto;padding:8px 16px 30px;font-size:11px;color:#6b6a65;line-height:1.6}}
</style></head><body>
<header><h1>75日線ブレイク候補　{len(rows)}銘柄</h1>
<div class="meta">{data_date} 終値時点（{now} 更新）｜東証プライム {checked}銘柄から検索<br>
条件：3ヶ月以上の下落後、75日線を下から突き抜けそう／抜けた直後・出来高{MIN_VOLUME:,}株以上<br>
<span class="lg"></span>紫の線＝75日線（チャートは約6ヶ月）。カードをタップするとYahoo!ファイナンスのチャートが開きます。</div></header>
<div class="wrap">{body}</div>
<footer>機械的に条件で拾った候補で、売買のおすすめではありません。最終判断はチャートを見てご自身で行ってください。データ：Yahoo Finance（yfinance）／東証上場銘柄一覧</footer>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def write_data_json(master: pd.DataFrame, prices: dict, path: str, days: int = 260) -> None:
    """全銘柄の日足を、Webのスクリーナーが読める軽いJSONにする。"""
    import json
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
                       "v": [int(round(x / 100)) for x in d["Volume"]]})
    data = {"asof": idx[-1].strftime("%Y-%m-%d"), "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market": "プライム", "fmt": 2, "dates": [x.strftime("%Y-%m-%d") for x in idx], "stocks": stocks}
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
