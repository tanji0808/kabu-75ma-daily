#!/usr/bin/env python3
"""
東証プライム全銘柄の財務データ（EPS・BPS・ROEなど）を Yahoo Finance（yfinance）から取得し、
data/fundamentals.json に保存します。

PER・PBR・配当利回り・時価総額は、ここで保存した「1株あたり」の値と毎日の終値から
スクリーナー画面で計算します。決算で変わる値なので、1銘柄あたり週1回ほど取り直せば十分です。

1回の実行で取得するのは「まだ無い銘柄」と「古い銘柄」から最大 FUND_LIMIT 件
（初期値 500件）。毎日2回の実行で、数日かけて全銘柄が揃い、その後は約1週間ごとに更新されます。
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

PATH = os.environ.get("FUND_PATH", "data/fundamentals.json")
LIMIT = int(os.environ.get("FUND_LIMIT", "500"))
STALE_DAYS = 7
WORKERS = 4

# Yahoo の項目名 → 保存する短い名前
FIELDS = {
    "trailingEps": "eps",          # 実績EPS（直近12ヶ月、円）
    "forwardEps": "feps",          # 予想EPS（円）
    "bookValue": "bps",            # 1株あたり純資産（円）
    "sharesOutstanding": "sh",     # 発行済株式数
    "returnOnEquity": "roe",       # ROE（0.12 = 12%）
    "operatingMargins": "om",      # 営業利益率
    "revenueGrowth": "rg",         # 売上高成長率（直近四半期の前年同期比）
    "earningsGrowth": "eg",        # 利益成長率（同上）
    "dividendRate": "div",         # 1株あたり年間配当（予想、円）
    "trailingAnnualDividendRate": "tdiv",  # 1株あたり年間配当（実績、円）
    "debtToEquity": "de",          # 負債資本倍率（%）
}


def fetch_one(code: str) -> dict | None:
    import yfinance as yf
    for attempt in range(3):
        try:
            info = yf.Ticker(f"{code}.T").info or {}
            out = {}
            for k, short in FIELDS.items():
                v = info.get(k)
                if isinstance(v, (int, float)) and v == v:   # NaNを除く
                    out[short] = round(float(v), 4) if abs(v) < 1e6 else int(v)
            return out
        except Exception as e:
            msg = str(e)
            if "Too Many Requests" in msg or "Rate" in msg:
                time.sleep(20 * (attempt + 1))
            else:
                time.sleep(2)
    return None


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kabu_75ma_search import load_prime_list

    cache: dict = {}
    if os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as f:
            cache = json.load(f)

    codes = load_prime_list()["コード"].tolist()
    today = date.today()

    def age(c: str) -> int:
        t = cache.get(c, {}).get("t")
        return 10_000 if not t else (today - date.fromisoformat(t)).days

    todo = sorted([c for c in codes if age(c) >= STALE_DAYS], key=age, reverse=True)[:LIMIT]
    print(f"財務データ: 保存済み {len(cache)} 銘柄 / 今回の取得対象 {len(todo)} 銘柄")

    ok = fail = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            r = fu.result()
            if r is None:
                fail += 1
            else:
                r["t"] = today.isoformat()
                cache[c] = r
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}（成功 {ok}・失敗 {fail}）")
            if fail > 50 and ok < fail:   # 取得制限にかかっているときは無理をせず止める
                print("  取得の失敗が続くため、今回はここまでにします。")
                for f2 in futs:
                    f2.cancel()
                break

    # 上場廃止などで一覧から消えた銘柄は削除
    cache = {c: cache[c] for c in codes if c in cache}
    os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    print(f"財務データを保存しました: 成功 {ok}・失敗 {fail}・保存済み合計 {len(cache)} 銘柄（{datetime.now():%Y-%m-%d %H:%M}）")


if __name__ == "__main__":
    main()
