#!/usr/bin/env python3
"""
上場廃止になった銘柄の株価を J-Quants から取ってくる（今の銘柄だけで検証している偏りを直すため。スタンダード以上）。

1. 過去の上場銘柄一覧（/equities/master?date=）を、2016年12月から3か月ごとに取得
2. その中で東証一部・プライムだったことがあり、今は上場していない銘柄を「上場廃止銘柄」とする
3. 上場廃止銘柄の日足（/equities/bars/daily?code=）を、2016年11月から取得
出力
  data/jq/master_history.csv.gz : 3か月ごとの上場銘柄一覧（日付・コード・名前・業種・市場・貸借区分）
  data/jq/delisted_bars.csv.gz  : 上場廃止銘柄の日足（J-Quants の項目そのまま）
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jquants_fetch as jf  # noqa: E402

OUT = "data/jq"
START = date(2016, 12, 1)
FROM = "20161128"


def main() -> None:
    if not jf.KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    t0 = time.time()
    snaps = []
    d = START
    while d <= date.today():
        for back in range(0, 7):                 # その月の最初の営業日（休みなら次の日）
            x = d + timedelta(days=back)
            if x.weekday() >= 5:
                continue
            st, rows, msg = jf.call("/equities/master", date=x.strftime("%Y%m%d"))
            if st == 200 and rows:
                snaps += rows
                print(f"  {x} の上場銘柄一覧: {len(rows)}件")
                break
            if st != 200:
                print(f"  {x} HTTP {st} {msg}")
        m = d.month + 3
        d = date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, 1)
    hist = pd.DataFrame(snaps)
    keep = [c for c in ["Date", "Code", "CoName", "S33", "S33Nm", "Mkt", "MktNm", "Mrgn", "MrgnNm", "ScaleCat"] if c in hist]
    hist = hist[keep]
    hist.to_csv(os.path.join(OUT, "master_history.csv.gz"), index=False, compression="gzip")
    now = pd.read_csv(os.path.join(OUT, "master.csv"), dtype=str)
    now_codes = set(now["Code"])
    big = hist[hist["MktNm"].fillna("").str.contains("一部|プライム")]
    gone = sorted(set(big["Code"]) - now_codes)
    print(f"東証一部・プライムだったことがあり、今は上場していない銘柄: {len(gone)}")
    bars = []
    for n, code in enumerate(gone):
        st, rows, msg = jf.call("/equities/bars/daily", code=code, **{"from": FROM})
        if st != 200:
            print(f"  {code} HTTP {st} {msg}")
            continue
        bars += rows
        if n % 50 == 0:
            print(f"  {n + 1}/{len(gone)} {code} {len(rows)}日分")
    if bars:
        b = pd.DataFrame(bars)
        b.to_csv(os.path.join(OUT, "delisted_bars.csv.gz"), index=False, compression="gzip")
        print(f"上場廃止銘柄の日足: {len(b)}件・{b['Code'].nunique()}銘柄 / 項目: {list(b.columns)}")
    print(f"かかった時間: {(time.time() - t0) / 60:.0f}分")


if __name__ == "__main__":
    main()
