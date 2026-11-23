#!/usr/bin/env python3
"""
IPO（新規上場）銘柄の一覧と、上場日からの日足を J-Quants から取ってくる（上場廃止になった銘柄も含む。スタンダード以上）。

IPO銘柄 = 3か月ごとの上場銘柄一覧（data/jq/master_history.csv.gz）に、2017年3月以降に初めて現れた銘柄のうち、
          株式の市場（プライム・スタンダード・グロース・旧一部・二部・マザーズ・JASDAQ）に上場したもの。
上場日 = 日足の最初の日（その日の始値が初値）。
出力
  data/jq/ipo_list.csv.gz : Code, CoName, MktNm（最初に載った市場）, S33, S33Nm, first_snap
  data/jq/ipo_bars.csv.gz : 日足（J-Quants の項目そのまま。調整済み AdjO/AdjH/AdjL/AdjC/AdjVo を含む）
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jquants_fetch as jf  # noqa: E402

OUT = "data/jq"
MKT = "プライム|スタンダード|グロース|一部|二部|マザーズ|JASDAQ"


def main() -> None:
    if not jf.KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    t0 = time.time()
    h = pd.read_csv(os.path.join(OUT, "master_history.csv.gz"), dtype=str)
    # 最新の一覧も足す（直近3か月のIPOを拾うため）
    st, rows, msg = jf.call("/equities/master")
    if st == 200 and rows:
        now = pd.DataFrame(rows)
        h = pd.concat([h, now[[c for c in h.columns if c in now]]], ignore_index=True)
    h = h.sort_values("Date")
    first = h.drop_duplicates("Code", keep="first")
    first0 = h["Date"].min()
    ipo = first[(first["Date"] > first0) & first["MktNm"].fillna("").str.contains(MKT)].copy()
    ipo = ipo.rename(columns={"Date": "first_snap"})[["Code", "CoName", "MktNm", "S33", "S33Nm", "first_snap"]]
    print(f"IPO銘柄: {len(ipo)}件（{ipo.first_snap.min()}〜{ipo.first_snap.max()}）")
    ipo.to_csv(os.path.join(OUT, "ipo_list.csv.gz"), index=False, compression="gzip")
    bars = []
    for n, code in enumerate(ipo["Code"]):
        st, rows, msg = jf.call("/equities/bars/daily", code=code, **{"from": "20161128"})
        if st != 200:
            print(f"  {code} HTTP {st} {msg}")
            continue
        bars += rows
        if n % 100 == 0:
            print(f"  {n + 1}/{len(ipo)} {code} {len(rows)}日分（{(time.time() - t0) / 60:.0f}分）", flush=True)
    b = pd.DataFrame(bars)
    b.to_csv(os.path.join(OUT, "ipo_bars.csv.gz"), index=False, compression="gzip")
    print(f"IPO銘柄の日足: {len(b)}件・{b['Code'].nunique()}銘柄 / {(time.time() - t0) / 60:.0f}分")


if __name__ == "__main__":
    main()
