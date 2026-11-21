#!/usr/bin/env python3
"""
J-Quants API から決算の概要（/fins/summary）を開示日ごとに取得して、data/jq/ にためる。
すでに取得した日は飛ばすので、2回目以降は新しい日の分だけ取りに行く。
無料プランは「直近12週間を除く2年分」・1分に5回までなので、1回ごとに約13秒待つ。

出力
  data/jq/fins_summary.csv.gz : 決算の概要（バックテストで使う項目だけ）
  data/jq/fins_dates.json     : 取得済みの日付と件数
  data/jq/master.csv          : 上場銘柄一覧（市場・業種・貸借区分）
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

KEY = os.environ.get("JQUANTS_API_KEY", "").strip()
BASE = "https://api.jquants.com/v2"
OUT = "data/jq"
START = os.environ.get("JQ_START", "")          # 例 2024-11-01。空なら今日から2年13週間前
WAIT = float(os.environ.get("JQ_WAIT", "13"))   # 無料プランは1分に5回まで
MAX_MIN = float(os.environ.get("JQ_MAX_MINUTES", "300"))
COLS = ["DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn",
        "Sales", "OP", "NP", "EPS", "BPS", "Eq", "TA", "DivAnn", "FDivAnn", "NxFDivAnn",
        "FSales", "FOP", "FNP", "FEPS", "NxFSales", "NxFOP", "NxFNp", "NxFEPS", "ShOutFY", "TrShFY"]


def call(path: str, **params) -> tuple[int, list, str]:
    rows, key = [], None
    while True:
        p = dict(params)
        if key:
            p["pagination_key"] = key
        for attempt in range(5):
            r = requests.get(BASE + path, headers={"x-api-key": KEY}, params=p, timeout=60)
            if r.status_code == 429:
                time.sleep(65)
                continue
            break
        time.sleep(WAIT)
        if r.status_code != 200:
            return r.status_code, rows, r.text[:200]
        j = r.json()
        rows += j.get("data", [])
        key = j.get("pagination_key")
        if not key:
            return 200, rows, ""


def main() -> None:
    if not KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    st, rows, msg = call("/equities/master")
    if st == 200 and rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "master.csv"), index=False, encoding="utf-8-sig")
        print(f"上場銘柄一覧: {len(rows)}件")
    dpath, fpath = os.path.join(OUT, "fins_dates.json"), os.path.join(OUT, "fins_summary.csv.gz")
    done = json.load(open(dpath)) if os.path.exists(dpath) else {}
    old = pd.read_csv(fpath, dtype=str) if os.path.exists(fpath) else pd.DataFrame(columns=COLS)
    start = date.fromisoformat(START) if START else date.today() - timedelta(weeks=2 * 52 + 13)
    days = []
    d = start
    while d <= date.today():
        if d.weekday() < 5 and d.isoformat() not in done:
            days.append(d)
        d += timedelta(days=1)
    print(f"取得する日数: {len(days)}（取得済み {len(done)}日）")
    new = []
    for n, d in enumerate(days):
        if (time.time() - t0) / 60 > MAX_MIN:
            print("時間の上限に達したので、続きは次回取得します")
            break
        st, rows, msg = call("/fins/summary", date=d.strftime("%Y%m%d"))
        if st != 200:
            # 契約プランの期間外（直近12週間など）は記録せず、次回また試す
            print(f"  {d} HTTP {st} {msg}")
            continue
        if d < date.today() - timedelta(weeks=13) or rows:
            done[d.isoformat()] = len(rows)   # 無料プランで見えない直近の日は、あとで見えるようになるので記録しない
        new +=[{c: x.get(c) for c in COLS} for x in rows]
        if n % 25 == 0:
            print(f"  {d} {len(rows)}件（{n + 1}/{len(days)}）")
    if new:
        df = pd.concat([old, pd.DataFrame(new, columns=COLS).astype(str)], ignore_index=True)
        df = df.drop_duplicates(["DiscDate", "DiscTime", "Code", "DocType"], keep="last")
        df.to_csv(fpath, index=False, compression="gzip")
        print(f"決算の概要: 合計 {len(df)}件（今回 {len(new)}件）")
    json.dump(dict(sorted(done.items())), open(dpath, "w"), indent=0)
    print(f"かかった時間: {(time.time() - t0) / 60:.0f}分 / 更新 {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
