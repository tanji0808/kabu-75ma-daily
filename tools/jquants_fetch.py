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
import subprocess
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
SAVE_MIN = float(os.environ.get("JQ_SAVE_MINUTES", "20"))   # この間隔で途中経過をGitHubに保存する（0なら保存しない）
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


def save(old: pd.DataFrame, new: list, done: dict, fpath: str, dpath: str, push: bool) -> pd.DataFrame:
    """取得した分をファイルに書き、push=True ならGitHubにも保存する（実行環境が途中で落ちても残るように）。"""
    df = old
    if new:
        df = pd.concat([old, pd.DataFrame(new, columns=COLS).astype(str)], ignore_index=True)
        df = df.drop_duplicates(["DiscDate", "DiscTime", "Code", "DocType"], keep="last")
        df.to_csv(fpath, index=False, compression="gzip")
    json.dump(dict(sorted(done.items())), open(dpath, "w"), indent=0)
    if push and os.environ.get("GITHUB_ACTIONS"):
        cmds = [["git", "add", OUT], ["git", "commit", "-q", "-m", f"J-Quantsの決算データ（途中経過・{len(done)}日分）"],
                ["git", "pull", "-q", "--rebase", "origin", "main"], ["git", "push", "-q"]]
        for c in cmds:
            r = subprocess.run(c, capture_output=True, text=True)
            if r.returncode and c[1] != "commit":
                print("  保存に失敗:", " ".join(c), r.stderr[:200])
                break
        print(f"  途中経過を保存しました（{len(done)}日分・{len(df)}件）")
    return df


MCOLS = ["Date", "PubDate", "Code", "IssType", "ShrtVol", "LongVol", "ShrtStdVol", "LongStdVol", "ShrtNegVol", "LongNegVol"]


def fetch_margin(start: date, t0: float) -> None:
    """信用取引の週末残高（/markets/margin-interest）を、週ごと（金曜。休みなら木曜）にためる。スタンダード以上のプランのみ。"""
    mpath, mdpath = os.path.join(OUT, "margin_interest.csv.gz"), os.path.join(OUT, "margin_dates.json")
    done = json.load(open(mdpath)) if os.path.exists(mdpath) else {}
    old = pd.read_csv(mpath, dtype=str) if os.path.exists(mpath) else pd.DataFrame(columns=MCOLS)
    new = []
    d = start
    while d <= date.today() - timedelta(days=4):   # 金曜の分は翌週月曜の夕方に出るので、火曜以降に取る
        if d.weekday() == 4 and d.isoformat() not in done:
            for back in range(0, 3):          # 金曜が休みなら木曜・水曜
                x = d - timedelta(days=back)
                st, rows, msg = call("/markets/margin-interest", date=x.strftime("%Y%m%d"))
                if st == 403:
                    print("  信用残はこのプランでは取れません")
                    return
                if st == 200 and rows:
                    new += [{c: r.get(c) for c in MCOLS} for r in rows]
                    break
            done[d.isoformat()] = len(rows) if st == 200 else 0
            if (time.time() - t0) / 60 > MAX_MIN:
                break
        d += timedelta(days=1)
    if new:
        df = pd.concat([old, pd.DataFrame(new, columns=MCOLS).astype(str)], ignore_index=True)
        df = df.drop_duplicates(["Date", "Code"], keep="last")
        df.to_csv(mpath, index=False, compression="gzip")
        print(f"信用残: 合計 {len(df)}件（今回 {len(new)}件）")
    json.dump(dict(sorted(done.items())), open(mdpath, "w"), indent=0)


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
    # 前に「契約プランの期間外」で取れなかった日は、プランを変えたら取れることがあるので毎回もう一度試す
    done = {d: n for d, n in done.items() if n != -1}
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
    last_save = time.time()
    for n, d in enumerate(days):
        if SAVE_MIN and (time.time() - last_save) / 60 > SAVE_MIN:
            old = save(old, new, done, fpath, dpath, push=True)
            new, last_save = [], time.time()
        if (time.time() - t0) / 60 > MAX_MIN:
            print("時間の上限に達したので、続きは次回取得します")
            break
        st, rows, msg = call("/fins/summary", date=d.strftime("%Y%m%d"))
        if st != 200:
            print(f"  {d} HTTP {st} {msg}")
            # 契約プランより古い日は取れないので記録して飛ばす。直近12週間などは記録せず、次回また試す
            if st in (400, 403) and d < date.today() - timedelta(weeks=100):
                done[d.isoformat()] = -1
            continue
        if d < date.today() - timedelta(weeks=13) or rows:
            done[d.isoformat()] = len(rows)   # 無料プランで見えない直近の日は、あとで見えるようになるので記録しない
        new +=[{c: x.get(c) for c in COLS} for x in rows]
        if n % 25 == 0:
            print(f"  {d} {len(rows)}件（{n + 1}/{len(days)}）")
    df = save(old, new, done, fpath, dpath, push=False)
    print(f"決算の概要: 合計 {len(df)}件")
    # まだ取れていない日（直近13週間を除く）の数。0になったら検証に進む
    limit = date.today() - timedelta(weeks=13)
    left = sum(1 for x in days if x.isoformat() not in done and x < limit)
    with open(os.path.join(OUT, "remaining.txt"), "w") as f:
        f.write(str(left))
    if os.environ.get("JQ_MARGIN") == "1" and (time.time() - t0) / 60 < MAX_MIN:
        fetch_margin(start, t0)
    print(f"まだ取れていない日: {left}日 / かかった時間: {(time.time() - t0) / 60:.0f}分 / 更新 {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
