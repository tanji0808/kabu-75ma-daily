#!/usr/bin/env python3
"""
プレミアムプランの間だけ取れる、古い時期（2008年〜2016年11月）のデータをまとめて取る（2026-11-28 作成）。
今の検証は2016年12月から。これで2008年のリーマン・ショックを含む約18年で検証できるようにする。

出力（data/jq_old/）
  bars_YYYY.csv.gz        : 全銘柄の日足（年ごと。上場廃止の銘柄も含む）
  fins_summary_old.csv.gz : 決算の概要
  margin_interest_old.csv.gz : 週ごとの信用残
  short_ratio_old.csv.gz  : 業種別の空売り比率
  topix_old.csv.gz / investor_types_old.csv.gz / calendar_old.csv.gz
  master_YYYYMM.csv.gz    : 3か月ごとの上場銘柄一覧（master_history_old.csv.gz にまとめる）
  done.json               : 取り終わった分の記録（途中で止まっても続きから取る）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pandas as pd

os.environ.setdefault("JQ_WAIT", "0.25")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jquants_fetch as jf  # noqa: E402

OUT = "data/jq_old"
FROM, TO = "2008-01-01", "2016-11-27"
MAX_MIN = float(os.environ.get("JQ_MAX_MINUTES", "330"))
T0 = time.time()
DONE_P = os.path.join(OUT, "done.json")
BAR_COLS = ["Date", "Code", "O", "H", "L", "C", "UL", "LL", "Vo", "Va", "AdjFactor", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]


def push(msg: str) -> None:
    for c in (["git", "add", OUT], ["git", "commit", "-q", "-m", msg], ["git", "pull", "-q", "--rebase", "origin", "main"], ["git", "push", "-q"]):
        subprocess.run(c, check=False)


def left() -> bool:
    return (time.time() - T0) / 60 < MAX_MIN


def main() -> None:
    if not jf.KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    os.makedirs(OUT, exist_ok=True)
    done = json.load(open(DONE_P)) if os.path.exists(DONE_P) else {}
    st, rows, msg = jf.call("/equities/bars/daily", date="20100105")
    print("プランの確認（2010-01-05 の日足）:", st, len(rows), msg[:150])
    if st != 200 or not rows:
        sys.exit("古い日足が取れません。プレミアムプランになっているか確認してください。")
    # 営業日カレンダー・TOPIX・投資部門別（期間で一度に取れるもの）
    for name, path in (("calendar", "/markets/calendar"), ("topix", "/indices/bars/daily/topix"), ("investor_types", "/equities/investor-types")):
        if done.get(name):
            continue
        st, rows, msg = jf.call(path, **{"from": FROM.replace("-", ""), "to": TO.replace("-", "")})
        print(name, st, len(rows), msg[:100])
        if st == 200 and rows:
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_old.csv.gz"), index=False, compression="gzip")
            done[name] = len(rows)
    json.dump(done, open(DONE_P, "w"))
    cal = pd.read_csv(os.path.join(OUT, "calendar_old.csv.gz"), dtype=str)
    hcol = [c for c in cal.columns if "Hol" in c or "hol" in c][0]
    days = sorted(d for d, h in zip(cal.Date, cal[hcol]) if h in ("1", "2") and FROM <= d <= TO)
    print("営業日", len(days), days[:2], days[-2:])
    # 日足（年ごと）
    for y in sorted({d[:4] for d in days}):
        key = f"bars_{y}"
        if done.get(key) or not left():
            continue
        parts = []
        for d in [d for d in days if d[:4] == y]:
            st, rows, msg = jf.call("/equities/bars/daily", date=d.replace("-", ""))
            if st != 200:
                print("  日足の失敗", d, st, msg[:100]); continue
            parts += rows
        df = pd.DataFrame(parts)
        df[[c for c in BAR_COLS if c in df.columns]].to_csv(os.path.join(OUT, f"{key}.csv.gz"), index=False, compression="gzip")
        done[key] = len(df); json.dump(done, open(DONE_P, "w"))
        print(key, len(df), f"{(time.time() - T0) / 60:.0f}分"); push(f"J-Quants 古い日足 {y}年")
    # 決算の概要・空売り比率（日ごと）
    for name, path, col in (("fins_summary", "/fins/summary", jf.COLS), ("short_ratio", "/markets/short-ratio", None)):
        if done.get(name) or not left():
            continue
        parts = []
        for d in days:
            st, rows, msg = jf.call(path, date=d.replace("-", ""))
            if st == 200:
                parts += rows
        df = pd.DataFrame(parts)
        if col and len(df):
            df = df[[c for c in col if c in df.columns]]
        df.to_csv(os.path.join(OUT, f"{name}_old.csv.gz"), index=False, compression="gzip")
        done[name] = len(df); json.dump(done, open(DONE_P, "w"))
        print(name, len(df), f"{(time.time() - T0) / 60:.0f}分"); push(f"J-Quants 古い{name}")
    # 信用残（週ごと：金曜。休みならその前の日）
    if not done.get("margin_interest") and left():
        ds = pd.to_datetime(days); fr = sorted({x.strftime("%Y-%m-%d") for x in ds[ds.weekday == 4]} | {ds[-1].strftime("%Y-%m-%d")})
        parts = []
        for d in fr:
            st, rows, msg = jf.call("/markets/margin-interest", date=d.replace("-", ""))
            if st == 200:
                parts += rows
        pd.DataFrame(parts).to_csv(os.path.join(OUT, "margin_interest_old.csv.gz"), index=False, compression="gzip")
        done["margin_interest"] = len(parts); json.dump(done, open(DONE_P, "w")); push("J-Quants 古い信用残")
    # 上場銘柄一覧（3か月ごと）
    if not done.get("master") and left():
        parts = []
        for d in [d for d in days if d[5:7] in ("01", "04", "07", "10")]:
            if parts and parts[-1]["_m"] == d[:7]:
                continue
            st, rows, msg = jf.call("/equities/master", date=d.replace("-", ""))
            if st == 200:
                for r in rows:
                    r["_m"] = d[:7]; r["SnapDate"] = d
                parts += rows
        pd.DataFrame(parts).drop(columns=["_m"]).to_csv(os.path.join(OUT, "master_history_old.csv.gz"), index=False, compression="gzip")
        done["master"] = len(parts); json.dump(done, open(DONE_P, "w")); push("J-Quants 古い上場銘柄一覧")
    print("取り終わった分:", done)
    print("のこり:", [k for k in ["calendar", "topix", "investor_types"] + [f"bars_{y}" for y in sorted({d[:4] for d in days})] + ["fins_summary", "short_ratio", "margin_interest", "master"] if not done.get(k)])


if __name__ == "__main__":
    main()
