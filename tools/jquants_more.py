#!/usr/bin/env python3
"""
J-Quants の追加データを取得して data/jq/ に保存する（スタンダードプラン用・取得済みの日は飛ばす）。

日ごとに取るもの（date などで1日ずつ）
  short_sale_report : 空売り残高報告（残高割合0.5%以上の報告）              /markets/short-sale-report?disc_date=
  short_ratio       : 業種別空売り比率                                     /markets/short-ratio?date=
  margin_alert      : 日々公表信用取引残高                                 /markets/margin-alert?date=
期間でまとめて取るもの（from / to）
  investor_types    : 投資部門別の売買状況（週ごと）                        /equities/investor-types
  topix             : TOPIXの四本値（指数そのもの）                         /indices/bars/daily/topix
  calendar          : 取引カレンダー                                       /markets/calendar
  earnings_calendar : これからの決算発表予定日（3月・9月決算の会社・最新分だけ） /equities/earnings-calendar
最初に各APIを1回ずつ呼んで、使えるか・項目名を表示する。使えないものは飛ばす。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jquants_fetch as jf  # noqa: E402

OUT = "data/jq"
START = date(2016, 11, 28)
DAILY = {"short_sale_report": ("/markets/short-sale-report", "disc_date"),
         "short_ratio": ("/markets/short-ratio", "date"),
         "margin_alert": ("/markets/margin-alert", "date")}
RANGE = {"investor_types": "/equities/investor-types", "topix": "/indices/bars/daily/topix",
         "calendar": "/markets/calendar"}
MAX_MIN = float(os.environ.get("JQ_MAX_MINUTES", "200"))


def probe() -> set[str]:
    ok = set()
    test_day = "20250805"
    for name, (path, prm) in DAILY.items():
        st, rows, msg = jf.call(path, **{prm: test_day})
        print(f"[確認] {name} HTTP {st} {len(rows)}件 {list(rows[0].keys()) if rows else msg[:120]}")
        if st == 200:
            ok.add(name)
    for name, path in RANGE.items():
        st, rows, msg = jf.call(path, **{"from": "20250801", "to": "20250808"})
        print(f"[確認] {name} HTTP {st} {len(rows)}件 {list(rows[0].keys()) if rows else msg[:120]}")
        if st == 200:
            ok.add(name)
    st, rows, msg = jf.call("/equities/earnings-calendar")
    print(f"[確認] earnings_calendar HTTP {st} {len(rows)}件 {list(rows[0].keys()) if rows else msg[:120]}")
    if st == 200 and rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "earnings_calendar.csv"), index=False, encoding="utf-8-sig")
    return ok


def fetch_range(name: str, path: str) -> None:
    st, rows, msg = jf.call(path, **{"from": START.strftime("%Y%m%d"), "to": date.today().strftime("%Y%m%d")})
    if st == 200 and rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}.csv.gz"), index=False, compression="gzip")
        print(f"{name}: {len(rows)}件を保存")
    else:
        print(f"{name}: HTTP {st} {msg[:200]}")


def fetch_daily(name: str, path: str, prm: str, t0: float) -> bool:
    dpath, fpath = os.path.join(OUT, f"{name}_dates.json"), os.path.join(OUT, f"{name}.csv.gz")
    done = json.load(open(dpath)) if os.path.exists(dpath) else {}
    old = pd.read_csv(fpath, dtype=str) if os.path.exists(fpath) else None
    new, d, last_save = [], START, time.time()

    def save():
        nonlocal old, new
        if new:
            df = pd.DataFrame(new).astype(str)
            old = df if old is None else pd.concat([old, df], ignore_index=True)
            old.drop_duplicates().to_csv(fpath, index=False, compression="gzip")
            new = []
        json.dump(done, open(dpath, "w"))

    while d <= date.today():
        if (time.time() - t0) / 60 > MAX_MIN:
            save()
            print(f"{name}: 時間の上限。{len(done)}日分まで保存")
            return False
        if d.weekday() < 5 and d.isoformat() not in done:
            st, rows, msg = jf.call(path, **{prm: d.strftime("%Y%m%d")})
            if st == 200:
                new += rows
                if d < date.today() - timedelta(days=7):
                    done[d.isoformat()] = len(rows)
            else:
                print(f"  {name} {d} HTTP {st} {msg[:100]}")
            if time.time() - last_save > 900:
                save()
                last_save = time.time()
                print(f"  {name}: {d} まで（{len(done)}日分）")
        d += timedelta(days=1)
    save()
    print(f"{name}: 完了（{len(done)}日分）")
    return True


def main() -> None:
    if not jf.KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    t0 = time.time()
    ok = probe()
    for name, path in RANGE.items():
        if name in ok:
            fetch_range(name, path)
    finished = True
    for name, (path, prm) in DAILY.items():
        if name in ok:
            finished = fetch_daily(name, path, prm, t0) and finished
    with open(os.path.join(OUT, "more_remaining.txt"), "w") as f:
        f.write("0" if finished else "1")
    print(f"かかった時間: {(time.time() - t0) / 60:.0f}分")


if __name__ == "__main__":
    main()
