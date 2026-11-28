#!/usr/bin/env python3
"""
J-Quants の一括ダウンロード（/bulk/list・/bulk/get）で、今のプランで取れるデータを全部・全期間取る（2026-11-28 作成）。
プレミアムの間に使う（2008年〜・先物・オプション・売買内訳・財務諸表・配当なども含む）。

ファイルは大きいので、リポジトリではなく GitHub の「リリース」（タグ jq-bulk）に置く。
  一覧：data/jq_bulk_manifest.csv（キー・データ名・大きさ・更新日時・リリースの中の名前）
取り直し：一覧にあって更新日時が同じものは飛ばすので、途中で止まっても続きから取れる。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pandas as pd
import requests

KEY = os.environ.get("JQUANTS_API_KEY", "").strip()
BASE = "https://api.jquants.com/v2"
WAIT = float(os.environ.get("JQ_WAIT", "0.15"))
MAX_MIN = float(os.environ.get("JQ_MAX_MINUTES", "320"))
TAG = "jq-bulk"
MAN = "data/jq_bulk_manifest.csv"
T0 = time.time()
TMP = "/tmp/jqb"


def api(path, **params):
    for _ in range(6):
        r = requests.get(BASE + path, headers={"x-api-key": KEY}, params=params, timeout=120)
        time.sleep(WAIT)
        if r.status_code == 429:
            time.sleep(30); continue
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    return 429, {}


def items_of(j):
    """返ってきたJSONの中の「ファイルの一覧」を探す（項目名が多少違っても動くように）。"""
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        for v in j.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def field(d, *names):
    for n in names:
        for k in d:
            if k.lower() == n.lower():
                return d[k]
    return None


def url_of(j):
    if isinstance(j, dict):
        for v in j.values():
            if isinstance(v, str) and v.startswith("http"):
                return v
            u = url_of(v)
            if u:
                return u
    if isinstance(j, list):
        for v in j:
            u = url_of(v)
            if u:
                return u
    return None


def sh(*c):
    return subprocess.run(list(c), check=False, capture_output=True, text=True)


def main():
    if not KEY:
        sys.exit("JQUANTS_API_KEY が設定されていません")
    os.makedirs(TMP, exist_ok=True)
    if sh("gh", "release", "view", TAG).returncode != 0:
        print(sh("gh", "release", "create", TAG, "--title", "J-Quants 一括データ", "--notes", "J-Quants の一括ダウンロードのファイル（data/jq_bulk_manifest.csv が一覧）").stderr)
    man = pd.read_csv(MAN, dtype=str) if os.path.exists(MAN) else pd.DataFrame(columns=["key", "endpoint", "size", "modified", "asset"])
    have = {(k, m) for k, m in zip(man.key, man.modified)}
    # 1) 月ごとに「その月の全データのファイル一覧」を取る
    st, j = api("/bulk/list", date="2010-01")
    print("一覧のお試し（2010-01）:", st, json.dumps(j, ensure_ascii=False)[:600])
    if st != 200:
        sys.exit("一括ダウンロードの一覧が取れません")
    files = {}
    months = pd.period_range("2006-01", pd.Timestamp.today().strftime("%Y-%m"), freq="M").strftime("%Y-%m")
    for m in months:
        st, j = api("/bulk/list", date=m)
        if st != 200:
            print("  一覧の失敗", m, st, str(j)[:120]); continue
        for it in items_of(j):
            k = field(it, "Key", "key", "FileKey")
            if k:
                files[k] = it
    print("ファイルの数", len(files))
    byep = {}
    for k, it in files.items():
        ep = field(it, "Endpoint", "endpoint") or "/".join(k.split("/")[:-1])
        byep.setdefault(ep, [0, 0]); byep[ep][0] += 1; byep[ep][1] += int(field(it, "Size", "size", "ContentLength") or 0)
    for ep, (n, b) in sorted(byep.items()):
        print(f"  {ep}: {n}ファイル・{b / 1e6:,.0f}MB")
    # 2) 一つずつ取ってリリースに上げる
    rows = man.to_dict("records"); n_new = 0
    for k, it in sorted(files.items()):
        mod = str(field(it, "LastModified", "lastModified", "modified") or "")
        if (k, mod) in have:
            continue
        if (time.time() - T0) / 60 > MAX_MIN:
            print("時間切れ。続きは次の実行で"); break
        st, j = api("/bulk/get", key=k)
        u = url_of(j)
        if st != 200 or not u:
            print("  URLの取得に失敗", k, st, str(j)[:120]); continue
        asset = k.replace("/", "__").lstrip("_")
        p = os.path.join(TMP, asset)
        with requests.get(u, stream=True, timeout=600) as r:
            if r.status_code != 200:
                print("  ダウンロード失敗", k, r.status_code); continue
            with open(p, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
        res = sh("gh", "release", "upload", TAG, p, "--clobber")
        os.remove(p)
        if res.returncode != 0:
            print("  リリースへの保存に失敗", k, res.stderr[:200]); continue
        rows = [x for x in rows if x["key"] != k]
        rows.append({"key": k, "endpoint": field(it, "Endpoint", "endpoint") or "/".join(k.split("/")[:-1]), "size": str(field(it, "Size", "size") or ""), "modified": mod, "asset": asset})
        n_new += 1
        if n_new % 50 == 0:
            pd.DataFrame(rows).to_csv(MAN, index=False)
            print(f"  {n_new}ファイル保存・{(time.time() - T0) / 60:.0f}分", flush=True)
            for c in (["git", "add", MAN], ["git", "commit", "-q", "-m", f"J-Quants 一括データ 途中経過 {len(rows)}ファイル"], ["git", "pull", "-q", "--rebase", "origin", "main"], ["git", "push", "-q"]):
                sh(*c)
    pd.DataFrame(rows).to_csv(MAN, index=False)
    left = [k for k, it in files.items() if (k, str(field(it, "LastModified", "lastModified", "modified") or "")) not in {(r["key"], r["modified"]) for r in rows}]
    print(f"今回 {n_new}ファイル・合計 {len(rows)}ファイル・のこり {len(left)}ファイル")


if __name__ == "__main__":
    main()
