"""J-Quants API の動作確認：APIキーが使えるか、各データの項目名と取れる期間を表示する（中身の数字は表示しない）。"""
import json
import os
import sys
import time

import requests

KEY = os.environ.get("JQUANTS_API_KEY", "").strip()
if not KEY:
    sys.exit("JQUANTS_API_KEY が設定されていません")
H = {"x-api-key": KEY}
BASE = "https://api.jquants.com/v2"


def get(path, **params):
    r = requests.get(BASE + path, headers=H, params=params, timeout=60)
    print(f"\n== {path} {params} → HTTP {r.status_code}")
    try:
        j = r.json()
    except Exception:
        print(r.text[:300])
        return None
    if r.status_code != 200:
        print(json.dumps(j, ensure_ascii=False)[:400])
        return None
    for k, v in j.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)}件")
            if v and isinstance(v[0], dict):
                print("  項目:", list(v[0].keys()))
                ds = [x.get("DiscDate") or x.get("DisclosedDate") or x.get("Date") for x in v]
                ds = [d for d in ds if d]
                if ds:
                    print("  日付の範囲:", min(ds), "〜", max(ds))
        else:
            print(f"  {k}: {str(v)[:80]}")
    return j


import os as _os
for _p in _os.environ.get("PROBE", "/fins/summary").split(","):
    _p, _, _d = _p.partition("@")      # 例 /fins/summary@20170105 → その日の分を確認
    get(_p, **({"date": _d} if _d else {"code": "7203"}))
    time.sleep(3)
