"""日証金（taisyaku.jp）の銘柄別の貸借取引残高・品貸料を、日付ごとにCSVで取ってためる。
data/taisyaku/daily.csv.gz に追記。取得済みの日は飛ばす。サイトに負担をかけないよう、1日分ごとに数秒待つ。
TS_START（既定 2022-06-01）から今日まで。データがない日（休日など）は飛ばす。"""
from __future__ import annotations
import io, json, os, re, time
from datetime import date, timedelta
import pandas as pd, requests

H = {"User-Agent": "Mozilla/5.0 (kabu-75ma-daily; daily research)"}
B = "https://www.taisyaku.jp/app/stock/"
OUT = "data/taisyaku/daily.csv.gz"
DONE = "data/taisyaku/dates.json"
WAIT = float(os.environ.get("TS_WAIT", "3"))
MAX_MIN = float(os.environ.get("TS_MAX_MINUTES", "100"))


def day(s: requests.Session, d: date) -> pd.DataFrame | None:
    r = s.get(B, headers=H, timeout=60)
    tok = re.search(r'name="csrf_test_name" value="([^"]+)"', r.text).group(1)
    ymd = f"{d.year} / {d.month:02d} / {d.day:02d}"
    r = s.post(B + "search", data={"csrf_test_name": tok, "mgrCd[]": "", "mgrMei": "", "mkYmd": ymd, "submit": "検索する", "favorites": ""},
               headers=H, timeout=90)
    r.encoding = r.apparent_encoding
    if "該当するデータはありません" in r.text:
        return None
    c = s.get(B + "csv", headers=H, timeout=120)
    if c.status_code != 200 or len(c.content) < 200:
        return None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(c.content), encoding=enc, dtype=str)
            break
        except Exception:
            df = None
    if df is None or not len(df):
        return None
    df.insert(0, "申込日", d.isoformat())
    return df


def main():
    os.makedirs("data/taisyaku", exist_ok=True)
    done = json.load(open(DONE)) if os.path.exists(DONE) else {}
    old = pd.read_csv(OUT, dtype=str) if os.path.exists(OUT) else pd.DataFrame()
    s = requests.Session(); t0 = time.time(); new = []
    d = date.fromisoformat(os.environ.get("TS_START", "2022-06-01")); end = date.today()
    first = True
    while d <= end:
        k = d.isoformat()
        if d.weekday() < 5 and (k not in done or k >= (end - timedelta(days=3)).isoformat()):
            try:
                df = day(s, d)
            except Exception as e:
                print(k, "失敗", e); df = None; time.sleep(10)
            done[k] = 0 if df is None else len(df)
            if df is not None:
                new.append(df)
                if first:
                    print("列:", list(df.columns)); print(df.head(2).to_string()); first = False
            time.sleep(WAIT)
            if (time.time() - t0) / 60 > MAX_MIN:
                print("時間切れ：", k); break
        d += timedelta(days=1)
    if new:
        df = pd.concat([old] + new, ignore_index=True).drop_duplicates(keep="last")
        df.to_csv(OUT, index=False, compression="gzip")
        print("保存しました", len(df), "行・データのある日", sum(1 for v in done.values() if v))
    json.dump(dict(sorted(done.items())), open(DONE, "w"), indent=0)


if __name__ == "__main__":
    main()
