#!/usr/bin/env python3
"""
JPX（日本取引所）の公開ファイルから、空売りの判定に使うデータを毎日ためる（J-Quantsのプランに関係なく使える）。

  data/jpx/margin.csv.gz   : 銘柄別信用取引残高（毎営業日のPDF。2026年9月から全銘柄が毎日）
                             Date（申込日）, Code（5桁）, ShrtVol, LongVol（株数・一般＋制度）, Kind（貸=貸借銘柄／制=制度信用銘柄／他）
  data/jpx/alert.csv.gz    : 日々公表銘柄等信用取引残高（規制・日々公表・日証金の注意喚起などに載った銘柄。毎営業日のExcel）
                             Date（申込日）, Code, Flags
  data/jpx/investor.csv.gz : 投資部門別売買状況（週ごと・東証名証の二市場）
                             StDate, EnDate, FrgnSell, FrgnBuy, FrgnBal（千円。J-Quants の FrgnBal と同じ単位）
JPXのページには直近の数日分しか載らないので、毎日動かしてためていく。
JPX_CHECK=1 のときは、J-Quantsで取ったデータ（data/jq/）と同じ日の値を比べてログに出す。
"""
from __future__ import annotations

import io
import os
import re
import sys
from datetime import date, timedelta

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (kabu-75ma-daily)"}
B = "https://www.jpx.co.jp"
OUT = "data/jpx"
DAYS = int(os.environ.get("JPX_DAYS", "12"))


def get(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=H, timeout=90)
        if r.status_code == 200 and len(r.content) > 1000:
            return r.content
    except Exception as e:
        print(f"  {url}: {e}")
    return None


def load(name: str, cols: list[str]) -> pd.DataFrame:
    p = os.path.join(OUT, name)
    return pd.read_csv(p, dtype=str) if os.path.exists(p) else pd.DataFrame(columns=cols)


def save(df: pd.DataFrame, name: str, keys: list[str]) -> None:
    os.makedirs(OUT, exist_ok=True)
    df = df.drop_duplicates(keys, keep="last").sort_values(keys)
    df.to_csv(os.path.join(OUT, name), index=False, compression="gzip")


def num(tokens: list[str]) -> list[float | None]:
    """「▲ 2,400」を −2400 に。「0.0%」「*」「-」も1つの値として数える。"""
    out, neg = [], False
    for t in tokens:
        if t == "▲":
            neg = True
            continue
        t2 = t.replace(",", "").rstrip("%")
        try:
            v = float(t2)
            out.append(-v if neg else v)
        except ValueError:
            out.append(None)
        neg = False
    return out


# ------------------------------------------------------------------ 銘柄別信用取引残高（PDF）
RE_SHS = re.compile(r"(?:^|\s)([0-9][0-9A-Z]{3}0)\s?((?:JP|US|GB|IE|LU)[0-9A-Z]{10})株数 Shs\. (.*)$")
RE_KIND = re.compile(r"(Loan|Margin|Other) ([0-9][0-9A-Z]{3}0) (?:JP|US|GB|IE|LU)[0-9A-Z]{10}金額")
KIND = {"Loan": "貸", "Margin": "制", "Other": "他"}


def parse_mtall(pdf_bytes: bytes, d: str) -> list[dict]:
    import pdfplumber
    rows, kind = {}, {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            for ln in (p.extract_text() or "").splitlines():
                m = RE_SHS.search(ln)
                if m:
                    v = num(m.group(3).split())
                    if len(v) >= 6 and v[0] is not None and v[3] is not None:
                        rows[m.group(1)] = {"Date": d, "Code": m.group(1), "ShrtVol": v[0], "LongVol": v[3]}
                    continue
                k = RE_KIND.search(ln)
                if k:
                    kind[k.group(2)] = KIND[k.group(1)]
    for c, r in rows.items():
        r["Kind"] = kind.get(c, "")
    return list(rows.values())


def fetch_margin() -> None:
    old = load("margin.csv.gz", ["Date", "Code", "ShrtVol", "LongVol", "Kind"])
    have = set(old["Date"])
    new = []
    for i in range(1, DAYS + 1):
        d = date.today() - timedelta(days=i)
        if d.weekday() >= 5 or d.isoformat() in have:
            continue
        b = get(f"{B}/markets/statistics-equities/margin/tvdivq0000001rnl-att/{d:%Y%m%d}_mtall.pdf")
        if not b:
            continue
        rows = parse_mtall(b, d.isoformat())
        print(f"  銘柄別信用取引残高 {d}: {len(rows)}銘柄（貸借 {sum(r['Kind'] == '貸' for r in rows)}）")
        new += rows
    if new:
        save(pd.concat([old, pd.DataFrame(new).astype(str)], ignore_index=True), "margin.csv.gz", ["Date", "Code"])


# ------------------------------------------------------------------ 日々公表銘柄等（Excel）
def fetch_alert() -> None:
    import openpyxl
    old = load("alert.csv.gz", ["Date", "Code", "Flags"])
    have = set(old["Date"])
    new = []
    for i in range(1, DAYS + 1):
        d = date.today() - timedelta(days=i)
        if d.weekday() >= 5 or d.isoformat() in have:
            continue
        b = get(f"{B}/markets/statistics-equities/margin/tvdivq0000001r92-att/{d:%Y%m%d}_mtdaily.xlsx")
        if not b:
            continue
        wb = openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)
        n = 0
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                if len(row) > 7 and isinstance(row[6], str) and re.fullmatch(r"[0-9][0-9A-Z]{3}0", row[6] or ""):
                    flags = "".join(str(x) for x in row[1:3] if x)
                    new.append({"Date": d.isoformat(), "Code": row[6], "Flags": flags})
                    n += 1
        # 載っている銘柄が0の日も「取れた日」として残す
        if n == 0:
            new.append({"Date": d.isoformat(), "Code": "", "Flags": ""})
        print(f"  日々公表銘柄等 {d}: {n}銘柄")
    if new:
        save(pd.concat([old, pd.DataFrame(new).astype(str)], ignore_index=True), "alert.csv.gz", ["Date", "Code"])


# ------------------------------------------------------------------ 投資部門別（Excel）
def parse_investor(b: bytes) -> dict | None:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    col = None
    for r in rows[:8]:                         # 「海外投資家」の見出しがある列 = 売
        for j, x in enumerate(r):
            if isinstance(x, str) and "海外投資家" in x:
                col = j
                break
        if col is not None:
            break
    if col is None:
        return None
    for i, r in enumerate(rows):
        if len(r) > 1 and isinstance(r[1], str) and r[1].startswith("二市場"):
            v = rows[i + 1]                    # 次の行が金額（千円）
            if isinstance(v[2], str) and "金額" in v[2]:
                return {"FrgnSell": v[col], "FrgnBuy": v[col + 1], "FrgnBal": v[col + 2]}
    return None


def fetch_investor() -> None:
    old = load("investor.csv.gz", ["StDate", "EnDate", "FrgnSell", "FrgnBuy", "FrgnBal"])
    b = get(B + "/markets/statistics-equities/investor-type/index.html")
    if not b:
        print("  投資部門別のページが取れませんでした")
        return
    links = re.findall(r'href="([^"]*stock_1_w_(\d{8})_(\d{8})\.xlsx)"', b.decode("utf-8", "ignore"))
    new = []
    for url, s, e in links:
        st, en = f"{s[:4]}-{s[4:6]}-{s[6:]}", f"{e[:4]}-{e[4:6]}-{e[6:]}"
        if en in set(old["EnDate"]):
            continue
        x = get(B + url)
        r = parse_investor(x) if x else None
        if r:
            new.append({"StDate": st, "EnDate": en, **r})
            print(f"  投資部門別 {st}〜{en}: 海外投資家 差引 {float(r['FrgnBal']) / 1e5:+,.0f}億円")
        else:
            print(f"  投資部門別 {st}〜{en}: 読めませんでした")
    if new:
        save(pd.concat([old, pd.DataFrame(new).astype(str)], ignore_index=True), "investor.csv.gz", ["EnDate"])


# ------------------------------------------------------------------ J-Quants と比べる
def check() -> None:
    print("===== J-Quants との照合")
    m = load("margin.csv.gz", [])
    if os.path.exists("data/jq/margin_interest.csv.gz") and len(m):
        jq = pd.read_csv("data/jq/margin_interest.csv.gz", dtype=str, usecols=["Date", "Code", "LongVol", "ShrtVol"])
        for d in sorted(set(m.Date) & set(jq.Date))[-3:]:
            a = m[m.Date == d].set_index("Code")
            b = jq[jq.Date == d].set_index("Code")
            c = a.join(b, rsuffix="_jq", how="inner")
            same = (c.LongVol.astype(float) == c.LongVol_jq.astype(float)).mean()
            same_s = (c.ShrtVol.astype(float) == c.ShrtVol_jq.astype(float)).mean()
            print(f"  信用残 {d}: JPX {len(a)}銘柄・J-Quants {len(b)}銘柄・共通 {len(c)}、買残一致 {same:.1%}・売残一致 {same_s:.1%}")
            bad = c[c.LongVol.astype(float) != c.LongVol_jq.astype(float)].head(5)
            if len(bad):
                print(bad.to_string())
    if os.path.exists("data/jq/master.csv") and len(m):
        jm = pd.read_csv("data/jq/master.csv", dtype=str)
        last = m[m.Date == m.Date.max()].set_index("Code")
        c = last.join(jm.set_index("Code")[["MrgnNm", "MktNm"]], how="inner")
        c = c[c.MktNm == "プライム"]
        print("  貸借の区分（プライム）JPX × J-Quants\n" + pd.crosstab(c.Kind, c.MrgnNm).to_string())
    a = load("alert.csv.gz", [])
    if os.path.exists("data/jq/margin_alert.csv.gz") and len(a):
        jq = pd.read_csv("data/jq/margin_alert.csv.gz", dtype=str, usecols=["PubDate", "AppDate", "Code"])
        for d in sorted(set(a.Date))[-3:]:
            x = set(a[(a.Date == d) & (a.Code != "")].Code)
            y = set(jq[jq.AppDate == d].Code)
            if y:
                print(f"  日々公表等 {d}: JPX {len(x)}・J-Quants {len(y)}・共通 {len(x & y)}")
    inv = load("investor.csv.gz", [])
    if os.path.exists("data/jq/investor_types.csv.gz") and len(inv):
        jq = pd.read_csv("data/jq/investor_types.csv.gz")
        jq = jq[jq.Section == "TokyoNagoya"].set_index("EnDate")
        for r in inv.itertuples():
            if r.EnDate in jq.index:
                print(f"  海外投資家 {r.EnDate}: JPX {float(r.FrgnBal):,.0f}・J-Quants {float(jq.loc[r.EnDate, 'FrgnBal']):,.0f}")


def main() -> None:
    for f in (fetch_margin, fetch_alert, fetch_investor):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__} に失敗: {e}", file=sys.stderr)
    if os.environ.get("JPX_CHECK") == "1":
        check()


if __name__ == "__main__":
    main()
