"""JPXの公開ファイル（銘柄別信用取引残高PDF・日々公表xlsx・投資部門別xlsx）の中身を確認する（ログに出すだけ）。"""
import io, re, requests, pdfplumber, openpyxl
H = {"User-Agent": "Mozilla/5.0"}
B = "https://www.jpx.co.jp"
r = requests.get(B + "/markets/statistics-equities/margin/tvdivq0000001rnl-att/20261113_mtall.pdf", headers=H, timeout=60)
print("mtall", r.status_code, len(r.content))
with pdfplumber.open(io.BytesIO(r.content)) as pdf:
    print("pages", len(pdf.pages))
    for i in (0, 1, len(pdf.pages) // 2, len(pdf.pages) - 1):
        t = pdf.pages[i].extract_text() or ""
        print(f"----- page {i}\n" + "\n".join(t.splitlines()[:40]))
    # 貸借／制度信用の見出しを探す
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ""
        for ln in t.splitlines():
            if "銘柄" in ln and ("貸借" in ln or "制度信用" in ln) and "（" in ln:
                print("HEAD", i, ln[:120])
r = requests.get(B + "/markets/statistics-equities/margin/tvdivq0000001r92-att/20261119_mtdaily.xlsx", headers=H, timeout=60)
print("mtdaily", r.status_code, len(r.content))
wb = openpyxl.load_workbook(io.BytesIO(r.content), read_only=True, data_only=True)
for ws in wb.worksheets:
    print("sheet", ws.title)
    for n, row in enumerate(ws.iter_rows(values_only=True)):
        if n < 14 or n % 50 == 0:
            print(n, row[:16])
idx = requests.get(B + "/markets/statistics-equities/investor-type/index.html", headers=H, timeout=60).text
links = re.findall(r'href="([^"]*stock_1_w_\d+_\d+\.xlsx)"', idx)
print("investor links", links[:5])
r = requests.get(B + links[0], headers=H, timeout=60)
wb = openpyxl.load_workbook(io.BytesIO(r.content), read_only=True, data_only=True)
for ws in wb.worksheets:
    print("sheet", ws.title)
    for n, row in enumerate(ws.iter_rows(values_only=True)):
        if n < 70:
            print(n, [x for x in row[:14]])
