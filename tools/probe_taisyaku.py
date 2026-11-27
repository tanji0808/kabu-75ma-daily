import re, requests
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
s=requests.Session()
r=s.get("https://www.taisyaku.jp/app/stock/",headers=H,timeout=40)
tok=re.search(r'name="csrf_test_name" value="([^"]+)"',r.text).group(1)
def search(ymd, code=""):
    d={"csrf_test_name":tok,"mgrCd[]":code,"mgrMei":"","mkYmd":ymd,"submit":"検索する","favorites":""}
    r=s.post("https://www.taisyaku.jp/app/stock/search",data=d,headers=H,timeout=60); r.encoding=r.apparent_encoding
    t=re.sub(r'<script.*?</script>','',r.text,flags=re.S)
    rows=re.findall(r'<tr[^>]*>(.*?)</tr>',t,re.S)
    cells=[re.sub(r'<[^>]+>|\s+',' ',x).strip()[:160] for x in rows[:4]]
    csv=sorted(set(re.findall(r'href=["\']([^"\']*csv[^"\']*)',r.text,re.I)))
    msg=re.findall(r'(該当[^<]{0,40}|件数[^<]{0,40}|\d+\s*件)',t)[:5]
    print("==",ymd,code,"行",len(rows),msg[:1], "|", cells[2] if len(cells)>2 else "")
for d in ["2026 / 11 / 02","2025 / 06 / 02","2024 / 06 / 03","2023 / 06 / 01","2022 / 06 / 01","2021 / 06 / 01"]:
    search(d,"7203")
