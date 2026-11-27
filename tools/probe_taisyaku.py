import re, requests, io
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
s=requests.Session()
r=s.get("https://www.taisyaku.jp/app/stock/",headers=H,timeout=40)
tok=re.search(r'name="csrf_test_name" value="([^"]+)"',r.text).group(1)
d={"csrf_test_name":tok,"mgrCd[]":"","mgrMei":"","mkYmd":"2026 / 11 / 26","submit":"検索する","favorites":""}
r=s.post("https://www.taisyaku.jp/app/stock/search",data=d,headers=H,timeout=60); r.encoding=r.apparent_encoding
for m in re.finditer(r'<a[^>]*csv[^>]*>',r.text,re.I): print("A:",m.group(0)[:300])
for m in re.finditer(r'.{0,200}csv.{0,200}',r.text,re.I|re.S): print("CTX:",re.sub(r'\s+',' ',m.group(0))[:400])
