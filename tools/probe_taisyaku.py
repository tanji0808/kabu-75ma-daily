import re, requests
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
s=requests.Session()
r=s.get("https://www.taisyaku.jp/app/stock/",headers=H,timeout=40); r.encoding=r.apparent_encoding
print("==",r.status_code,len(r.text))
for m in re.findall(r'<form[^>]*>',r.text): print(" form",m[:200])
for m in re.findall(r'<input[^>]*>',r.text)[:40]: print(" input",m[:160])
for m in sorted(set(re.findall(r'["\'](/app/[^"\'\s]+)["\']',r.text)))[:40]: print(" path",m)
for m in sorted(set(re.findall(r'src=["\']([^"\']+\.js[^"\']*)',r.text))): print(" js",m)
r2=s.get("https://www.taisyaku.jp/robots.txt",headers=H,timeout=20); print("robots:",r2.status_code,r2.text[:500])
