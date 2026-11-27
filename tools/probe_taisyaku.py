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
    print("==",ymd,code,r.status_code,len(r.text),"行",len(rows),msg,csv[:5]); [print("   ",c) for c in cells]
search("2026 / 11 / 26","7203")
search("2026 / 11 / 26","")
search("2021 / 01 / 05","7203")
search("2017 / 01 / 05","7203")
