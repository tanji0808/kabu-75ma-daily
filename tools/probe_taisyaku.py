import re, requests
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
for url in ["https://www.taisyaku.jp/search/guide/","https://www.taisyaku.jp/about/report/","https://www.taisyaku.jp/app/stock/top","https://www.taisyaku.jp/app/stock/detail?code=7203"]:
    r=requests.get(url,headers=H,timeout=40); r.encoding=r.apparent_encoding
    t=re.sub(r'<script.*?</script>|<style.*?</style>','',r.text,flags=re.S); t=re.sub(r'<[^>]+>',' ',t); t=re.sub(r'\s+',' ',t)
    print("==",url,r.status_code); 
    for kw in ("CSV","csv","ダウンロード","過去","期間","遡","日分","ヶ月","か月","年分"):
        for m in re.finditer(kw,t):
            print("  …",t[max(0,m.start()-80):m.start()+80])
    print("  links:", sorted(set(re.findall(r'href=["\'](/app/[^"\']+)',r.text)))[:30])
