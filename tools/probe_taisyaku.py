"""日証金（taisyaku.jp）のサイトで、どんなデータが取れるか調べる（ログに出すだけ）。"""
import re, requests
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
for url in ["https://www.taisyaku.jp/","https://www.taisyaku.jp/search/","https://www.taisyaku.jp/search/guide/"]:
    r=requests.get(url,headers=H,timeout=40); r.encoding=r.apparent_encoding
    print("==",url,r.status_code)
    for href,txt in re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',r.text,re.S):
        t=re.sub(r'<[^>]+>|\s+',' ',txt).strip()
        if any(k in t+href for k in ("残高","品貸","貸借","データ","CSV","csv","ダウンロード","一覧","検索","速報","確報","過去")):
            print("  ",t[:40],"→",href)
    for m in re.findall(r'(?:action|src)=["\']([^"\']+)["\']',r.text):
        if "search" in m or "api" in m or ".csv" in m: print("   form/src:",m)
