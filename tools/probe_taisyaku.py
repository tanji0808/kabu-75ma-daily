"""日証金（taisyaku.jp）のサイトで、どんなデータが取れるか調べる（ログに出すだけ）。"""
import re, requests
H={"User-Agent":"Mozilla/5.0 (kabu-75ma-daily)"}
seen=set()
def links(url):
    try:
        r=requests.get(url,headers=H,timeout=40); r.encoding=r.apparent_encoding
    except Exception as e:
        print("取れない",url,e); return []
    print("==",url,r.status_code,len(r.text))
    out=re.findall(r'href=["\']([^"\']+)["\']',r.text)
    return [u if u.startswith("http") else requests.compat.urljoin(url,u) for u in out]
start=["https://www.taisyaku.jp/","https://www.taisyaku.jp/data/","https://www.taisyaku.jp/search/","https://www.taisyaku.jp/sys-list/"]
q=list(start)
while q and len(seen)<60:
    u=q.pop(0)
    if u in seen or "taisyaku.jp" not in u: continue
    seen.add(u)
    for l in links(u):
        if re.search(r'\.(csv|zip|xls|xlsx|pdf)(\?|$)',l,re.I):
            print("  ファイル:",l)
        elif "taisyaku.jp" in l and l not in seen and not re.search(r'\.(jpg|png|gif|css|js)$',l) and len(seen)<60:
            if any(k in l for k in ("data","search","list","zandaka","shina","balance","pcsl","history","csv")):
                q.append(l)
