"""J-Quants の大量保有報告書（/edinet/large-volume-shareholders）を提出日ごとに取って、data/jq/large_holdings.csv.gz にためる。
データは2021年7月1日提出分から。取得済みの日は飛ばす。1行＝1書類×1保有者。"""
from __future__ import annotations
import json, os, sys, time
from datetime import date, timedelta
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jquants_fetch as jf  # noqa: E402

OUT = "data/jq/large_holdings.csv.gz"
DONE = "data/jq/large_holdings_dates.json"
MAX_MIN = float(os.environ.get("JQ_MAX_MINUTES", "60"))


def flat(d: dict) -> list[dict]:
    rows = []
    base = {k: d.get(k) for k in ("DocId", "Code", "IsrName", "DocTypeCode", "SubDate", "SubTime", "RptOblgDate", "ParDocId",
                                  "LargeHldgTypeCode", "DocTitle", "ChgRsn", "TotalShsRatio", "TotalShsRatioLast", "TotalOutStks")}
    for hd in d.get("Hldrs") or [{}]:
        acq = hd.get("AcqDisp") or []
        on_buy = sum(float(a.get("Shs") or 0) for a in acq if str(a.get("MktCode")) == "1" and str(a.get("TxnTypeCode")) == "1")
        on_sell = sum(float(a.get("Shs") or 0) for a in acq if str(a.get("MktCode")) == "1" and str(a.get("TxnTypeCode")) == "2")
        px = [float(a["Price"]) for a in acq if a.get("Price") not in (None, "") and str(a.get("TxnTypeCode")) == "1"]
        rows.append(dict(base, HldrName=hd.get("HldrName"), HldrNameEn=hd.get("HldrNameEn"), HldrEdinetCode=hd.get("HldrEdinetCode"),
                         LargeHldrTypeCode=hd.get("LargeHldrTypeCode"), HldgPurp=(hd.get("HldgPurp") or "")[:300], ImpProp=(hd.get("ImpProp") or "")[:300],
                         ShsHeld=hd.get("ShsHeld"), ShsRatio=hd.get("ShsRatio"), ShsRatioLast=hd.get("ShsRatioLast"),
                         OwnFund=hd.get("OwnFund"), TotalBrw=hd.get("TotalBrw"), TotalFund=hd.get("TotalFund"),
                         OnMktBuy=on_buy, OnMktSell=on_sell, BuyPxMax=max(px) if px else None, NAcq=len(acq)))
    return rows


def main():
    t0 = time.time()
    done = json.load(open(DONE)) if os.path.exists(DONE) else {}
    old = pd.read_csv(OUT, dtype=str) if os.path.exists(OUT) else pd.DataFrame()
    new = []
    d = date(2021, 7, 1); end = date.today()
    while d <= end:
        ds = d.isoformat()
        if d.weekday() < 5 and (ds not in done or ds >= (end - timedelta(days=3)).isoformat()):
            st, rows, err = jf.call("/edinet/large-volume-shareholders", date=ds)
            if st != 200:
                print(ds, "失敗", st, err); 
                if st in (401, 403):
                    break
            else:
                for r in rows:
                    new += flat(r)
                done[ds] = len(rows)
            if (time.time() - t0) / 60 > MAX_MIN:
                print("時間切れ：", ds); break
        d += timedelta(days=1)
    df = pd.concat([old, pd.DataFrame(new).astype(str)], ignore_index=True) if new else old
    if len(df):
        df = df.drop_duplicates(["DocId", "HldrEdinetCode", "HldrName"], keep="last")
        df.to_csv(OUT, index=False, compression="gzip")
    json.dump(dict(sorted(done.items())), open(DONE, "w"), indent=0)
    print("保存しました", len(df), "行・", len(done), "日")


if __name__ == "__main__":
    main()
