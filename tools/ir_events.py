"""IRトレードの材料：J-Quants の開示（決算短信・業績予想の修正・配当予想の修正）ごとに、内容の数字・発表後の値動き・その後の損益を並べる。
出力 backtest/ir_events.csv.gz
内容の数字（営業利益。無ければ純利益）
  yoy   : 前年同期比（累計）            prog : 会社予想に対する進捗 − 経過割合（1Q 25%・2Q 50%・3Q 75%）
  fchg  : この発表で今期予想がどれだけ変わったか   guide: 本決算の来期予想 ÷ 今期実績 − 1
  rev   : 業績予想の修正率（新しい予想 ÷ 前の予想 − 1）   drev : 配当予想の修正率
反応（r＝発表を受けた最初の日：15時までの発表はその日、15時以降は翌営業日）
  gap=寄りの窓、day=その日の騰落、body=寄りから引け、vr=出来高÷20日平均、clv=引けの位置、pre20=発表前20日の騰落、tv20=売買代金
その後（買い。空売りは符号を反対にして使う）
  o{N}  : r の寄りで買って N 営業日後の引けで売る（15時以降の発表＝数字を見て翌朝の寄りで買う）
  n{N}  : r+1 の寄り（反応を1日見てから）で買って N 営業日後の引けで売る
  s8h20 / s8h60 : r+1 の寄りで買い、−8%で損切り、20/60営業日で手じまい"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("K_DELISTED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kabu_75ma_search as k  # noqa: E402
import short_backtest as sb  # noqa: E402
import newhigh_candidates as nh  # noqa: E402

HOLDS = (1, 3, 5, 10, 20, 40, 60)


def num(x):
    try:
        x = float(x)
        return x if x == x else None
    except Exception:  # noqa
        return None


def main():
    f = pd.read_csv("data/jq/fins_summary.csv.gz", dtype=str)
    f["code"] = f.Code.str[:4]
    f = f.sort_values(["code", "DiscDate", "DiscTime"])
    master = k.load_prime_list()
    prices = k.download_prices(master["コード"].tolist(), period="11y")
    out = []
    for code, g in f.groupby("code"):
        df = prices.get(code)
        if df is None or len(df) < 80:
            continue
        df = sb.clean(df); df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, h, l, c, v = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close", "Volume"))
        n = len(c); dates = df.index.strftime("%Y-%m-%d").tolist()
        a = {"o": o, "h": h, "l": l, "c": c, "pivot": np.full(n, np.nan), "dates": dates}
        tv20 = pd.Series(c * v).rolling(20).mean().to_numpy(); va = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        recs = g.to_dict("records")
        last_f = {}      # 年度の始まり → 直近の今期予想（営業利益・純利益・配当）
        hist = {}        # (CurPerType, 年度の始まり) → 実績
        for r in recs:
            dt, typ = r["DiscDate"], str(r["DocType"])
            fy = r.get("CurFYSt")
            op, np_ = num(r["OP"]), num(r["NP"])
            fop, fnp, fdiv = num(r["FOP"]), num(r["FNP"]), num(r["FDivAnn"])
            ev = {"code": code, "date": dt, "time": str(r["DiscTime"])[:5], "kind": None}
            prev = last_f.get(fy, {})
            if "FinancialStatements" in typ:
                pt = r["CurPerType"]
                ev["kind"] = "決算" + pt
                key = "OP" if op is not None and prev.get("useop", True) else "NP"
                val = op if key == "OP" else np_
                fc = fop if key == "OP" else fnp
                try:
                    pfy = str(int(fy[:4]) - 1) + fy[4:]
                except Exception:  # noqa
                    pfy = None
                pv = hist.get((pt, pfy))
                if val is not None and pv is not None and pv > 0:
                    ev["yoy"] = val / pv - 1
                q = {"1Q": 0.25, "2Q": 0.5, "3Q": 0.75}.get(pt)
                if q and val is not None and fc and fc > 0:
                    ev["prog"] = val / fc - q
                pf = prev.get(key)
                if pt != "FY" and fc and pf and pf > 0:
                    ev["fchg"] = fc / pf - 1
                if pt == "FY":
                    nx = num(r["NxFOP"]) if key == "OP" else num(r["NxFNp"])
                    if nx is not None and val and val > 0:
                        ev["guide"] = nx / val - 1
                if val is not None:
                    hist[(pt, fy)] = val
                if pt != "FY":
                    if fop is not None: prev["OP"] = fop
                    if fnp is not None: prev["NP"] = fnp
                    if fdiv is not None: prev["DIV"] = fdiv
                    last_f[fy] = prev
                else:   # 本決算：来期の予想を次の年度の「前の予想」に
                    nfy = str(int(fy[:4]) + 1) + fy[4:] if fy else None
                    last_f[nfy] = {"OP": num(r["NxFOP"]), "NP": num(r["NxFNp"]), "DIV": num(r["NxFDivAnn"])}
            elif typ == "EarnForecastRevision":
                ev["kind"] = "業績修正"
                if fop is not None and prev.get("OP") and prev["OP"] > 0:
                    ev["rev"] = fop / prev["OP"] - 1
                elif fnp is not None and prev.get("NP") and prev["NP"] > 0:
                    ev["rev"] = fnp / prev["NP"] - 1
                if fop is not None: prev["OP"] = fop
                if fnp is not None: prev["NP"] = fnp
                last_f[fy] = prev
            elif typ == "DividendForecastRevision":
                ev["kind"] = "配当修正"
                if fdiv is not None and prev.get("DIV") and prev["DIV"] > 0:
                    ev["drev"] = fdiv / prev["DIV"] - 1
                if fdiv is not None: prev["DIV"] = fdiv
                last_f[fy] = prev
            else:
                continue
            if dt < "2016-12-01":
                continue
            # 反応の日
            i0 = int(np.searchsorted(np.array(dates), dt))
            if i0 >= n:
                continue
            ridx = i0 if (dates[i0] > dt or ev["time"] < "15:00") else i0 + 1
            if dates[i0] > dt:
                ridx = i0
            if ridx < 21 or ridx + 2 >= n:
                continue
            rr = ridx
            ev.update(rdate=dates[rr], gap=o[rr] / c[rr - 1] - 1, day=c[rr] / c[rr - 1] - 1, body=c[rr] / o[rr] - 1,
                      vr=v[rr] / va[rr] if va[rr] and va[rr] > 0 else np.nan,
                      clv=(c[rr] - l[rr]) / (h[rr] - l[rr]) if h[rr] > l[rr] else np.nan,
                      pre20=c[rr - 1] / c[rr - 21] - 1, tv20=tv20[rr - 1], close=c[rr - 1])
            for N in HOLDS:
                if rr + N - 1 < n:
                    ev[f"o{N}"] = c[rr + N - 1] / o[rr] - 1
                if rr + N < n:
                    ev[f"n{N}"] = c[rr + N] / o[rr + 1] - 1
            for nm, hd in (("s8h20", 20), ("s8h60", 60)):
                x = nh.sim(a, rr, 0.08, hold=hd)
                if x:
                    ev[nm] = x["ret"] / 100
            out.append(ev)
    pd.DataFrame(out).to_csv("backtest/ir_events.csv.gz", index=False, compression="gzip")
    print("保存しました", len(out))


if __name__ == "__main__":
    main()
