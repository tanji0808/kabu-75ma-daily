"""IRトレード（2026-11-27 採用）：決算・業績予想の修正ごとの「良いニュース」の判定。tools/ir_events.py（10年の検証）と同じ計算。
良いニュース：上方修正+15%以上／決算で今期予想を+5%以上引き上げ／本決算で来期予想が前期実績の+20%以上（営業利益。無ければ純利益）"""


def num(x):
    try:
        x = float(x)
        return x if x == x else None
    except Exception:  # noqa
        return None


def events_from_recs(recs: list[dict]) -> list[dict]:
    """1銘柄の開示（古い順・fins_summary の行）から、開示ごとの内容の数字を作る。"""
    out = []
    last_f, hist = {}, {}
    for r in recs:
        dt, typ = r["DiscDate"], str(r["DocType"])
        fy = r.get("CurFYSt")
        op, np_ = num(r["OP"]), num(r["NP"])
        fop, fnp, fdiv = num(r["FOP"]), num(r["FNP"]), num(r["FDivAnn"])
        ev = {"date": dt, "time": str(r["DiscTime"])[:5], "kind": None}
        prev = last_f.get(fy, {})
        if "FinancialStatements" in typ:
            pt = r["CurPerType"]
            ev["kind"] = "決算" + str(pt)
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
            else:
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
        out.append(ev)
    return out


def good_news(ev: dict) -> str | None:
    """良いニュースなら、その中身（表示用）を返す。"""
    k = ev.get("kind") or ""
    if k == "業績修正" and (ev.get("rev") or -9) >= 0.15:
        return f'業績予想の上方修正 +{ev["rev"] * 100:.0f}%'
    if k.startswith("決算") and (ev.get("fchg") or -9) >= 0.05:
        return f'決算で今期予想を +{ev["fchg"] * 100:.0f}% 引き上げ'
    if k == "決算FY" and (ev.get("guide") or -9) >= 0.20:
        return f'本決算の来期予想 +{ev["guide"] * 100:.0f}%（前期実績比）'
    return None
