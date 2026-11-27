"""ミニ株版の「今日の売買」ページ（mini.html）と台帳（data/forward_mini.json）。2026-11-27 作成。

単元未満株（ミニ株）で、注文はすべて「寄り付き」だけで回す版。逆指値・指値は使わない。
  買い  : 候補は個人の「今日の売買」と同じ（同じ条件・同じ見送り条件）。翌朝の寄りで買う（リバウンドも指値ではなく寄り）。
          窓あけ下げは「急落の朝」に寄りで買ったものとして、その日の夕方に記録する。
  売り  : 損切り・利確・新高値の追いかけは、すべてその日の終値で判定し、翌朝の寄りで売る。
  空売り・年末の反発はなし。資金 100万円（MINI_CAPITAL で変更可）。株数は1株単位。
10年の検証（100万円・単利）：+355万円・最大の落ち込み −13万円・負けた年なし。
"""
from __future__ import annotations

import html as h
import json
import os

import numpy as np
import pandas as pd

MINI_CAPITAL = int(os.environ.get("MINI_CAPITAL", "1000000"))
LEDGER = "data/forward_mini.json"
UNIT = {"buy": None, "newhigh": 0.04, "ipo": 0.05, "ir": 0.05, "rebound": 0.05, "gapdown": 0.05, "kenri": 0.20}
MAX_N = {"buy": 10, "newhigh": 12, "ipo": 10, "ir": 20, "rebound": 10, "gapdown": 10, "kenri": 1}
ORDER = ["gapdown", "kenri", "rebound", "ipo", "newhigh", "ir", "buy"]
NAME = {"buy": "75日線の買い", "newhigh": "新高値", "ipo": "IPO", "ir": "IRトレード", "rebound": "リバウンド", "gapdown": "窓あけ下げ", "kenri": "権利取り前"}
COST = 0.002
# 売り方（すべて終値で判定 → 翌朝の寄りで売る）
RULE = {"newhigh": dict(stop=0.06, arm=0.20, trail=0.20, hold=120),
        "ir": dict(stop=0.10, hold=60),
        "ipo": dict(stop=0.10, tp=0.30, hold=120),
        "rebound": dict(stop=0.10, hold=3),
        "gapdown": dict(stop=0.10, tp=0.10, hold=2)}


def _frame(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df, [d.strftime("%Y-%m-%d") for d in df.index]


def _eval_one(s: dict, df: pd.DataFrame) -> dict:
    r = dict(s)
    df, dates = _frame(df)
    if s["date"] not in dates:
        r["status"] = "株価なし"; return r
    i = dates.index(s["date"])
    e = i if s["side"] == "gapdown" else i + 1
    if e >= len(dates):
        r["status"] = "明日の寄りで買い"; return r
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    entry = float(s.get("entry_override") or o[e])
    p = RULE[s["side"]]
    stop, tp, arm, trail, hold = p["stop"], p.get("tp"), p.get("arm"), p.get("trail"), p["hold"]
    if s["side"] == "rebound":                                  # 利確は合図の日の25日線の−10%
        m25 = float(np.mean(c[max(0, i - 24):i + 1])); tp_px = max(m25 * 0.90, entry * 1.001)
    else:
        tp_px = entry * (1 + tp) if tp else None
    line = entry * (1 - stop); hc = 0.0
    sh = s.get("shares") or 0
    amt = sh * entry
    for j in range(e, min(len(c), e + hold)):
        hc = max(hc, c[j])
        if arm is not None and hc >= entry * (1 + arm):
            line = max(line, entry, hc * (1 - trail))
        why = None
        if c[j] <= line:
            why = "損切り" if line < entry * (1 - stop) + 1e-9 else "追いかけのラインを割った"
        elif tp_px and c[j] >= tp_px:
            why = "利確"
        elif j == e + hold - 1:
            why = f"{hold}営業日で手じまい"
        if why:
            if j + 1 < len(c):
                ret = o[j + 1] / entry - 1
                r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[e], exit_date=dates[j + 1], ret=round(ret * 100, 2),
                         why=why + "（翌朝の寄りで売り）", pnl=round(amt * ret - amt * COST))
            else:
                ret = c[j] / entry - 1
                r.update(status="保有中", entry=round(entry, 1), entry_date=dates[e], now=round(c[j], 1), ret=round(ret * 100, 2),
                         pnl=round(amt * ret - amt * COST), sell_tomorrow=True, order=f"<b>明日の寄りで売る</b>（{why}）")
            return r
    ret = c[-1] / entry - 1
    note = [f"終値が {line:,.0f}円 以下なら翌朝売り"]
    if tp_px:
        note.append(f"終値が {tp_px:,.0f}円 以上なら翌朝売り")
    r.update(status="保有中", entry=round(entry, 1), entry_date=dates[e], now=round(c[-1], 1), ret=round(ret * 100, 2),
             pnl=round(amt * ret - amt * COST), order="・".join(note))
    return r


def evaluate_mini(led: dict, prices: dict) -> list[dict]:
    import forward_test2 as ft
    rows = []
    std = {"start": led["start"], "signals": [s for s in led["signals"] if s["side"] in ("buy", "kenri")]}   # 75日線の買い・権利取り前は元から寄り・引けだけ
    if std["signals"]:
        for r in ft.evaluate(std, prices):
            if r["side"] == "buy" and r.get("shares"):
                pass
            rows.append(r)
    for s in led["signals"]:
        if s["side"] in RULE:
            df = prices.get(s["code"])
            rows.append(_eval_one(s, df) if df is not None else dict(s, status="株価なし"))
    return rows


def run_mini(prices: dict, cands: dict, info: dict, data_date: str, out_dir: str) -> None:
    import forward_test2 as ft
    led = json.load(open(LEDGER, encoding="utf-8")) if os.path.exists(LEDGER) else {"start": data_date, "capital": MINI_CAPITAL, "signals": []}
    prev = evaluate_mini(led, prices)
    eq = MINI_CAPITAL + sum(r.get("pnl", 0) for r in prev if r.get("status") == "決済済み")
    open_ = [r for r in prev if r.get("status") == "保有中" or str(r.get("status", "")).startswith("明日")]
    used = sum((r.get("amount") or 0) for r in open_)
    held = {r["code"] for r in open_}
    n = {k: sum(1 for r in open_ if r["side"] == k) for k in MAX_N}
    todays = []
    for side in ORDER:
        for x0 in cands.get(side, []):
            x = {k: v for k, v in x0.items() if k not in ("slot", "why", "shares", "amount", "want", "cap")}
            x["side"] = side
            px = float(x.get("last_close") or x.get("close") or 0)       # リバウンドは指値ではなく寄りで買うので今日の終値で計算
            x["close"] = px
            want = x0.get("lot", 1.0) * 1e6 * ft.UNIT_BUY * eq / 1e7 if side == "buy" else eq * UNIT[side]
            sh = int(want // px) if px else 0
            x.update(want=want, shares=sh, amount=round(sh * px))
            if side == "buy" and ft.UNIT_BUY == 0:
                x["slot"], x["why"] = "見送り（75日線の買いは休止中）", ""
            elif x0.get("mult") == 0:
                x["slot"], x["why"] = "見送り（相場の条件）", x0.get("why_mkt", "")
            elif not sh:
                x["slot"], x["why"] = "1株に届かない", ""
            elif x["code"] in held:
                x["slot"], x["why"] = "すでに保有中", ""
            elif n[side] >= MAX_N[side]:
                x["slot"], x["why"] = "枠がいっぱい", f"{NAME[side]}は同時に{MAX_N[side]}銘柄まで"
            elif used + x["amount"] > eq:
                x["slot"], x["why"] = "枠がいっぱい", f"資金の残り {max(eq - used, 0) / 1e4:,.1f}万円"
            else:
                x["slot"], x["why"] = "枠あり", ""
                n[side] += 1; used += x["amount"]; held.add(x["code"])
            todays.append(x)
    have = {(s["date"], s["side"], s["code"]) for s in led["signals"]}
    for x in todays:
        if x.get("date", "") < led["start"]:
            continue
        if x["slot"] == "枠あり" and (x["date"], x["side"], x["code"]) not in have:
            led["signals"].append({k: v for k, v in x.items() if k not in ("slot", "why")})
    os.makedirs("data", exist_ok=True)
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = evaluate_mini(led, prices)
    page(rows, todays, info, eq, data_date, led["start"], os.path.join(out_dir, "mini.html"))
    print(f"ミニ株ページ：明日の買い {sum(x['slot'] == '枠あり' and x['side'] != 'gapdown' for x in todays)}件・明日売る {sum(1 for r in rows if r.get('sell_tomorrow'))}件・台帳 {len(led['signals'])}件")


def page(rows, todays, info, eq, data_date, start, path):
    man = lambda v: f"{v / 1e4:,.1f}万円"
    yen = lambda v: f"{v / 1e4:+,.1f}万円"
    link = lambda x: f'<a href="https://finance.yahoo.co.jp/quote/{x["code"]}.T/chart" target="_blank" rel="noopener">{x["code"]} {h.escape(str(x.get("name", "")))}</a>'

    def table(items, cols):
        if not items:
            return '<p class="none">なし</p>'
        head = "".join(f"<th>{h.escape(c)}</th>" for c, _ in cols)
        body = "".join("<tr>" + "".join(f"<td>{f(x)}</td>" for _, f in cols) + "</tr>" for x in items)
        return f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>'
    buy = [x for x in todays if x["slot"] == "枠あり" and x["side"] != "gapdown"]
    skip = [x for x in todays if x["slot"] != "枠あり"]
    hold = [r for r in rows if r.get("status") == "保有中"]
    sell = [r for r in hold if r.get("sell_tomorrow")]
    closed = sorted([r for r in rows if r.get("status") == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    bcols = [("手法", lambda x: NAME[x["side"]]), ("銘柄", link), ("今日の終値", lambda x: f'{x.get("close", 0):,.1f}円'),
             ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("金額の目安", lambda x: man(x.get("amount", 0)))]
    hcols = [("手法", lambda x: NAME.get(x["side"], x["side"])), ("銘柄", link), ("買った日", lambda x: x.get("entry_date", "")),
             ("株数", lambda x: f'{x.get("shares", 0):,}株'), ("買値", lambda x: f'{x.get("entry", 0):,.1f}円'), ("今", lambda x: f'{x.get("now", 0):,.1f}円'),
             ("損益", lambda x: f'{x.get("ret", 0):+.1f}%（{yen(x.get("pnl", 0))}）'), ("売る条件", lambda x: x.get("order") or "毎日の判定どおり")]
    ccols = [("手法", lambda x: NAME.get(x["side"], x["side"])), ("銘柄", link), ("買った日", lambda x: x.get("entry_date", "")),
             ("売った日", lambda x: x.get("exit_date", "")), ("損益", lambda x: f'{x.get("ret", 0):+.1f}%（{yen(x.get("pnl", 0))}）'), ("理由", lambda x: h.escape(x.get("why", "")))]
    scols = [("手法", lambda x: NAME[x["side"]]), ("銘柄", link), ("理由", lambda x: h.escape(x["slot"] + ("：" + x["why"] if x.get("why") else "")))]
    cl = sum(r.get("pnl", 0) for r in closed); op = sum(r.get("pnl", 0) for r in hold)
    win = round(sum(r.get("pnl", 0) > 0 for r in closed) / len(closed) * 100) if closed else None
    gi = info.get("gd") or {}
    gd_txt = ("<p class='warn'>窓あけ下げの対象期間です。明日の朝、大きく下げて始まりそうなら、個人の「今日の売買」の窓あけ下げの一覧から、"
              "気配が−5%以下の銘柄を下げの大きい順に寄りで買う（1銘柄 資金の7%）。売りは終値で判定して翌朝の寄り。</p>") if gi.get("active_tomorrow") else ""
    html = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ミニ株の売買（寄り付きだけ版）</title><style>
body{{font-family:'Noto Sans JP',system-ui,sans-serif;margin:0;background:#f5f4f0;color:#1d1d1b;font-size:14px;line-height:1.55}}
main{{max-width:1100px;margin:0 auto;padding:16px}}h1{{font-size:18px;margin:4px 0 2px}}h2{{font-size:15px;margin:0 0 8px}}
.card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;padding:12px 14px;margin:10px 0}}.sub{{font-size:12.5px;color:#6b6a65}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}}.stat .k{{font-size:11.5px;color:#6b6a65}}.stat .v{{font-size:17px;font-weight:700}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:5px 7px;border-bottom:1px solid #eeece6;text-align:left;vertical-align:top}}
th{{font-size:12px;color:#6b6a65;font-weight:500;white-space:nowrap}}.none{{color:#999}}.warn{{background:#fff4e5;border:1px solid #f0c27b;border-radius:8px;padding:8px 10px}}a{{color:#1f4e79}}
</style></head><body><main>
<a href="forward2.html" class="sub">← 個人の今日の売買</a>　<a href="houjin.html" class="sub">🏢 法人の売買</a>　<a href="docs.html" class="sub">📘 まとめ資料</a>
<h1>ミニ株の売買（注文は寄り付きだけ）</h1>
<div class="sub">{h.escape(data_date)} 終値時点。記録開始 {h.escape(start)}。逆指値・指値は使わず、買いも売りも翌朝の寄り付き。空売り・年末の反発はなし。</div>
<div class="card"><div class="stats">
<div class="stat"><div class="k">資金（最初 {MINI_CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{eq / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">1回の金額</div><div class="v">新高値 {eq * 0.04 / 1e4:,.1f}万円／ほか {eq * 0.05 / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">決済済みの損益（勝率）</div><div class="v">{yen(cl)}（{win if win is not None else '—'}%）</div></div>
<div class="stat"><div class="k">保有中の含み損益</div><div class="v">{yen(op)}</div></div></div>
<p class="sub">10年の検証（100万円・単利）：+355万円・最大の落ち込み −13万円・負けた年なし。税引き後の複利なら100万円→約1,138万円。</p></div>
{gd_txt}
<div class="card"><h2>明日の寄りで売る（{len(sell)}件）</h2>{table(sell, hcols)}</div>
<div class="card"><h2>明日の寄りで買う（{len(buy)}件）</h2>{table(buy, bcols)}
<p class="sub">ミニ株の「寄付」の注文で買う。株数は今日の終値で計算した目安。</p></div>
<div class="card"><h2>保有中（{len(hold)}件）</h2>{table(hold, hcols)}
<p class="sub">毎日の夕方、売る条件に当たったものは上の「明日の寄りで売る」に出ます。</p></div>
<div class="card"><h2>見送った候補（{len(skip)}件）</h2>{table(skip, scols)}</div>
<div class="card"><h2>決済済み（{len(closed)}件）</h2>{table(closed[:60], ccols)}</div>
</main></body></html>"""
    open(path, "w", encoding="utf-8").write(html)
