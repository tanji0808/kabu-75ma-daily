"""自動売買版（三菱UFJ eスマート証券・kabuステーションAPI）の注文ファイルとページ。2026-11-27 作成。

  data/forward_auto.json : 台帳（個人とは別。最初の資金 300万円＝AUTO_CAPITAL）
  data/auto_orders.json  : 翌営業日にパソコン側のプログラム（bridge/kabus_bridge.py）が出す注文の一覧
  auto.html              : 人が見る確認用ページ

ルールは個人の「今日の売買」と同じ（候補・損切り・利確・追いかけ）。違うのは：
  - 100株単位。1回の金額で100株に届かない時は、1回の金額の1.5倍までなら100株買う（10年の検証で 300万円→+1,214万円・空売りと75日線の買いなし・2026-11-28の1回の金額で）。
  - 最初は空売りと75日線の買いはなし（毎日の細かい判定が多いため。慣れたら足す）。年末の反発もなし。
注文の種類（kabuステーションAPIの言葉）：寄成＝寄りで成行、寄指＝寄りで指値、指値、逆指値（損切り）、引成＝引けで成行。
"""
from __future__ import annotations

import html as h
import json
import os

import numpy as np
import pandas as pd

AUTO_CAPITAL = int(os.environ.get("AUTO_CAPITAL", "3000000"))
LEDGER = "data/forward_auto.json"
ORDERS = "data/auto_orders.json"
SIDES = ("gapdown", "kenri", "rebound", "ipo", "newhigh", "ir")
OVER = 1.5
NAME = {"newhigh": "新高値", "ipo": "IPO", "ir": "IRトレード", "rebound": "リバウンド", "gapdown": "窓あけ下げ", "kenri": "権利取り前"}


def _unit(side):
    import forward_test2 as ft
    return {"newhigh": ft.UNIT_NH, "ipo": ft.UNIT_IPO, "ir": ft.UNIT_IR, "rebound": ft.UNIT_RB, "gapdown": ft.UNIT_GD, "kenri": ft.UNIT_KR}[side]


def _shares(want, px, side):
    if not px:
        return 0
    if side == "kenri":
        return int(want // px)                                          # ETF（1306）は1口から
    sh = int(want // px // 100 * 100)
    if not sh and px * 100 <= want * OVER:
        sh = 100
    return sh


def _tv20(df):
    if df is None or len(df) < 21:
        return None
    c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
    return float(np.mean(c[-20:] * v[-20:]))


def sign(payload: dict) -> str | None:
    """注文ファイルの署名（HMAC-SHA256）。鍵は GitHub の Secrets の AUTO_SIGN_KEY。パソコン側は同じ鍵で確かめる。"""
    import hashlib, hmac
    key = os.environ.get("AUTO_SIGN_KEY", "")
    if not key:
        return None
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()


def _sell_date(r, df):
    """時間で手じまう日（その日の引けで売る）。"""
    import forward_test2 as ft
    idx = pd.to_datetime(df.index).tz_localize(None).normalize()
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    if r["date"] not in dates:
        return None
    i = dates.index(r["date"])
    k = {"newhigh": i + 120, "ipo": i + 120, "ir": i + ft.IR_HOLD, "rebound": i + 1 + ft.RB_HOLD - 1, "gapdown": i + 1}.get(r["side"])
    if r["side"] == "kenri":
        return r.get("sell_date")
    if k is None:
        return None
    if k < len(dates):
        return dates[k]
    return str(np.busday_offset(np.datetime64(dates[-1]), k - (len(dates) - 1), roll="forward"))


def run_auto(prices: dict, cands: dict, info: dict, data_date: str, out_dir: str) -> None:
    import forward_test2 as ft
    led = json.load(open(LEDGER, encoding="utf-8")) if os.path.exists(LEDGER) else {"start": data_date, "capital": AUTO_CAPITAL, "signals": []}
    prev = ft.evaluate(led, prices) if led["signals"] else []
    eq = AUTO_CAPITAL + sum(r.get("pnl", 0) for r in prev if r.get("status") == "決済済み")
    open_ = [r for r in prev if r.get("status") == "保有中" or str(r.get("status", "")).startswith("明日")]
    used = sum(r.get("amount") or 0 for r in open_)
    held = {r["code"] for r in open_}
    _CAP, _cap_add = ft.SIDE_CAP, {}   # 2026-11-28：手法ごとの合計の上限（落ち込み対策）
    n = {k: sum(1 for r in open_ if r["side"] == k) for k in SIDES}
    todays = []
    for side in SIDES:
        for x0 in cands.get(side, []):
            x = {k: v for k, v in x0.items() if k not in ("slot", "why", "shares", "amount", "want", "cap")}
            x["side"] = side
            px = float(x.get("limit") or x.get("close") or 0)
            if side == "kenri" and prices.get("1306") is not None:
                px = float(prices["1306"]["Close"].iloc[-1]); x["close"] = px
            want = min(eq * (ft.UNIT_TM if x0.get("kind") == "monthend" else _unit(side)), ft.CAP_PER if side != "kenri" else 1e12)
            sh = _shares(want, px, side)
            x.update(want=want, shares=sh, amount=round(sh * px))
            if x0.get("mult") == 0:
                x["slot"], x["why"] = "見送り（相場の条件）", x0.get("why_mkt", "")
            elif not sh:
                x["slot"], x["why"] = "100株に届かない", f"100株 {px * 100 / 1e4:,.1f}万円 ＞ 1回の金額の{OVER}倍"
            elif x["code"] in held:
                x["slot"], x["why"] = "すでに保有中", ""
            elif side in _CAP and sum((r.get("amount") or 0) for r in open_ if r["side"] == side) + _cap_add.get(side, 0) + x["amount"] > eq * _CAP[side] * 1.0001:
                x["slot"], x["why"] = "枠がいっぱい", f"{NAME[side]}は合計で資金の{_CAP[side] * 100:.0f}%まで（落ち込みを抑えるため）"
            elif n[side] >= ft.MAX_N[side]:
                x["slot"], x["why"] = "枠がいっぱい", f"{NAME[side]}は同時に{ft.MAX_N[side]}銘柄まで"
            elif used + x["amount"] > eq:
                x["slot"], x["why"] = "枠がいっぱい", f"資金の残り {max(eq - used, 0) / 1e4:,.1f}万円"
            else:
                x["slot"], x["why"] = "枠あり", ""
                n[side] += 1; used += x["amount"]; held.add(x["code"])
                _cap_add[side] = _cap_add.get(side, 0) + x["amount"]
            todays.append(x)
    have = {(s["date"], s["side"], s["code"]) for s in led["signals"]}
    for x in todays:
        if x.get("date", "") >= led["start"] and x["slot"] == "枠あり" and (x["date"], x["side"], x["code"]) not in have:
            led["signals"].append({k: v for k, v in x.items() if k not in ("slot", "why")})
    os.makedirs("data", exist_ok=True)
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = ft.evaluate(led, prices) if led["signals"] else []
    # ---- 明日の注文の一覧（パソコン側のプログラムが読む）
    orders = []
    for r in rows:
        st = str(r.get("status", ""))
        if r["side"] == "gapdown":
            continue
        if st.startswith("明日"):
            if r["side"] == "rebound":
                orders.append(dict(action="buy", kind="指値", code=r["code"], qty=r["shares"], price=r.get("limit"), est=r.get("last_close"), tv20=_tv20(prices.get(r["code"])), side=r["side"], note="前日終値−3%の指値。当日だけ有効"))
            elif r["side"] == "kenri":
                orders.append(dict(action="buy", kind="引成", code="1306", qty=r["shares"], est=r.get("close"), side=r["side"], date=r.get("buy_date"), note="権利取り前：この日の引けで買う"))
            else:
                orders.append(dict(action="buy", kind="寄成", code=r["code"], qty=r["shares"], est=r.get("close"), tv20=_tv20(prices.get(r["code"])), side=r["side"], note="寄りで成行。約定したら損切りの逆指値を入れる"))
        elif st == "保有中":
            df = prices.get(r["code"])
            sd = _sell_date(r, df) if df is not None else None
            o = dict(action="hold", code=r["code"], qty=r["shares"], side=r["side"], entry=r.get("entry"),
                     stop=r.get("stop_px"), tp=r.get("tp_px"), sell_close_on=sd)
            orders.append(o)
    gi = info.get("gd") or {}
    gd = None
    if gi.get("active_tomorrow"):
        unit_ = min(eq * ft.UNIT_GD, ft.CAP_PER)
        gd = dict(active=True, max_n=ft.MAX_N["gapdown"], unit=unit_, gap=ft.GD_GAP, tp=ft.GD_TP, stop=ft.GD_STOP,
                  watch=[dict(code=w["code"], name=w.get("name", ""), prev_close=w["close"], limit=round(w["close"] * (1 - ft.GD_GAP), 1), tv20=w.get("tv20"),
                              qty=_shares(unit_, w["close"] * (1 - ft.GD_GAP), "gapdown")) for w in gi.get("tomorrow") or []])
    nxt = str(np.busday_offset(np.datetime64(data_date), 1, roll="forward"))
    out = dict(made=pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"), data_date=data_date, for_date=nxt, rule_id=ft.RULE_ID,
               capital=eq, orders=orders, gapdown=gd,
               limits=dict(max_order_yen=int(min(eq * 0.05 * OVER, ft.CAP_PER * OVER)), max_day_buy_yen=int(eq), max_buy_orders=15, min_tv20=3e7))
    out["sig"] = sign(out)
    json.dump(out, open(ORDERS, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    page(rows, todays, out, eq, data_date, led["start"], os.path.join(out_dir, "auto.html"))
    print(f"自動売買版：明日の買い {sum(1 for o in orders if o['action'] == 'buy')}件・保有 {sum(1 for o in orders if o['action'] == 'hold')}件・"
          f"窓あけ下げ {'対象期間' if gd else 'なし'}・台帳 {len(led['signals'])}件")


def page(rows, todays, out, eq, data_date, start, path):
    yen = lambda v: f"{v / 1e4:+,.1f}万円"
    man = lambda v: f"{v / 1e4:,.1f}万円"
    link = lambda x: f'<a href="https://finance.yahoo.co.jp/quote/{x["code"]}.T/chart" target="_blank" rel="noopener">{x["code"]} {h.escape(str(x.get("name", "")))}</a>'

    def table(items, cols):
        if not items:
            return '<p class="none">なし</p>'
        head = "".join(f"<th>{h.escape(c)}</th>" for c, _ in cols)
        body = "".join("<tr>" + "".join(f"<td>{f(x)}</td>" for _, f in cols) + "</tr>" for x in items)
        return f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>'
    buy = [o for o in out["orders"] if o["action"] == "buy"]
    hold = [o for o in out["orders"] if o["action"] == "hold"]
    names = {r["code"]: r.get("name", "") for r in rows}
    for o in buy + hold:
        o.setdefault("name", names.get(o["code"], ""))
    closed = sorted([r for r in rows if r.get("status") == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    skip = [x for x in todays if x["slot"] != "枠あり"]
    cl = sum(r.get("pnl", 0) for r in closed); op = sum(r.get("pnl", 0) for r in rows if r.get("status") == "保有中")
    bcols = [("手法", lambda o: NAME.get(o["side"], o["side"])), ("銘柄", link), ("注文", lambda o: h.escape(o["kind"] + (f' {o["price"]:,.0f}円' if o.get("price") else ""))),
             ("株数", lambda o: f'{o["qty"]:,}'), ("メモ", lambda o: h.escape(o.get("note", "")))]
    hcols = [("手法", lambda o: NAME.get(o["side"], o["side"])), ("銘柄", link), ("株数", lambda o: f'{o["qty"]:,}'), ("買値", lambda o: f'{o.get("entry") or 0:,.1f}円'),
             ("損切りの逆指値", lambda o: f'{o["stop"]:,.0f}円' if o.get("stop") else "—"), ("利確の指値", lambda o: f'{o["tp"]:,.0f}円' if o.get("tp") else "なし"),
             ("時間で手じまう日（引成）", lambda o: o.get("sell_close_on") or "—")]
    ccols = [("手法", lambda x: NAME.get(x["side"], x["side"])), ("銘柄", link), ("買った日", lambda x: x.get("entry_date", "")), ("売った日", lambda x: x.get("exit_date", "")),
             ("損益", lambda x: f'{x.get("ret", 0):+.1f}%（{yen(x.get("pnl", 0))}）'), ("理由", lambda x: h.escape(x.get("why", "")))]
    scols = [("手法", lambda x: NAME[x["side"]]), ("銘柄", link), ("理由", lambda x: h.escape(x["slot"] + ("：" + x["why"] if x.get("why") else "")))]
    gd = out.get("gapdown")
    gd_txt = (f"<p class='warn'>窓あけ下げの対象期間：明日8:50〜8:58に、プログラムが下の監視リスト{len(gd['watch'])}銘柄の気配を見て、"
              f"−5%以下の銘柄を下げの大きい順に{gd['max_n']}銘柄まで寄指で注文します。</p>") if gd else ""
    html = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>自動売買（eスマート証券）</title><style>
body{{font-family:'Noto Sans JP',system-ui,sans-serif;margin:0;background:#f5f4f0;color:#1d1d1b;font-size:14px;line-height:1.55}}
main{{max-width:1100px;margin:0 auto;padding:16px}}h1{{font-size:18px;margin:4px 0 2px}}h2{{font-size:15px;margin:0 0 8px}}
.card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;padding:12px 14px;margin:10px 0}}.sub{{font-size:12.5px;color:#6b6a65}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}}.stat .k{{font-size:11.5px;color:#6b6a65}}.stat .v{{font-size:17px;font-weight:700}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:5px 7px;border-bottom:1px solid #eeece6;text-align:left;vertical-align:top}}
th{{font-size:12px;color:#6b6a65;font-weight:500;white-space:nowrap}}.none{{color:#999}}.warn{{background:#fff4e5;border:1px solid #f0c27b;border-radius:8px;padding:8px 10px}}a{{color:#1f4e79}}
</style></head><body><main>
<a href="forward2.html" class="sub">← 個人の今日の売買</a>　<a href="mini.html" class="sub">🪙 ミニ株</a>　<a href="houjin.html" class="sub">🏢 法人</a>
<h1>自動売買（三菱UFJ eスマート証券・kabuステーションAPI）</h1>
<div class="sub">{h.escape(data_date)} 終値時点・{h.escape(out['made'])} 作成。{h.escape(out['for_date'])} の注文。記録開始 {h.escape(start)}。パソコン側のプログラムは data/auto_orders.json を読んで注文します（今は練習モード）。</div>
<div class="card"><div class="stats">
<div class="stat"><div class="k">資金（最初 {AUTO_CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{eq / 1e4:,.1f}万円</div></div>
<div class="stat"><div class="k">1回の注文の上限（安全装置）</div><div class="v">{man(out['limits']['max_order_yen'])}</div></div>
<div class="stat"><div class="k">決済済みの損益</div><div class="v">{yen(cl)}</div></div>
<div class="stat"><div class="k">保有中の含み損益</div><div class="v">{yen(op)}</div></div></div>
<p class="sub">100株単位（100株に届かなくても1回の金額の1.5倍までなら100株）。手法は新高値・IR・IPO・リバウンド・窓あけ下げ・権利取り前。1回の金額は個人と同じ割合（新高値4%・IR／IPO／窓あけ下げ7%・リバウンド9%、信用は使わない）。空売りは慣れたら足す（75日線の買いは休止中）。</p></div>
{gd_txt}
<div class="card"><h2>明日出す買い注文（{len(buy)}件）</h2>{table(buy, bcols)}</div>
<div class="card"><h2>保有中と、入れておく売り注文（{len(hold)}件）</h2>{table(hold, hcols)}
<p class="sub">損切りは逆指値（新高値は追いかけのラインも同じ）、利確は指値。値段が変わったら、プログラムが毎朝入れ直す。</p></div>
<div class="card"><h2>見送った候補（{len(skip)}件）</h2>{table(skip, scols)}</div>
<div class="card"><h2>決済済み（{len(closed)}件）</h2>{table(closed[:60], ccols)}</div>
</main></body></html>"""
    open(path, "w", encoding="utf-8").write(html)
