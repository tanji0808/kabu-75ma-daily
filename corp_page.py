"""法人用の「今日の売買」ページ（houjin.html）と台帳（data/forward_corp.json）。

証券会社を通して少ない手間で回すための版（2026-11-27 作成）。手法は5つだけ：
  IRトレード   : 資金の5%・20銘柄まで。寄りで成行買い＋買値−10%に逆指値。60営業日目の引けで売る。
  IPOセカンダリー: 資金の5%・10銘柄まで。寄りで成行買い＋−10%／+30%のOCO。120営業日で手じまい。
  新高値ブレイク : 資金の4%・12銘柄まで。寄りで成行買い＋−6%に逆指値。月1回（21営業日ごと）だけ見直し、
                 終値の最高値が買値+25%以上なら、逆指値を「終値の最高値−20%（最低でも買値）」へ引き上げ。120営業日で手じまい。
  権利取り前   : 3月・9月、TOPIX連動ETF（1306）を資金の20%。
  年末の反発   : 20銘柄を資金の1%ずつ。大納会の5営業日前の引けで買い、1月の5営業日目の引けで売る。
1回の金額は、その銘柄の直前20日の売買代金の平均の5%まで（大きな注文で値段を動かさないため）。
候補の判定は個人の「今日の売買」と同じ（同じ条件・同じ見送り条件）。
10年の検証（1億円・単利）：1年平均 約2,700万円・最大の落ち込み 約−990万円・負けた年なし。
"""
from __future__ import annotations

import html as h
import json
import os
from datetime import date

import numpy as np
import pandas as pd

CORP_CAPITAL = int(os.environ.get("CORP_CAPITAL", "100000000"))
UNIT = {"ir": 0.05, "ipo": 0.05, "newhigh": 0.04, "kenri": 0.20, "yearend": 0.01}
MAX_N = {"ir": 20, "ipo": 10, "newhigh": 12, "kenri": 1, "yearend": 20}
ORDER = ["kenri", "ipo", "newhigh", "ir", "yearend"]
LIQ = 0.05
NH_STOP, NH_ARM, NH_TRAIL, NH_REVIEW, NH_HOLD = 0.06, 0.25, 0.20, 21, 120
COST = 0.002
LEDGER = "data/forward_corp.json"
NAME = {"ir": "IRトレード", "ipo": "IPO", "newhigh": "新高値", "kenri": "権利取り前", "yearend": "年末の反発"}


def _tv20(df) -> float | None:
    if df is None or len(df) < 21:
        return None
    c, v = df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
    return float(np.mean(c[-20:] * v[-20:]))


def _nh_eval(s: dict, df: pd.DataFrame) -> dict:
    """新高値：月1回だけ逆指値を見直す形で、今の状態を計算。"""
    r = dict(s)
    idx = pd.to_datetime(df.index).tz_localize(None).normalize()
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    o, l, c = (df[x].to_numpy(float) for x in ("Open", "Low", "Close"))
    if s["date"] not in dates:
        r["status"] = "株価なし"
        return r
    i = dates.index(s["date"])
    if i + 1 >= len(dates):
        r["status"] = "明日の寄りで買い"
        return r
    e = i + 1
    entry = o[e]
    line = entry * (1 - NH_STOP)
    hc = 0.0
    amt = s.get("amount") or 0
    for j in range(e, min(len(c), e + NH_HOLD)):
        if j > e and o[j] <= line:
            ret = o[j] / entry - 1
            r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[e], exit_date=dates[j], ret=round(ret * 100, 2),
                     why="逆指値で売り" if line > entry * (1 - NH_STOP) + 1e-9 else f"損切り（−{NH_STOP * 100:.0f}%）")
            r["pnl"] = round(amt * ret - amt * COST)
            return r
        if l[j] <= line:
            ret = line / entry - 1
            r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[e], exit_date=dates[j], ret=round(ret * 100, 2),
                     why="逆指値で売り" if line > entry * (1 - NH_STOP) + 1e-9 else f"損切り（−{NH_STOP * 100:.0f}%）")
            r["pnl"] = round(amt * ret - amt * COST)
            return r
        hc = max(hc, c[j])
        if (j - e) % NH_REVIEW == NH_REVIEW - 1 and hc >= entry * (1 + NH_ARM):
            line = max(line, entry, hc * (1 - NH_TRAIL))
    if len(c) >= e + NH_HOLD:
        j = e + NH_HOLD - 1
        ret = c[j] / entry - 1
        r.update(status="決済済み", entry=round(entry, 1), entry_date=dates[e], exit_date=dates[j], ret=round(ret * 100, 2), why="120営業日で手じまい")
        r["pnl"] = round(amt * ret - amt * COST)
        return r
    k = len(c) - 1 - e                                     # 今日が買ってから何営業日目か（0始まり）
    nxt = NH_REVIEW - 1 - (k % NH_REVIEW)                    # 次の見直しまでの営業日
    nxt_d = str(np.busday_offset(np.datetime64(dates[-1]), nxt, roll="forward")) if nxt else dates[-1]
    new_line = max(line, entry, hc * (1 - NH_TRAIL)) if hc >= entry * (1 + NH_ARM) else line
    ret = c[-1] / entry - 1
    end_d = str(np.busday_offset(np.datetime64(dates[e]), NH_HOLD - 1, roll="forward"))
    order = f"逆指値 {line:,.0f}円"
    if nxt == 0:
        order = (f"<b>今日が見直しの日</b>：逆指値を {new_line:,.0f}円 に" + ("引き上げ" if new_line > line + 1e-9 else "そのまま（まだ+25%に届いていない）"))
    else:
        order += f"。次の見直し {nxt_d}（今の最高値なら {new_line:,.0f}円）"
    r.update(status="保有中", entry=round(entry, 1), entry_date=dates[e], now=round(c[-1], 1), ret=round(ret * 100, 2),
             order=order + f"。{end_d}の引けで手じまい", review_today=(nxt == 0))
    r["pnl"] = round(amt * ret - amt * COST)
    return r


def evaluate_corp(ledger: dict, prices: dict) -> list[dict]:
    import forward_test2 as ft
    rows = []
    other = {"start": ledger["start"], "signals": [s for s in ledger["signals"] if s["side"] != "newhigh"]}
    if other["signals"]:
        rows += ft.evaluate(other, prices)
    for s in ledger["signals"]:
        if s["side"] == "newhigh":
            df = prices.get(s["code"])
            rows.append(_nh_eval(s, df) if df is not None else dict(s, status="株価なし"))
    return rows


def run_corp(prices: dict, cands: dict, info: dict, data_date: str, out_dir: str) -> None:
    led = json.load(open(LEDGER, encoding="utf-8")) if os.path.exists(LEDGER) else {"start": data_date, "capital": CORP_CAPITAL, "signals": []}
    prev = evaluate_corp(led, prices)
    eq = CORP_CAPITAL + sum(r.get("pnl", 0) for r in prev if r.get("status") == "決済済み")
    open_ = [r for r in prev if r.get("status") == "保有中" or str(r.get("status", "")).startswith("明日")]
    used = sum(r.get("amount") or 0 for r in open_)
    held = {r["code"] for r in open_}
    n = {k: sum(1 for r in open_ if r["side"] == k) for k in MAX_N}
    todays = []
    for side in ORDER:
        for x0 in cands.get(side, []):
            x = {k: v for k, v in x0.items() if k not in ("slot", "why", "shares", "amount", "want", "cap")}
            x["side"] = side
            px = x.get("close") or 0
            if side == "kenri" and prices.get("1306") is not None:
                px = float(prices["1306"]["Close"].iloc[-1]); x["close"] = px
            want = eq * UNIT[side]
            tv = None if side == "kenri" else _tv20(prices.get(x["code"]))
            if tv:
                want = min(want, tv * LIQ)
            sh = int(want // px // 100 * 100) if px else 0
            if side == "kenri":
                sh = int(want // px) if px else 0
            x.update(want=want, shares=sh, amount=round(sh * px), tv20=tv)
            if x0.get("mult") == 0:
                x["slot"], x["why"] = "見送り（相場の条件）", x0.get("why_mkt", "")
            elif not sh:
                x["slot"], x["why"] = "100株に届かない", f"1回の上限 {want / 1e4:,.0f}万円"
            elif x["code"] in held:
                x["slot"], x["why"] = "すでに保有中", ""
            elif n[side] >= MAX_N[side]:
                x["slot"], x["why"] = "枠がいっぱい", f"{NAME[side]}は同時に{MAX_N[side]}銘柄まで"
            elif used + x["amount"] > eq:
                x["slot"], x["why"] = "枠がいっぱい", f"資金の残り {max(eq - used, 0) / 1e4:,.0f}万円"
            else:
                x["slot"], x["why"] = "枠あり", ""
                n[side] += 1; used += x["amount"]; held.add(x["code"])
            todays.append(x)
    have = {(s["date"], s["side"], s["code"]) for s in led["signals"]}
    for x in todays:
        if x["slot"] == "枠あり" and (x["date"], x["side"], x["code"]) not in have:
            led["signals"].append({k: v for k, v in x.items() if k not in ("slot", "why")})
    os.makedirs("data", exist_ok=True)
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = evaluate_corp(led, prices)
    page(rows, todays, info, eq, data_date, led["start"], os.path.join(out_dir, "houjin.html"))
    print(f"法人ページ：今日の注文 {sum(x['slot'] == '枠あり' for x in todays)}件・見送り {sum(x['slot'] != '枠あり' for x in todays)}件・台帳 {len(led['signals'])}件")


def page(rows, todays, info, eq, data_date, start, path):
    man = lambda v: f"{v / 1e4:,.0f}万円"
    yen = lambda v: f"{v / 1e4:+,.1f}万円"
    link = lambda x: f'<a href="https://finance.yahoo.co.jp/quote/{x["code"]}.T/chart" target="_blank" rel="noopener">{x["code"]} {h.escape(str(x.get("name", "")))}</a>'

    def table(items, cols):
        if not items:
            return '<p class="none">なし</p>'
        head = "".join(f"<th>{h.escape(c)}</th>" for c, _ in cols)
        body = "".join("<tr>" + "".join(f"<td>{f(x)}</td>" for _, f in cols) + "</tr>" for x in items)
        return f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>'

    def order_text(x):
        p = x.get("close", 0)
        s = x["side"]
        if s == "ir":
            return f"明日の寄りで成行買い。約定したら買値−10%（目安 {p * 0.9:,.0f}円）に逆指値。60営業日目の引けで売る"
        if s == "ipo":
            return f"明日の寄りで成行買い。約定したら −10%（目安 {p * 0.9:,.0f}円）／+30%（目安 {p * 1.3:,.0f}円）のOCO。120営業日で手じまい"
        if s == "newhigh":
            return f"明日の寄りで成行買い。約定したら買値−6%（目安 {p * 0.94:,.0f}円）に逆指値。月1回の見直しは保有中の表に出ます"
        if s == "kenri":
            return f"{x.get('buy_date', '明日')}の引けで買う（引け成行）。{x.get('sell_date', '')}（権利付き最終日）の引けで売る"
        if s == "yearend":
            return f"{x.get('buy_date', '明日')}の引けで買う（引け成行）。{x.get('sell_date', '')}の引けで売る"
        return ""
    buy = [x for x in todays if x["slot"] == "枠あり"]
    skip = [x for x in todays if x["slot"] != "枠あり"]
    cols = [("手法", lambda x: NAME[x["side"]]), ("銘柄", link), ("今日の終値", lambda x: f'{x.get("close", 0):,.1f}円'),
            ("株数", lambda x: f'{x.get("shares", 0):,}'), ("金額", lambda x: man(x.get("amount", 0))), ("注文", lambda x: h.escape(order_text(x)))]
    scols = [("手法", lambda x: NAME[x["side"]]), ("銘柄", link), ("理由", lambda x: h.escape(x["slot"] + ("：" + x["why"] if x.get("why") else "")))]
    hold = [r for r in rows if r.get("status") == "保有中"]
    review = [r for r in hold if r.get("review_today")]
    closed = sorted([r for r in rows if r.get("status") == "決済済み"], key=lambda r: r.get("exit_date") or "", reverse=True)
    hcols = [("手法", lambda x: NAME.get(x["side"], x["side"])), ("銘柄", link), ("買った日", lambda x: x.get("entry_date", "")),
             ("株数", lambda x: f'{x.get("shares", 0):,}'), ("買値", lambda x: f'{x.get("entry", 0):,.1f}円'), ("今", lambda x: f'{x.get("now", 0):,.1f}円'),
             ("損益", lambda x: f'{x.get("ret", 0):+.1f}%（{yen(x.get("pnl", 0))}）'), ("注文・予定", lambda x: x.get("order") or "")]
    ccols = [("手法", lambda x: NAME.get(x["side"], x["side"])), ("銘柄", link), ("買った日", lambda x: x.get("entry_date", "")),
             ("売った日", lambda x: x.get("exit_date", "")), ("損益", lambda x: f'{x.get("ret", 0):+.1f}%（{yen(x.get("pnl", 0))}）'), ("理由", lambda x: h.escape(x.get("why", "")))]
    cl_pnl = sum(r.get("pnl", 0) for r in closed)
    op_pnl = sum(r.get("pnl", 0) for r in hold)
    win = round(sum(r.get("pnl", 0) > 0 for r in closed) / len(closed) * 100) if closed else None
    kw = info.get("kenri")
    yw = info.get("ye")
    sched = []
    if kw:
        sched.append(f"権利取り前：{kw[0]}の引けで買い → {kw[1]}の引けで売る（資金の20%）")
    if yw:
        sched.append(f"年末の反発：{yw[0]}の引けで買い → {yw[1]}の引けで売る（20銘柄・各資金の1%）")
    html = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>法人の売買（少ない手間版）</title><style>
body{{font-family:'Noto Sans JP',system-ui,sans-serif;margin:0;background:#f5f4f0;color:#1d1d1b;font-size:14px;line-height:1.55}}
main{{max-width:1100px;margin:0 auto;padding:16px}}h1{{font-size:18px;margin:4px 0 2px}}h2{{font-size:15px;margin:0 0 8px}}
.card{{background:#fff;border:1px solid #e2e0d8;border-radius:10px;padding:12px 14px;margin:10px 0}}.sub{{font-size:12.5px;color:#6b6a65}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}}.stat .k{{font-size:11.5px;color:#6b6a65}}.stat .v{{font-size:17px;font-weight:700}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:5px 7px;border-bottom:1px solid #eeece6;text-align:left;vertical-align:top}}
th{{font-size:12px;color:#6b6a65;font-weight:500;white-space:nowrap}}.none{{color:#999}}.warn{{background:#fff4e5;border:1px solid #f0c27b;border-radius:8px;padding:8px 10px}}a{{color:#1f4e79}}
</style></head><body><main>
<a href="forward2.html" class="sub">← 個人の今日の売買</a>　<a href="backtest.html" class="sub">バックテスト結果</a>　<a href="docs.html" class="sub">📘 まとめ資料</a>
<h1>法人の売買（証券会社に頼む・少ない手間版）</h1>
<div class="sub">{h.escape(data_date)} 終値時点。記録開始 {h.escape(start)}。IRトレード・IPO・新高値・権利取り前・年末の反発の5つだけ。毎日の細かい指値の入れ直しはなし。</div>
<div class="card"><div class="stats">
<div class="stat"><div class="k">資金（最初 {CORP_CAPITAL / 1e4:,.0f}万円＋決済済みの損益）</div><div class="v">{eq / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">1回の金額</div><div class="v">IR・IPO {eq * 0.05 / 1e4:,.0f}万円／新高値 {eq * 0.04 / 1e4:,.0f}万円</div></div>
<div class="stat"><div class="k">決済済みの損益（勝率）</div><div class="v">{yen(cl_pnl)}（{win if win is not None else '—'}%）</div></div>
<div class="stat"><div class="k">保有中の含み損益</div><div class="v">{yen(op_pnl)}</div></div></div>
<p class="sub">1回の金額は、その銘柄の1日の売買代金（20日平均）の5%まで。同時に持てる数は IR20・IPO10・新高値12銘柄。10年の検証（1億円・単利）：1年平均 約2,700万円・最大の落ち込み 約−990万円・負けた年なし。</p></div>
{('<p class="warn"><b>今日は新高値の見直しの日の銘柄があります</b>：' + "・".join(f"{r['code']} {h.escape(str(r.get('name', '')))}" for r in review) + '（下の保有中の表の「注文・予定」のとおりに逆指値を直す）</p>') if review else ''}
<div class="card"><h2>明日お願いする注文（{len(buy)}件）</h2>{table(buy, cols)}
<p class="sub">「目安」は今日の終値で計算。実際は約定した値段（買値）から計算し直す。</p></div>
<div class="card"><h2>保有中（{len(hold)}件）</h2>{table(hold, hcols)}
<p class="sub">新高値は買ってから21営業日ごとに見直し：それまでの終値の最高値が買値+25%以上なら、逆指値を「最高値−20%（最低でも買値）」へ引き上げる。それ以外の日は何もしない。</p></div>
<div class="card"><h2>この先の予定</h2><p>{'<br>'.join(h.escape(s) for s in sched) or '—'}</p></div>
<div class="card"><h2>見送った候補（{len(skip)}件）</h2>{table(skip, scols)}</div>
<div class="card"><h2>決済済み（{len(closed)}件）</h2>{table(closed[:60], ccols)}</div>
</main></body></html>"""
    open(path, "w", encoding="utf-8").write(html)
