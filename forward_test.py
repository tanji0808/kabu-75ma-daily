"""
フォワードテスト（これからの成績の記録）と「今日の売買」

採用ルール（バックテストで一番バランスが良かった組み合わせ）
  買い   : 75日線ブレイク（本日突破）の合図のうち、
           ・合図までの5日間の上昇が+3%以下　・RSI(14)が60以下　・ファンダメンタルランク6以上
           を満たすものを、翌営業日の始値で買う
  ロスカット（引き上げ式。終値で判断し、割ったら翌営業日の始値で売る）
           ① 最初は −1σ（日々動く）
           ② 終値が「買値＋幅」以上になったら、ロスカットを「買値＋幅」に上げて利益を確保
              （幅 ＝ 買値 −（合図の日の −1σ））
           ③ その後、終値が +1σ より上で終わった日は、ロスカットをその日の +1σ まで引き上げる（下げない）
  最長 120 営業日。同じ銘柄を持っている間の新しい合図は使わない。

台帳 data/forward.json には「合図（銘柄・日付・その日のランク）」だけを保存し、
買値・ロスカット・損益は毎回、株価データから計算し直す（あとから結果を書き換えられない形）。
"""

from __future__ import annotations

import html as h
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

LEDGER = os.environ.get("FORWARD_PATH", "data/forward.json")
MAX_HOLD = 120
MIN_RANK = 6
SIGNAL_FILTER = "calm5+rsi60"
RULE_ID = "trail_calm5_rsi60_rank6"
RULE = "75日線ブレイク＋急騰除外（5日+3%以下・RSI60以下）＋ランク6以上 → 引き上げ式ロスカット"


def load_ledger(path: str = LEDGER) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            led = json.load(f)
        if led.get("rule_id") == RULE_ID:
            return led
        # ルールが変わったときは、前の記録を残したうえで新しく始める
        prev = led.get("previous", [])
        if led.get("signals") or led.get("start"):
            prev = prev + [{k: led.get(k) for k in ("start", "rule", "signals")}]
        return {"start": None, "rule_id": RULE_ID, "rule": RULE, "signals": [], "previous": prev}
    return {"start": None, "rule_id": RULE_ID, "rule": RULE, "signals": []}


def record_signals(ledger: dict, master: pd.DataFrame, prices: dict, ranks: dict) -> list[dict]:
    """合図を台帳に追加する。取りこぼしがあっても、記録開始日以降の直近の合図は拾い直す。"""
    from backtest import signal_days
    last = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    if not ledger.get("start"):
        ledger["start"] = last
    have = {(s["code"], s["signal"]) for s in ledger["signals"]}
    new = []
    for _, m in master.iterrows():
        df = prices.get(m["コード"])
        if df is None or len(df) < 150:
            continue
        for t in signal_days(df, SIGNAL_FILTER):
            d = df.index[t].strftime("%Y-%m-%d")
            if t < len(df) - 5 or d < ledger["start"] or (m["コード"], d) in have:
                continue
            sig = {"code": m["コード"], "name": m["銘柄名"], "sector": m["業種"], "signal": d,
                   "close": round(float(df["Close"].iloc[t]), 1), "frank": ranks.get(m["コード"]),
                   "recorded": datetime.now().strftime("%Y-%m-%d %H:%M")}
            ledger["signals"].append(sig)
            new.append(sig)
    ledger["signals"].sort(key=lambda s: (s["signal"], s["code"]))
    return new


def _bands(df: pd.DataFrame):
    c = df["Close"]
    m25 = c.rolling(25).mean()
    sd = c.rolling(25).std(ddof=0)
    return (df["Open"].to_numpy(float), c.to_numpy(float), (m25 - sd).to_numpy(), (m25 + sd).to_numpy())


def evaluate(ledger: dict, prices: dict, bench: pd.DataFrame | None) -> list[dict]:
    """台帳の合図ごとに、ルールどおり売買した結果と、今のロスカットの位置を計算する。"""
    out, busy_until = [], {}
    for s in ledger["signals"]:
        r = dict(s)
        df = prices.get(s["code"])
        pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)} if df is not None else {}
        if s["signal"] not in pos:
            r["status"] = "データなし"
            out.append(r)
            continue
        if s.get("frank") is None or s["frank"] < MIN_RANK:
            r["status"] = f"ランク{s.get('frank') if s.get('frank') is not None else '不明'}のため見送り"
            out.append(r)
            continue
        if busy_until.get(s["code"], "") >= s["signal"]:
            r["status"] = "保有中のため見送り"
            out.append(r)
            continue
        o, c, d1, u1 = _bands(df)
        n, t = len(df), pos[s["signal"]]
        e = t + 1
        if e >= n:                                   # 今日の合図 → 明日の始値で買う
            est = c[t]
            width = est - d1[t] if d1[t] < est else est * 0.03
            r.update(status="買い待ち", init_stop=round(float(d1[t]), 1), est_buy=round(float(est), 1),
                     est_lock=round(float(est + width), 1), width_pct=round(float(width / est * 100), 2))
            busy_until[s["code"]] = "9999"
            out.append(r)
            continue
        buy = o[e]
        width = buy - d1[t] if d1[t] < buy else buy * 0.03
        r.update(entry=df.index[e].strftime("%Y-%m-%d"), buy=round(float(buy), 1),
                 lock_price=round(float(buy + width), 1), width_pct=round(float(width / buy * 100), 2))
        lock = stop = None
        exit_sig, reason = None, None
        for j in range(e, min(n, e + MAX_HOLD)):
            if stop is None:
                if c[j] < d1[j]:
                    exit_sig, reason = j, "−1σ割れ（損切り）"
                    break
                if c[j] >= buy + width:
                    lock = stop = buy + width
            else:
                if c[j] < stop:
                    exit_sig, reason = j, ("利益確保ライン割れ" if stop == lock else "+1σ引き上げライン割れ")
                    break
                if c[j] > u1[j]:
                    stop = max(stop, u1[j])
        if exit_sig is None and e + MAX_HOLD - 1 < n:
            exit_sig, reason = e + MAX_HOLD - 1, "期間満了"
        if exit_sig is None:                         # まだ持っている
            L = n - 1
            if stop is None:
                stage, cur_stop = "① −1σ（損切りライン）", d1[L]
                nxt = f"終値が {buy + width:,.1f}円 以上で ② 利益確保ラインへ"
            elif stop == lock:
                stage, cur_stop = "② 利益確保ライン", stop
                nxt = f"終値が +1σ（今日 {u1[L]:,.1f}円）より上で ③ 引き上げ"
            else:
                stage, cur_stop = "③ +1σ引き上げ", stop
                nxt = f"終値が +1σ（今日 {u1[L]:,.1f}円）より上なら、その値まで引き上げ"
            r.update(status="保有中", now=round(float(c[L]), 1), ret=round((c[L] / buy - 1) * 100, 2),
                     days=int(L - e), stage=stage, stop=round(float(cur_stop), 1),
                     to_stop=round((cur_stop / c[L] - 1) * 100, 2), next=nxt, plus_stop=bool(cur_stop > buy))
            r["bench"] = _bench(bench, r["entry"], None)
            busy_until[s["code"]] = "9999"
        elif exit_sig + 1 >= n:                      # 今日、売りの合図が出た
            r.update(status="売り合図", reason=reason, now=round(float(c[-1]), 1),
                     ret=round((c[-1] / buy - 1) * 100, 2), days=int(n - 1 - e))
            r["bench"] = _bench(bench, r["entry"], None)
            busy_until[s["code"]] = "9999"
        else:
            x = exit_sig + 1
            r.update(status="決済済み", exit=df.index[x].strftime("%Y-%m-%d"), sell=round(float(o[x]), 1),
                     ret=round((o[x] / buy - 1) * 100, 2), days=int(x - e), reason=reason)
            r["bench"] = _bench(bench, r["entry"], r["exit"])
            busy_until[s["code"]] = r["exit"]
        if r.get("bench") is not None and r.get("ret") is not None:
            r["excess"] = round(r["ret"] - r["bench"], 2)
        out.append(r)
    return out


def _bench(bench: pd.DataFrame | None, d1: str, d2: str | None) -> float | None:
    if bench is None or not len(bench):
        return None
    try:
        a = bench["Open"].asof(pd.Timestamp(d1))
        b = bench["Open"].asof(pd.Timestamp(d2)) if d2 else bench["Close"].iloc[-1]
        return round((b / a - 1) * 100, 2) if a == a and b == b and a else None
    except Exception:
        return None


def stats(rows: list[dict]) -> dict:
    def grp(rs):
        if not rs:
            return {"n": 0}
        r = np.array([x["ret"] for x in rs])
        ex = [x["excess"] for x in rs if x.get("excess") is not None]
        return {"n": len(rs), "win": round(float((r > 0).mean() * 100), 1), "avg": round(float(r.mean()), 2),
                "sum": round(float(r.sum()), 2), "excess": round(float(np.mean(ex)), 2) if ex else None}
    closed = [x for x in rows if x["status"] == "決済済み"]
    hold = [x for x in rows if x["status"] in ("保有中", "売り合図")]
    return {"closed": grp(closed), "hold": grp(hold), "signals": len(rows)}


def build_page(ledger: dict, rows: list[dict], data_date: str, path: str) -> None:
    from kabu_75ma_search import rank_color
    S = stats(rows)
    pct = lambda v: "—" if v is None else f"{v:+.2f}%"
    yen = lambda v: "—" if v is None else f"{v:,.1f}"
    cls = lambda v: "" if v is None else "up" if v > 0 else "down" if v < 0 else ""
    fr = lambda v: f'<span class="fr" style="background:{rank_color(v)}">{v if v is not None else "—"}</span>'
    link = lambda r: f'<a href="index.html?c={h.escape(r["code"])}&amp;v=book">{h.escape(r["code"])} {h.escape(r["name"])}</a>'
    yahoo = lambda r: f'<a class="y" href="https://finance.yahoo.co.jp/quote/{h.escape(r["code"])}.T/chart" target="_blank" rel="noopener">Yahoo↗</a>'

    buys = [r for r in rows if r["status"] == "買い待ち"]
    sells = [r for r in rows if r["status"] == "売り合図"]
    hold = sorted([r for r in rows if r["status"] == "保有中"], key=lambda r: -r.get("to_stop", 0))
    closed = [r for r in rows if r["status"] == "決済済み"][::-1]
    skipped = [r for r in rows if "見送り" in r["status"] or r["status"] == "データなし"]

    def kv(label, value, note=""):
        return (f'<div class="tk"><span class="l">{label}</span><div class="r"><b>{value}</b>'
                f'{f"<small>{note}</small>" if note else ""}</div></div>')

    def buy_card(r):
        return (f'<div class="tc buy"><div class="tt">{link(r)} {fr(r.get("frank"))} {yahoo(r)}</div>'
                + kv("買い", "明日の始値", f"今日の終値 {yen(r['est_buy'])}円")
                + kv("最初のロスカット", f"{yen(r['init_stop'])}円", "−1σ。日々動きます")
                + kv("利益確保ライン", f"約{yen(r['est_lock'])}円", "終値でここ以上になったら、ロスカットをここへ")
                + f'<div class="tn">確保ラインは「買値＋（買値−最初のロスカット）」。今日の終値で買った場合の目安で、明日の始値で決まります（幅 約{r["width_pct"]:.1f}%）。</div></div>')

    def sell_card(r):
        return (f'<div class="tc sell"><div class="tt">{link(r)} {fr(r.get("frank"))} {yahoo(r)}</div>'
                + kv("売り", "明日の始値", h.escape(r["reason"]))
                + kv("買値 → 今日の終値", f'{yen(r["buy"])} → {yen(r["now"])}円',
                     f'<span class="{cls(r["ret"])}">{pct(r["ret"])}</span>')
                + '</div>')

    def hold_card(r):
        plus = '<span class="plus">利益確保</span>' if r.get("plus_stop") else ""
        return (f'<div class="tc hold"><div class="tt">{link(r)} {fr(r.get("frank"))} {yahoo(r)}</div>'
                + kv("ロスカット", f'{yen(r["stop"])}円{plus}', f'あと{abs(r["to_stop"]):.1f}%下がると届く')
                + kv("段階", h.escape(r["stage"]))
                + kv("買値 → 今日の終値", f'{yen(r["buy"])} → {yen(r["now"])}円',
                     f'<span class="{cls(r["ret"])}">{pct(r["ret"])}</span>・{r["days"]}日目（{r["entry"]}に購入）')
                + kv("次の引き上げ", "", h.escape(r["next"]))
                + '</div>')

    def closed_row(r):
        return (f'<tr><td class="t">{link(r)}</td><td>{fr(r.get("frank"))}</td><td>{r["entry"]}</td><td>{yen(r["buy"])}</td>'
                f'<td>{r["exit"]}</td><td>{yen(r["sell"])}</td><td class="{cls(r["ret"])}">{pct(r["ret"])}</td>'
                f'<td>{r["days"]}</td><td class="{cls(r.get("excess"))}">{pct(r.get("excess"))}</td><td class="t">{h.escape(r["reason"])}</td></tr>')

    buy_html = "".join(buy_card(r) for r in buys) or '<p class="none">今日は買う銘柄はありません。</p>'
    sell_html = "".join(sell_card(r) for r in sells) or '<p class="none">今日は売る銘柄はありません。</p>'
    hold_html = "".join(hold_card(r) for r in hold) or '<p class="none">いまは保有している銘柄はありません。</p>'
    closed_html = "".join(closed_row(r) for r in closed) or '<tr><td class="t" colspan="10">まだ決済した取引はありません</td></tr>'
    skip_html = "".join(f'<tr><td class="t">{link(r)}</td><td>{r["signal"]}</td><td>{fr(r.get("frank"))}</td><td class="t">{h.escape(r["status"])}</td></tr>' for r in skipped)
    skip_card = (f'<div class="card"><h2>買わなかった合図</h2><div class="scroll"><table><tr><th>銘柄</th><th>合図の日</th><th>F</th><th>理由</th></tr>'
                 f'{skip_html}</table></div></div>') if skipped else ""

    def kpi(title, g, note=""):
        if not g.get("n"):
            return f'<div class="kpi"><div class="k">{title}</div><div class="v">—</div><div class="s">まだありません</div></div>'
        return (f'<div class="kpi"><div class="k">{title}</div><div class="v {cls(g["avg"])}">{pct(g["avg"])}</div>'
                f'<div class="s">{g["n"]}回・勝率{g["win"]}%・TOPIX比 {pct(g["excess"])}{note}</div></div>')

    page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>今日の売買（フォワードテスト）</title>
<style>
:root{{--line:#e3e1db;--soft:#efede7;--sub:#6b6a65;--up:#c8312b;--down:#1f5fbf;--accent:#1f4e79;--mono:ui-monospace,"SF Mono",Menlo,monospace}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f4f0;color:#1d1d1b;font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;font-size:14px;line-height:1.55}}
header{{background:#fff;border-bottom:1px solid var(--line);padding:16px}}h1{{font-size:18px;margin:0 0 4px}}.meta{{font-size:12px;color:var(--sub)}}
.wrap{{max-width:960px;margin:0 auto;padding:12px;display:grid;gap:12px;grid-template-columns:minmax(0,1fr)}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}}.card h2{{font-size:15px;margin:0 0 10px}}
.card h2 small{{font-weight:400;color:var(--sub);font-size:12px;margin-left:6px}}
.today{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px}}
.tc{{border-radius:10px;padding:12px;border:1px solid var(--line)}}.tc.buy{{background:#fdf1ef;border-color:#f3cfc9}}.tc.sell{{background:#eef3fb;border-color:#cddbf0}}
.tc.hold{{background:#fff}}
.tt{{font-weight:700;font-size:15px;margin-bottom:6px}}
.tk{{display:flex;gap:10px;font-size:13px;padding:4px 0;border-top:1px dashed rgba(0,0,0,.06)}}
.tk .l{{color:var(--sub);flex:0 0 118px}}.tk .r{{flex:1;min-width:0}}.tk b{{font-family:var(--mono)}}
.tk small{{display:block;color:var(--sub);font-size:11.5px;line-height:1.5}}.tn{{font-size:11.5px;color:var(--sub);margin-top:6px}}
.none{{color:var(--sub);margin:4px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}}
.kpi{{background:#faf9f6;border:1px solid var(--soft);border-radius:10px;padding:10px 12px}}.kpi .k{{font-size:11.5px;color:var(--sub)}}
.kpi .v{{font-family:var(--mono);font-size:20px;font-weight:600}}.kpi .s{{font-size:11.5px;color:var(--sub)}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{font-weight:500;color:var(--sub);font-size:11.5px;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
td{{padding:7px 8px;border-bottom:1px solid var(--soft);text-align:right;font-family:var(--mono);white-space:nowrap}}
th:first-child,td:first-child,td.t{{text-align:left;font-family:inherit}}td.small{{font-size:11.5px;color:var(--sub);white-space:normal;min-width:180px}}
a{{color:var(--accent);text-decoration:none}}a.y{{font-size:11.5px;font-weight:400;margin-left:4px}}
.fr{{color:#fff;border-radius:5px;padding:1px 6px;font-weight:700;font-size:11.5px}}
.plus{{display:inline-block;margin-left:4px;font-size:10.5px;background:#e6f3ea;color:#2a7a3b;border-radius:3px;padding:0 5px;font-family:inherit}}
.rules{{font-size:13px;line-height:1.75;margin:0;padding-left:18px}}
</style></head><body>
<header><h1>今日の売買（フォワードテスト）</h1>
<div class="meta">{data_date} 終値時点｜記録開始 {ledger.get("start") or "—"}｜合図 {S["signals"]}件</div>
<div class="meta"><a href="index.html">← スクリーナー</a>　<a href="cards.html">75日線ブレイク候補（広め）</a>　<a href="backtest.html">バックテスト結果</a></div></header>
<div class="wrap">
<div class="card"><h2>明日の始値で買う<small>{len(buys)}銘柄</small></h2><div class="today">{buy_html}</div></div>
<div class="card"><h2>明日の始値で売る<small>{len(sells)}銘柄</small></h2><div class="today">{sell_html}</div></div>
<div class="card"><h2>保有中の銘柄とロスカットの位置<small>{len(hold)}銘柄・ロスカットに近い順</small></h2><div class="today">{hold_html}</div>
<p class="meta" style="margin:8px 0 0">ロスカットは終値で判断します。終値がロスカットを割ったら、翌日の始値で売ります。①の−1σは日々動くので、毎日このページで確認してください。</p></div>
<div class="card"><h2>成績</h2><div class="kpis">{kpi("決済済み", S["closed"])}{kpi("保有中（含み損益）", S["hold"], "（今日の終値）")}</div>
<p class="meta" style="margin:8px 0 0">バックテスト（ランク6以上）では、勝率 約50%・1回の平均 +1.5% でした。数十回たまるまでは数字は大きくぶれます。</p></div>
<div class="card"><h2>決済済み（新しい順）</h2><div class="scroll"><table>
<tr><th>銘柄</th><th>F</th><th>買った日</th><th>買値</th><th>売った日</th><th>売値</th><th>損益</th><th>保有日数</th><th>TOPIX比</th><th>理由</th></tr>{closed_html}</table></div></div>
{skip_card}
<div class="card"><h2>ルール</h2><ul class="rules">
<li><b>買い</b>：75日線ブレイク（3ヶ月以上の下落後、終値が75日線を下から上に抜けた日・出来高50万株以上）のうち、<b>合図までの5日間の上昇が+3%以下</b>、<b>RSIが60以下</b>、<b>ファンダメンタルランク6以上</b>の銘柄を、翌日の始値で買う。</li>
<li><b>ロスカット①</b>：最初は−1σ。終値が−1σを割ったら翌日の始値で売る（損切り）。</li>
<li><b>ロスカット②</b>：終値が「買値＋幅」（幅＝買値−合図の日の−1σ）以上になったら、ロスカットを「買値＋幅」に上げて利益を確保。</li>
<li><b>ロスカット③</b>：その後、終値が+1σより上で終わった日は、ロスカットをその日の+1σまで引き上げる（下げない）。</li>
<li>最長120営業日で売る。手数料・税金は含めていません。実際のお金は動いていません。売買のおすすめではありません。</li></ul></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def run(master: pd.DataFrame, prices: dict, ranks: dict, out_dir: str) -> None:
    ledger = load_ledger()
    bench = None
    try:
        import yfinance as yf
        b = yf.download("1306.T", period="1y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(b.columns, pd.MultiIndex):
            b.columns = b.columns.get_level_values(0)
        bench = b[["Open", "Close"]].dropna()
    except Exception as e:
        print(f"TOPIX連動ETFの取得に失敗（{e}）")
    new = record_signals(ledger, master, prices, ranks)
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1)
    rows = evaluate(ledger, prices, bench)
    data_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    build_page(ledger, rows, data_date, os.path.join(out_dir, "forward.html"))
    S = stats(rows)
    n_buy = sum(r["status"] == "買い待ち" for r in rows)
    n_sell = sum(r["status"] == "売り合図" for r in rows)
    print(f"フォワードテスト: 新しい合図 {len(new)}件／合計 {S['signals']}件・明日買う {n_buy}・明日売る {n_sell}・"
          f"保有中 {S['hold'].get('n', 0)}・決済済み {S['closed'].get('n', 0)}")
