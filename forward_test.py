"""
フォワードテスト（これからの成績の記録）

ルール（バックテストで一番良かった組み合わせ）
  買い : 75日線ブレイク（本日突破）＋買いの絞り込み（25日線上向き・出来高1.5倍）の合図が出た翌営業日の始値
  売り : 終値が25日線を下回ったら翌営業日の始値。最長120営業日。
  同じ銘柄を持っている間の新しい合図は記録だけして、売買には使わない。

台帳 data/forward.json には「合図（銘柄・日付・その日のファンダメンタルランク）」だけを保存し、
買値・売値・含み損益は毎回、株価データから計算し直す（あとから結果を書き換えられない形）。
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
RULE = "75日線ブレイク＋買いの絞り込み → 25日線割れで売る"


def load_ledger(path: str = LEDGER) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"start": None, "rule": RULE, "signals": []}


def record_signals(ledger: dict, master: pd.DataFrame, prices: dict, ranks: dict) -> list[dict]:
    """合図を台帳に追加する。取りこぼしがあっても、記録開始日以降の直近の合図は拾い直す。"""
    from backtest import signal_days
    last = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
    if not ledger.get("start"):
        ledger["start"] = last                      # 初回：今日の合図から記録を始める
    have = {(s["code"], s["signal"]) for s in ledger["signals"]}
    new = []
    for _, m in master.iterrows():
        df = prices.get(m["コード"])
        if df is None or len(df) < 150:
            continue
        for t in signal_days(df, "f"):
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


def evaluate(ledger: dict, prices: dict, bench: pd.DataFrame | None) -> list[dict]:
    """台帳の合図ごとに、ルールどおり売買した結果を株価から計算する。"""
    out, busy_until = [], {}
    for s in ledger["signals"]:
        r = dict(s)
        df = prices.get(s["code"])
        pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df.index)} if df is not None else {}
        if s["signal"] not in pos:
            r["status"] = "データなし"
            out.append(r)
            continue
        if busy_until.get(s["code"], "") >= s["signal"]:
            r["status"] = "保有中のため見送り"
            out.append(r)
            continue
        o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
        ma25 = df["Close"].rolling(25).mean().to_numpy()
        n, t = len(df), pos[s["signal"]]
        e = t + 1
        if e >= n:
            r.update(status="買い待ち", note="翌営業日の始値で買う")
            busy_until[s["code"]] = "9999"
            out.append(r)
            continue
        buy = o[e]
        r.update(entry=df.index[e].strftime("%Y-%m-%d"), buy=round(float(buy), 1))
        exit_sig, reason = None, None
        for j in range(e, min(n, e + MAX_HOLD)):
            if c[j] < ma25[j]:
                exit_sig, reason = j, "25日線割れ"
                break
        if exit_sig is None and e + MAX_HOLD - 1 < n:
            exit_sig, reason = e + MAX_HOLD - 1, "期間満了"
        if exit_sig is None:                        # まだ持っている
            r.update(status="保有中", now=round(float(c[-1]), 1), ret=round((c[-1] / buy - 1) * 100, 2),
                     days=int(n - 1 - e), ma25=round(float(ma25[-1]), 1),
                     to_ma25=round((c[-1] / ma25[-1] - 1) * 100, 2))
            r["bench"] = _bench(bench, r["entry"], None)
            busy_until[s["code"]] = "9999"
        elif exit_sig + 1 >= n:                     # 今日売りの合図が出た
            r.update(status="売り合図", note="翌営業日の始値で売る", reason=reason, now=round(float(c[-1]), 1),
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
    hi = lambda rs: [x for x in rs if x.get("frank") is not None and x["frank"] >= 6]
    lo = lambda rs: [x for x in rs if x.get("frank") is None or x["frank"] <= 5]
    return {"closed": grp(closed), "closed_hi": grp(hi(closed)), "closed_lo": grp(lo(closed)),
            "hold": grp(hold), "hold_hi": grp(hi(hold)), "hold_lo": grp(lo(hold)),
            "signals": len(rows)}


def build_page(ledger: dict, rows: list[dict], data_date: str, path: str) -> None:
    from kabu_75ma_search import rank_color
    S = stats(rows)
    pct = lambda v: "—" if v is None else f"{v:+.2f}%"
    cls = lambda v: "" if v is None else "up" if v > 0 else "down" if v < 0 else ""
    fr = lambda v: (f'<span class="fr" style="background:{rank_color(v)}">{v if v is not None else "—"}</span>')
    link = lambda r: f'<a href="index.html?c={h.escape(r["code"])}&amp;v=book">{h.escape(r["code"])} {h.escape(r["name"])}</a>'

    def kpi(title, g, note=""):
        if not g.get("n"):
            return f'<div class="kpi"><div class="k">{title}</div><div class="v">—</div><div class="s">まだありません</div></div>'
        return (f'<div class="kpi"><div class="k">{title}</div><div class="v {cls(g["avg"])}">{pct(g["avg"])}</div>'
                f'<div class="s">{g["n"]}回・勝率{g["win"]}%・TOPIX比 {pct(g["excess"])}{note}</div></div>')

    hold = [r for r in rows if r["status"] in ("保有中", "売り合図", "買い待ち")]
    closed = [r for r in rows if r["status"] == "決済済み"][::-1]
    skipped = [r for r in rows if r["status"] in ("保有中のため見送り", "データなし")]
    new_today = [r for r in rows if r["signal"] == data_date]

    def hold_row(r):
        buy = "—" if r.get("buy") is None else f"{r['buy']:,.1f}"
        now = "—" if r.get("now") is None else f"{r['now']:,.1f}"
        if r["status"] == "売り合図":
            state = '<b class="sell">売り合図（翌日の始値で売る）</b>'
        elif r["status"] == "買い待ち":
            state = "買い待ち（翌日の始値で買う）"
        else:
            state = f"25日線まで {pct(r.get('to_ma25'))}"
        return (f'<tr><td class="t">{link(r)}</td><td>{fr(r.get("frank"))}</td><td>{r["signal"]}</td>'
                f'<td>{r.get("entry", "—")}</td><td>{buy}</td><td>{now}</td>'
                f'<td class="{cls(r.get("ret"))}">{pct(r.get("ret"))}</td><td>{r.get("days", "—")}</td>'
                f'<td class="t">{state}</td></tr>')

    hold_rows = "".join(hold_row(r) for r in hold) or '<tr><td class="t" colspan="9">いまは保有している銘柄はありません</td></tr>'
    closed_rows = "".join(
        f'<tr><td class="t">{link(r)}</td><td>{fr(r.get("frank"))}</td><td>{r["entry"]}</td><td>{r["buy"]:,.1f}</td>'
        f'<td>{r["exit"]}</td><td>{r["sell"]:,.1f}</td><td class="{cls(r["ret"])}">{pct(r["ret"])}</td>'
        f'<td>{r["days"]}</td><td class="{cls(r.get("excess"))}">{pct(r.get("excess"))}</td><td class="t">{r["reason"]}</td></tr>'
        for r in closed) or '<tr><td class="t" colspan="10">まだ決済した取引はありません</td></tr>'
    skip_rows = "".join(f'<tr><td class="t">{link(r)}</td><td>{r["signal"]}</td><td class="t">{r["status"]}</td></tr>' for r in skipped)

    if new_today:
        names = "、".join(f"{h.escape(r['code'])} {h.escape(r['name'])}（F{r.get('frank') if r.get('frank') is not None else '—'}）"
                         for r in new_today)
        new_html = f"今日の新しい買いの合図：{names}　→ 翌営業日の始値で買った扱いになります"
    else:
        new_html = "今日は新しい買いの合図はありませんでした。"
    page = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>フォワードテスト</title>
<style>
:root{{--line:#e3e1db;--soft:#efede7;--sub:#6b6a65;--up:#c8312b;--down:#1f5fbf;--accent:#1f4e79;--mono:ui-monospace,"SF Mono",Menlo,monospace}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f4f0;color:#1d1d1b;font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;font-size:14px;line-height:1.55}}
header{{background:#fff;border-bottom:1px solid var(--line);padding:16px}}h1{{font-size:18px;margin:0 0 4px}}.meta{{font-size:12px;color:var(--sub)}}
.wrap{{max-width:900px;margin:0 auto;padding:12px;display:grid;gap:12px;grid-template-columns:minmax(0,1fr)}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}}.card h2{{font-size:14.5px;margin:0 0 10px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}}
.kpi{{background:#faf9f6;border:1px solid var(--soft);border-radius:10px;padding:10px 12px}}.kpi .k{{font-size:11.5px;color:var(--sub)}}
.kpi .v{{font-family:var(--mono);font-size:20px;font-weight:600}}.kpi .s{{font-size:11.5px;color:var(--sub)}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{font-weight:500;color:var(--sub);font-size:11.5px;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
td{{padding:7px 8px;border-bottom:1px solid var(--soft);text-align:right;font-family:var(--mono);white-space:nowrap}}
th:first-child,td:first-child,td.t{{text-align:left;font-family:inherit}}
a{{color:var(--accent);text-decoration:none}}.sell{{color:var(--up)}}
.fr{{color:#fff;border-radius:5px;padding:1px 6px;font-weight:700;font-size:11.5px}}
.rules{{font-size:13px;line-height:1.7;margin:0;padding-left:18px}}
.new{{background:#e6eef6;border-radius:10px;padding:10px 12px;font-size:13.5px}}
</style></head><body>
<header><h1>フォワードテスト（これからの成績の記録）</h1>
<div class="meta">記録開始 {ledger.get("start") or "—"}｜{data_date} 終値時点｜合図 {S["signals"]}件</div>
<div class="meta"><a href="index.html">← スクリーナー</a>　<a href="backtest.html">バックテスト結果</a></div></header>
<div class="wrap">
<div class="new">{new_html}</div>
<div class="card"><h2>成績</h2><div class="kpis">
{kpi("決済済み（すべて）", S["closed"])}{kpi("決済済み（ランク6以上）", S["closed_hi"])}{kpi("決済済み（ランク5以下）", S["closed_lo"])}
{kpi("保有中の含み損益", S["hold"], "（今日の終値）")}
</div><p class="meta" style="margin:8px 0 0">1回の平均損益。バックテストでは、全体 +0.82%・ランク6以上 +1.54%（勝率35〜37%）でした。数十回たまるまでは、数字は大きくぶれます。</p></div>
<div class="card"><h2>保有中・売買待ち</h2><div class="scroll"><table>
<tr><th>銘柄</th><th>F</th><th>合図の日</th><th>買った日</th><th>買値</th><th>現在値</th><th>含み損益</th><th>保有日数</th><th>状態</th></tr>{hold_rows}</table></div></div>
<div class="card"><h2>決済済み（新しい順）</h2><div class="scroll"><table>
<tr><th>銘柄</th><th>F</th><th>買った日</th><th>買値</th><th>売った日</th><th>売値</th><th>損益</th><th>保有日数</th><th>TOPIX比</th><th>理由</th></tr>{closed_rows}</table></div></div>
{f'<div class="card"><h2>売買に使わなかった合図</h2><div class="scroll"><table><tr><th>銘柄</th><th>合図の日</th><th>理由</th></tr>{skip_rows}</table></div></div>' if skipped else ''}
<div class="card"><h2>ルール</h2><ul class="rules">
<li><b>買い</b>：75日線ブレイク（3ヶ月以上の下落後、終値が75日線を下から上に抜けた日）＋25日線が上向き・出来高が20日平均の1.5倍以上・出来高50万株以上。合図の翌営業日の始値で買った扱い。</li>
<li><b>売り</b>：終値が25日線を下回ったら、翌営業日の始値で売った扱い。最長120営業日。</li>
<li>F＝合図の日のファンダメンタルランク（その日の株価と財務データで計算。先読みなし）。</li>
<li>手数料・税金は含めていません。実際のお金は動いていません。売買のおすすめではありません。</li></ul></div>
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
    print(f"フォワードテスト: 新しい合図 {len(new)}件／合計 {S['signals']}件・決済済み {S['closed'].get('n', 0)}件・保有中 {S['hold'].get('n', 0)}件")
