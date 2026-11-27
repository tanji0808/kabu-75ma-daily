"""kabuステーションAPI（三菱UFJ eスマート証券）に、毎日の注文を出すプログラム。Windows のパソコン（またはWindowsサーバー）で動かす。

前提
  - kabuステーションを起動してログインしておく（APIの設定でパスワードを決めておく）
  - Python 3.10 以上。追加のライブラリは不要（標準ライブラリだけ）
  - 環境変数 KABUS_API_PASSWORD に APIパスワード

モード（--mode）
  dry    : 練習モード（初期値）。注文を出さず、出すはずの注文を画面とログに出すだけ。APIにもつながない。
  verify : 検証環境（ポート18081）。kabuステーションの「検証用」につないで、本物のお金を使わずに注文を試す。
  live   : 本番（ポート18080）。実際に注文が出る。最初の数週間は使わない。

仕事（--job）。Windowsの「タスク スケジューラ」で平日に次の時刻に動かす：
  morning  8:30  前の晩に作った注文（data/auto_orders.json）を読み、寄りの買い・リバウンドの指値・保有株の損切り（逆指値）を出す
  gapdown  8:52  窓あけ下げの対象期間なら、監視リストの気配を見て、−5%以下の銘柄に寄指を出す（下げの大きい順に最大10銘柄）
  after    9:10  寄りで約定した買いに、損切りの逆指値を付ける
  watch    9:15〜15:20 の間ずっと  利確の値段に届いたら、損切りの注文を取り消して売る（APIにOCOがないため）
  close    14:45 時間で手じまう日の銘柄を、損切りの注文を取り消して引成で売る

安全装置
  - 1回の注文の上限金額（auto_orders.json の limits.max_order_yen）を超える注文は出さない
  - 1日の買いの合計が資金を超える注文は出さない
  - 注文ファイルの日付（for_date）が今日でなければ何もしない
  - 環境変数 KABUS_STOP=1 か、このフォルダに STOP というファイルがあれば何もしない

注意：kabuステーションAPIの項目名・コード（FrontOrderType など）は公式の資料で確認してから live にすること。
      まず dry → verify の順に動かし、ログ（bridge_log.txt）を見て確かめる。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

ORDERS_URL = os.environ.get("KABUS_ORDERS_URL", "https://raw.githubusercontent.com/tanji0808/kabu-75ma-daily/main/data/auto_orders.json")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "bridge_log.txt")
STATE = os.path.join(HERE, "bridge_state.json")   # 今日出した注文の控え（同じ注文を二重に出さないため）

# kabuステーションAPI の注文の種類（公式資料で要確認）
FOT = {"寄成": 13, "引成": 16, "指値": 20, "寄指": 21, "成行": 10, "逆指値": 30}
SIDE_BUY, SIDE_SELL = "2", "1"


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Kabus:
    def __init__(self, mode: str):
        self.mode = mode
        self.base = f"http://localhost:{18081 if mode == 'verify' else 18080}/kabusapi"
        self.token = None

    def _req(self, method, path, body=None, auth=True):
        if self.mode == "dry":
            raise RuntimeError("dry モードではAPIにつながない")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        if auth:
            req.add_header("X-API-KEY", self.token or "")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode() or "null")

    def login(self):
        if self.mode == "dry":
            return
        pw = os.environ.get("KABUS_API_PASSWORD", "")
        if not pw:
            raise SystemExit("環境変数 KABUS_API_PASSWORD がありません")
        self.token = self._req("POST", "/token", {"APIPassword": pw}, auth=False)["Token"]

    def board(self, code):
        return self._req("GET", f"/board/{code}@1")

    def positions(self):
        return self._req("GET", "/positions?product=1")      # 現物

    def orders(self):
        return self._req("GET", "/orders?product=0")

    def cancel(self, order_id):
        if self.mode == "dry":
            log(f"  [練習] 取消 {order_id}"); return
        return self._req("PUT", "/cancelorder", {"OrderId": order_id, "Password": os.environ.get("KABUS_TRADE_PASSWORD", "")})

    def send(self, side, code, qty, kind, price=0, trigger=None, note=""):
        body = {"Password": os.environ.get("KABUS_TRADE_PASSWORD", ""), "Symbol": str(code), "Exchange": 9,   # 9＝SOR（手数料無料の条件）
                "SecurityType": 1, "Side": side, "CashMargin": 1, "DelivType": 2 if side == SIDE_BUY else 0,
                "FundType": "AA" if side == SIDE_BUY else "  ", "AccountType": 4, "Qty": int(qty),
                "FrontOrderType": FOT[kind], "Price": float(price or 0), "ExpireDay": 0}
        if kind == "逆指値":
            body["Price"] = 0
            body["ReverseLimitOrder"] = {"TriggerSec": 1, "TriggerPrice": float(trigger), "UnderOver": 1, "AfterHitOrderType": 1, "AfterHitPrice": 0}
        what = f"{'買い' if side == SIDE_BUY else '売り'} {code} {qty}株 {kind}" + (f" {price:,.0f}円" if price else "") + (f" 逆指値 {trigger:,.0f}円以下" if trigger else "")
        if self.mode == "dry":
            log(f"  [練習] {what}  {note}")
            return {"OrderId": f"dry-{code}-{kind}"}
        r = self._req("POST", "/sendorder", body)
        log(f"  [{self.mode}] {what} → {r}  {note}")
        return r


def load_orders():
    if os.path.exists(ORDERS_URL):                                    # 手元のファイルを指定した場合（試し用）
        return json.load(open(ORDERS_URL, encoding="utf-8"))
    with urllib.request.urlopen(ORDERS_URL, timeout=20) as r:
        return json.loads(r.read().decode())


def state():
    try:
        s = json.load(open(STATE, encoding="utf-8"))
    except Exception:
        s = {}
    today = dt.date.today().isoformat()
    return s if s.get("date") == today else {"date": today, "sent": {}}


def save(s):
    json.dump(s, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def guard(o, limits, spent, price):
    amt = (price or 0) * o["qty"]
    if amt > limits["max_order_yen"]:
        log(f"  見送り（1回の上限 {limits['max_order_yen']:,}円 を超える）：{o['code']} {amt:,.0f}円"); return False
    if spent + amt > limits["max_day_buy_yen"]:
        log(f"  見送り（1日の上限を超える）：{o['code']}"); return False
    return True


def job_morning(api, od, st):
    limits = od["limits"]; spent = st.get("spent", 0)
    for o in od["orders"]:
        key = f"{o['action']}-{o['code']}-{o.get('kind', 'stop')}"
        if key in st["sent"]:
            continue
        if o["action"] == "buy":
            if o["kind"] == "引成" and o.get("date") and o["date"] != dt.date.today().isoformat():
                continue
            px = o.get("price") or 0
            est = px or o.get("est") or 0                               # 寄成は前日終値で金額を見積もる
            if not est or not guard(o, limits, spent, est * 1.1):       # 寄りで1割上がっても上限内か
                if not est:
                    log(f"  見送り（値段の目安がない）：{o['code']}")
                continue
            r = api.send(SIDE_BUY, o["code"], o["qty"], o["kind"], price=px, note=o.get("note", ""))
            spent += est * o["qty"]; st["sent"][key] = r.get("OrderId") if r else None
        elif o["action"] == "hold" and o.get("stop"):
            if o.get("sell_close_on") == dt.date.today().isoformat():
                continue                                       # 今日が手じまいの日：close の仕事で引成
            r = api.send(SIDE_SELL, o["code"], o["qty"], "逆指値", trigger=o["stop"], note=f"損切り（{o['side']}）")
            st["sent"][key] = r.get("OrderId") if r else None
    st["spent"] = spent


def job_gapdown(api, od, st):
    gd = od.get("gapdown")
    if not gd:
        log("  窓あけ下げ：対象期間ではない"); return
    if api.mode == "dry":
        log(f"  [練習] 窓あけ下げ：監視 {len(gd['watch'])}銘柄の気配を見て、−5%以下を下げの大きい順に{gd['max_n']}銘柄まで寄指"); return
    hits = []
    for w in gd["watch"]:
        try:
            b = api.board(w["code"])
        except Exception as e:
            log(f"  気配の取得に失敗 {w['code']}：{e}"); continue
        ask = b.get("AskPrice") or b.get("CurrentPrice")             # 寄り前の売り気配（要確認：特別気配の扱い）
        if ask and ask <= w["limit"]:
            hits.append((ask / w["prev_close"] - 1, w))
        time.sleep(0.12)                                             # APIの回数制限への配慮
    hits.sort(key=lambda z: z[0])
    for g, w in hits[:gd["max_n"]]:
        key = f"gd-{w['code']}"
        if key in st["sent"] or not w["qty"]:
            continue
        r = api.send(SIDE_BUY, w["code"], w["qty"], "寄指", price=w["limit"], note=f"窓あけ下げ（気配 {g * 100:+.1f}%）")
        st["sent"][key] = r.get("OrderId") if r else None


def job_after(api, od, st):
    """寄りで約定した買いに、損切りの逆指値を付ける（買値から計算）。"""
    if api.mode == "dry":
        log("  [練習] 約定した買いに損切りの逆指値を付ける（新高値−6%・IR/IPO/リバウンド/窓あけ下げ−10%）"); return
    stop_pct = {"newhigh": 0.06, "ipo": 0.10, "ir": 0.10, "rebound": 0.10, "gapdown": 0.10}
    side_of = {o["code"]: o.get("side") for o in od["orders"] if o["action"] == "buy"}
    side_of.update({w["code"]: "gapdown" for w in (od.get("gapdown") or {}).get("watch", [])})
    for p in api.positions() or []:
        code = p.get("Symbol"); sd = side_of.get(code)
        key = f"stop-new-{code}"
        if not sd or key in st["sent"] or sd == "kenri":
            continue
        px = p.get("Price"); qty = int(p.get("LeavesQty") or 0)
        if px and qty:
            r = api.send(SIDE_SELL, code, qty, "逆指値", trigger=round(px * (1 - stop_pct[sd])), note=f"損切り（新規・{sd}）")
            st["sent"][key] = r.get("OrderId") if r else None


def job_watch(api, od, st):
    """利確の値段に届いたら、損切りの注文を取り消して成行で売る。"""
    tps = {o["code"]: o for o in od["orders"] if o["action"] == "hold" and o.get("tp")}
    if api.mode == "dry":
        log(f"  [練習] 利確の見張り：{len(tps)}銘柄"); return
    end = dt.datetime.now().replace(hour=15, minute=20)
    while dt.datetime.now() < end:
        for code, o in list(tps.items()):
            try:
                cur = api.board(code).get("CurrentPrice")
            except Exception:
                continue
            if cur and cur >= o["tp"] and f"tp-{code}" not in st["sent"]:
                oid = st["sent"].get(f"hold-{code}-stop")
                if oid:
                    api.cancel(oid)
                r = api.send(SIDE_SELL, code, o["qty"], "成行", note=f"利確（{o['tp']:,.0f}円に届いた）")
                st["sent"][f"tp-{code}"] = r.get("OrderId") if r else None; save(st)
            time.sleep(0.2)
        time.sleep(30)


def job_close(api, od, st):
    today = dt.date.today().isoformat()
    for o in od["orders"]:
        if o["action"] == "hold" and o.get("sell_close_on") == today and f"close-{o['code']}" not in st["sent"]:
            oid = st["sent"].get(f"hold-{o['code']}-stop")
            if oid:
                api.cancel(oid)
            r = api.send(SIDE_SELL, o["code"], o["qty"], "引成", note="時間で手じまい")
            st["sent"][f"close-{o['code']}"] = r.get("OrderId") if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["morning", "gapdown", "after", "watch", "close"])
    ap.add_argument("--mode", default="dry", choices=["dry", "verify", "live"])
    a = ap.parse_args()
    if os.environ.get("KABUS_STOP") == "1" or os.path.exists(os.path.join(HERE, "STOP")):
        log("停止中（KABUS_STOP または STOP ファイル）"); return
    od = load_orders()
    today = dt.date.today().isoformat()
    if od.get("for_date") != today:
        log(f"注文ファイルの日付が今日ではない（{od.get('for_date')}）。何もしない"); return
    api = Kabus(a.mode); api.login()
    st = state()
    log(f"== {a.job}（{a.mode}）資金 {od['capital']:,.0f}円・注文 {len(od['orders'])}件")
    {"morning": job_morning, "gapdown": job_gapdown, "after": job_after, "watch": job_watch, "close": job_close}[a.job](api, od, st)
    save(st)


if __name__ == "__main__":
    main()
