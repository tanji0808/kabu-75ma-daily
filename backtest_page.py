"""バックテスト結果をスマホでも見やすい1枚のページにする。"""

from __future__ import annotations

import json

TEMPLATE = r"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>75日線ブレイク バックテスト</title>
<style>
:root{--bg:#f5f4f0;--panel:#fff;--ink:#1d1d1b;--sub:#6b6a65;--line:#e3e1db;--soft:#efede7;--accent:#1f4e79;--accent-soft:#e6eef6;--up:#c8312b;--down:#1f5fbf;--good:#2a7a3b;--mono:ui-monospace,"SF Mono",Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;font-size:14px;line-height:1.55}
header{background:#fff;border-bottom:1px solid var(--line);padding:16px}
h1{font-size:18px;margin:0 0 4px}.meta{font-size:12px;color:var(--sub)}
.wrap{max-width:860px;margin:0 auto;padding:12px;display:grid;gap:12px;grid-template-columns:minmax(0,1fr)}
.card{min-width:0}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:14.5px;margin:0 0 10px}
.rules{font-size:13px;line-height:1.7;margin:0;padding-left:18px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button{flex:1;min-width:140px;border:1px solid var(--line);background:#fff;border-radius:10px;padding:9px 10px;cursor:pointer;text-align:left;font:inherit}
.tabs button b{display:block;font-size:13.5px}.tabs button span{font-size:11.5px;color:var(--sub)}
.tabs button.on{border-color:var(--accent);background:var(--accent-soft)}
.verdict{font-size:14px;line-height:1.75;background:var(--accent-soft);border-radius:10px;padding:12px 14px;color:#173b5c}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.kpi{background:#faf9f6;border:1px solid var(--soft);border-radius:10px;padding:9px 11px}
.kpi .k{font-size:11px;color:var(--sub)}.kpi .v{font-family:var(--mono);font-size:18px;font-weight:600}
.kpi .s{font-size:11px;color:var(--sub)}
.up{color:var(--up)}.down{color:var(--down)}.good{color:var(--good)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{font-weight:500;color:var(--sub);font-size:11.5px;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--soft);text-align:right;font-family:var(--mono);white-space:nowrap}
th:first-child,td:first-child{text-align:left}td.t{font-family:inherit}
.scroll{overflow:auto;max-height:520px}
svg{display:block;width:100%}
.note{font-size:12px;color:var(--sub);line-height:1.7;margin:0;padding-left:18px}
a{color:var(--accent)}
</style></head><body>
<header><h1>75日線ブレイク手法のバックテスト</h1>
<div class="meta" id="meta"></div><div class="meta"><a href="index.html">← スクリーナーに戻る</a></div></header>
<div class="wrap">
<div class="card"><h2>試したルール</h2><ul class="rules">
<li><b>買い</b>：3ヶ月以上の下落のあと、終値が75日線を下から上に抜けた日（スクリーナーの「本日突破」と同じ条件・出来高50万株以上）の<b>翌日の始値</b></li>
<li><b>利確</b>：終値がボリンジャーバンド(25日)の<b>+1σより上に乗ったあと、+1σを下回ったら</b>翌日の始値で売る</li>
<li><b>損切り</b>：下の3通りを比較。どれも最長<span id="mh"></span>営業日（約半年）で売る</li>
<li>手数料・税金は含めていません。同じ銘柄を持っている間の新しい合図は無視しています。</li></ul></div>
<div class="tabs" id="tabs"></div>
<div class="verdict" id="verdict"></div>
<div class="card"><h2>成績</h2><div class="kpis" id="kpis"></div></div>
<div class="card"><h2>損益の積み上げ<span class="meta">（1回10万円ずつ、全部の合図で買った場合）</span></h2><div id="cum"></div></div>
<div class="card"><h2>1回ごとの損益の分布</h2><div id="hist"></div></div>
<div class="card"><h2>売った理由別</h2><div class="scroll"><table id="reason"></table></div></div>
<div class="card"><h2>年ごとの成績<span class="meta">（買った年）</span></h2><div class="scroll"><table id="year"></table></div></div>
<div class="card"><h2>最近の取引<span class="meta">（新しい順・最大300件）</span></h2><div class="scroll"><table id="trades"></table></div></div>
<div class="card"><h2>この結果を見るときの注意</h2><ul class="note">
<li>対象は<b>今</b>プライムに上場している銘柄だけです。この5年で上場廃止になった銘柄（業績が悪化した会社を含む）は入っていないため、成績は実際より良く出ている可能性があります。</li>
<li>始値でいつも買える・売れる前提です。実際は気配値が飛んだり、寄り付きで不利な値段になったりすることがあります。</li>
<li>同じ時期に合図がたくさん出ることがあります（例えば相場全体が反発したとき）。全部は買えない場合、結果は変わります。</li>
<li>過去の成績が良くても、これからも同じように勝てるとは限りません。売買のおすすめではありません。</li></ul></div>
</div>
<script>
const R=__DATA__;
const $=id=>document.getElementById(id);
const pct=(v,d=2)=>v==null?'—':(v>0?'+':'')+v.toFixed(d)+'%';
const cls=v=>v==null?'':v>0?'up':v<0?'down':'';
const yen=v=>(v>0?'+':'')+Math.round(v).toLocaleString('ja-JP')+'円';
$('meta').textContent=`検証期間のデータ：${R.data_from}〜${R.data_to}（東証プライム ${R.stocks}銘柄）｜作成：${R.generated}`;
$('mh').textContent=R.max_hold;
const keys=Object.keys(R.variants);let cur=keys[0];
function tabs(){$('tabs').innerHTML=keys.map(k=>{const s=R.variants[k];return `<button data-k="${k}" class="${k===cur?'on':''}"><b>${k}：${s.name}</b><span>${s.n?`勝率 ${s.win}%・平均 ${pct(s.avg)}`:'取引なし'}</span></button>`;}).join('');
  $('tabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{cur=b.dataset.k;tabs();render();});}
function verdict(s){
  if(!s.n)return 'この条件では取引がありませんでした。';
  const parts=[];
  parts.push(`約${Math.round((new Date(R.data_to)-new Date(s.first_entry))/3.156e10*10)/10}年間で<b>${s.n.toLocaleString()}回</b>の合図があり、<b>勝率は${s.win}%</b>でした。`);
  parts.push(`1回あたりの平均は<b class="${cls(s.avg)}">${pct(s.avg)}</b>（勝ったときの平均 ${pct(s.avg_win)}、負けたときの平均 ${pct(s.avg_loss)}）。`);
  if(s.excess_avg!=null)parts.push(`同じ期間TOPIX（1306）を持っていた場合と比べると、1回あたり<b class="${cls(s.excess_avg)}">${pct(s.excess_avg)}</b>で、TOPIXより成績が良かったのは${s.beat}%の取引でした。`);
  parts.push(s.avg>0&&(s.excess_avg??1)>0?'過去のデータでは、市場全体を上回るプラスの成績でした。':s.avg>0?'プラスではありますが、市場全体の上昇に乗っていた分が大きく、手法そのものの強みは小さめです。':'過去のデータでは、トータルでマイナスでした。');
  return parts.join('');
}
function kpi(k,v,s,c){return `<div class="kpi"><div class="k">${k}</div><div class="v ${c||''}">${v}</div>${s?`<div class="s">${s}</div>`:''}</div>`;}
function lineChart(pts){
  if(!pts.length)return '';
  const W=800,H=240,p=36,ys=pts.map(x=>x[1]*1000),mn=Math.min(0,...ys),mx=Math.max(0,...ys),rg=(mx-mn)||1;
  const X=i=>p+i*(W-p-8)/Math.max(1,pts.length-1),Y=v=>8+(mx-v)/rg*(H-30);
  const d=pts.map((x,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(x[1]*1000).toFixed(1)}`).join('');
  const last=ys[ys.length-1];
  let lab='';const yrs={};pts.forEach((x,i)=>{const y=x[0].slice(0,4);if(!(y in yrs))yrs[y]=i;});
  for(const [y,i] of Object.entries(yrs))lab+=`<text x="${X(i)}" y="${H-4}" font-size="11" fill="#8a8983">${y}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}"><line x1="${p}" x2="${W-8}" y1="${Y(0)}" y2="${Y(0)}" stroke="#d6d3cb"/>
  <text x="2" y="${Y(0)+4}" font-size="11" fill="#8a8983">0</text>
  <path d="${d}" fill="none" stroke="${last>=0?'#c8312b':'#1f5fbf'}" stroke-width="2"/>${lab}
  <text x="${W-8}" y="${Math.max(16,Y(last)-8)}" font-size="13" text-anchor="end" fill="#1d1d1b" font-weight="600">${yen(last)}</text></svg>`;
}
function histChart(h,b){
  const W=800,H=200,n=h.length,mx=Math.max(...h,1),bw=(W-20)/n;
  let s='';h.forEach((c,i)=>{const x=10+i*bw,hh=c/mx*(H-40),lo=b[i];s+=`<rect x="${x+1}" y="${H-24-hh}" width="${bw-2}" height="${hh}" fill="${lo>=0?'rgba(200,49,43,.55)':'rgba(31,95,191,.55)'}"/>`;});
  [-30,-20,-10,0,10,20,30].forEach(v=>{const i=b.indexOf(v);const x=10+(i<0?n:i)*bw;s+=`<text x="${x}" y="${H-8}" font-size="11" text-anchor="middle" fill="#8a8983">${v>0?'+':''}${v}%</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}">${s}</svg><div class="meta">端の棒は、−30%以下・+30%以上をまとめています。</div>`;
}
function render(){
  const s=R.variants[cur];$('verdict').innerHTML=verdict(s);
  if(!s.n){['kpis','cum','hist','reason','year','trades'].forEach(id=>$(id).innerHTML='');return;}
  $('kpis').innerHTML=kpi('取引回数',s.n.toLocaleString(),'')+kpi('勝率',s.win+'%','')+kpi('平均損益',pct(s.avg),'中央値 '+pct(s.median),cls(s.avg))
    +kpi('平均利益',pct(s.avg_win),'勝ったとき','up')+kpi('平均損失',pct(s.avg_loss),'負けたとき','down')
    +kpi('損益比',s.payoff??'—','平均利益÷平均損失')+kpi('プロフィットファクター',s.pf??'—','利益合計÷損失合計（1超で黒字）')
    +kpi('平均保有',s.days+'日','営業日')+kpi('TOPIXとの差',pct(s.excess_avg),`TOPIXに勝った割合 ${s.beat??'—'}%`,cls(s.excess_avg))
    +kpi('最大連敗',s.max_streak+'回','')+kpi('+1σに乗った割合',s.touched+'%','利確の条件を満たせた取引')
    +kpi('最大の勝ち／負け',pct(s.best,1),pct(s.worst,1));
  $('cum').innerHTML=lineChart(s.cum);
  $('hist').innerHTML=histChart(s.hist,s.bins);
  $('reason').innerHTML='<tr><th>売った理由</th><th>回数</th><th>勝率</th><th>平均損益</th><th>平均保有</th></tr>'+s.by_reason.map(r=>`<tr><td class="t">${r.reason}</td><td>${r.n}</td><td>${r.win}%</td><td class="${cls(r.avg)}">${pct(r.avg)}</td><td>${r.days}日</td></tr>`).join('');
  $('year').innerHTML='<tr><th>年</th><th>回数</th><th>勝率</th><th>平均損益</th><th>TOPIX</th><th>損益合計</th></tr>'+s.by_year.map(r=>`<tr><td>${r.year}</td><td>${r.n}</td><td>${r.win}%</td><td class="${cls(r.avg)}">${pct(r.avg)}</td><td class="${cls(r.bench)}">${pct(r.bench)}</td><td class="${cls(r.sum)}">${yen(r.sum*1000)}</td></tr>`).join('');
  const RS={take:'利確',stop:'損切り',time:'期間満了'};
  $('trades').innerHTML='<tr><th>買った日</th><th>銘柄</th><th>買値</th><th>売った日</th><th>売値</th><th>損益</th><th>理由</th></tr>'+(s.recent||[]).map(t=>`<tr><td>${t.entry}</td><td class="t">${t.code} ${t.name}</td><td>${t.buy.toLocaleString()}</td><td>${t.exit}</td><td>${t.sell.toLocaleString()}</td><td class="${cls(t.ret)}">${pct(t.ret,1)}</td><td class="t">${RS[t.reason]}</td></tr>`).join('');
}
tabs();render();
</script></body></html>"""


def build(result: dict, path: str) -> None:
    data = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", data.replace("</", "<\\/")))
