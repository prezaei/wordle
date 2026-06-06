"""Live Wordle visualizer (driver, not committed): watch the model play + the learning curve, in a browser.

A tiny stdlib HTTP server. A background thread auto-follows the ACTIVE training run (the newest
freshly-written runs/*.pt + runs/*.log), replays a fixed set of 10 held-out games (greedy
ephemeral-CoT) whenever the checkpoint changes, and parses the win-rate curve live from the log. The
page renders the 10 Wordle boards + a canvas line chart, polling /data. Honest: greedy, held-out, no
inference rules.

    uv run python scripts/live_viz.py                 # auto-follow the active run (fallback: best ckpt)
    uv run python scripts/live_viz.py runs/dpo.pt runs/dpo.log 8800   # pin a specific ckpt/log/port
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from wordle_slm.config import ModelConfig
from wordle_slm.data import split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.scoring import score
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

from viz_progress import read_progress  # scripts/ helper: per-epoch / per-update boards + grades

DEV = "mps"
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
_NAME = {Color.GREEN: "green", Color.YELLOW: "yellow", Color.GRAY: "gray"}
ROWS = 6
_, HELD = split(seed=0)
VIZ_SECRETS = list(HELD[:10])
REFS = [("clean-SFT 0.166", 0.166)]  # honest held-out floor (overnight clean run); fair run should beat it

STATE: dict = {"games": [], "curve": [], "curve2": [], "ckpt": "—", "log": "—", "metric": "win",
               "ts": 0, "status": "starting…", "refs": REFS}
LOCK = threading.Lock()


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


@torch.no_grad()
def play(model, secret, rows=ROWS):
    g = Game(secret, max_guesses=rows)
    while g.status is Status.ONGOING:
        seq = board_only(g.turns)
        guess, committed = [], False
        for _ in range(48):
            nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
            seq.append(nxt)
            if committed:
                if nxt in LETTER_SET:
                    guess.append(nxt)
                if len(guess) >= 5:
                    break
            elif nxt == tok.guess_id:
                committed = True
        word = "".join(tok.id_to_token(t) for t in guess[:5])
        g.guess(word if len(word) == 5 else "zzzzz")
    return g


def game_json(g):
    turns = []
    for t in g.turns:
        if t.feedback is None:  # invalid word: no real feedback, but show what it WOULD score (ghost)
            turns.append({"guess": t.guess, "fb": None, "ghost": [_NAME[c] for c in score(t.guess, g.secret)]})
        else:
            turns.append({"guess": t.guess, "fb": [_NAME[c] for c in t.feedback]})
    return {"secret": g.secret, "status": g.status.value, "used": g.guesses_used, "turns": turns}


_METRIC_RE = re.compile(r"(held10|held6|win)\s*[=:]\s*([0-9]*\.?[0-9]+)")


def parse_curve(logpath):
    if not logpath or not os.path.exists(logpath):
        return [], "win"
    series = {"held10": [], "held6": [], "win": []}
    try:
        with open(logpath) as f:
            for line in f:
                for key, val in _METRIC_RE.findall(line):
                    series[key].append(float(val))
    except OSError:
        return [], "win"
    for key in ("held10", "held6", "win"):
        if series[key]:
            return [{"i": i, "v": v} for i, v in enumerate(series[key])], key
    return [], "win"


def _logs():
    return [f for f in glob.glob("runs/*.log") if os.path.basename(f) not in ("viz.log",) and os.path.getsize(f) > 0]


def newest(files):
    return max(files, key=os.path.getmtime) if files else None


def newest_fresh(files, max_age=2400):
    f = newest(files)
    return f if (f and time.time() - os.path.getmtime(f) < max_age) else None


BEST_PRIORITY = ["runs/dpo.pt", "runs/cot_eph_aux.pt", "runs/rl_expert.pt", "runs/cot_eph.pt"]


def best_ckpt():
    for p in BEST_PRIORITY:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return newest(glob.glob("runs/*.pt"))


def progress_state(prog):
    """Build the dashboard STATE from a per-epoch/-update progress file (the actual logged
    inferences + grades) — no model reload, no re-play, so it never contends with training."""
    recs = read_progress(prog)
    if not recs:
        return None
    latest = recs[-1]
    metric = "win" if latest.get("win") is not None else "reward_mean"  # SFT: win · RL: reward_mean
    curve = [{"i": r["epoch"], "v": r[metric]} for r in recs if r.get(metric) is not None]
    curve2 = [{"i": r["epoch"], "v": r["valid"]} for r in recs if r.get("valid") is not None]  # validity line
    refs = REFS if metric == "win" else []  # the 0.166 ref is a win rate — hide it on a reward curve
    won = sum(g.get("status") == "win" for g in latest["games"])
    val = latest.get(metric)
    xlabel = "update" if latest.get("kind") == "rl" else "epoch"
    extra = f" · eval_win={latest['eval_win']:.3f}" if latest.get("eval_win") is not None else ""
    status = (f"{latest.get('kind', 'sft')} {xlabel} {latest['epoch']} · "
              f"{metric}={val:.3f}{extra} · {won}/{len(latest['games'])} shown won"
              if val is not None else f"{xlabel} {latest['epoch']}")
    return {"games": latest["games"], "curve": curve, "curve2": curve2, "metric": metric, "refs": refs,
            "ckpt": os.path.basename(prog), "log": os.path.basename(prog), "status": status}


def worker(fixed_ckpt, fixed_log):
    model, sig, off = None, None, 0
    pool = list(HELD)  # rotate through the full held-out set so the page shows fresh games each cycle
    while True:
        # Prefer the active run's progress file (real per-epoch/-update boards + grades) when unpinned.
        if not fixed_ckpt:
            prog = newest_fresh(glob.glob("runs/*_progress.jsonl"))
            ps = progress_state(prog) if prog else None
            if ps:
                with LOCK:
                    STATE.update(**ps, ts=time.time())
                time.sleep(3)
                continue
        ckpt = fixed_ckpt or newest_fresh(glob.glob("runs/*.pt")) or best_ckpt()
        log = fixed_log or newest_fresh(_logs()) or newest(_logs())
        curve, metric = parse_curve(log)
        try:
            cur_sig = (ckpt, os.path.getmtime(ckpt)) if ckpt else None
            if ckpt and cur_sig != sig:
                m = WordleGenerator(CFG, VOCAB).to(DEV)
                load_checkpoint(ckpt, m)
                m.eval()
                model, sig = m, cur_sig
            if model:  # play 10 NEW (rotating) held-out games every cycle -> live updates
                secs = [pool[(off + i) % len(pool)] for i in range(10)]
                off = (off + 10) % len(pool)
                games = [game_json(play(model, s)) for s in secs]
                wins = sum(g["status"] == "win" for g in games)
                with LOCK:
                    STATE.update(games=games, status=f"{wins}/10 won (rotating held-out)")
            with LOCK:
                STATE.update(curve=curve, metric=metric, ckpt=os.path.basename(ckpt) if ckpt else "—",
                             log=os.path.basename(log) if log else "—", ts=time.time())
        except Exception as e:  # mid-write checkpoint / transient
            with LOCK:
                STATE.update(curve=curve, metric=metric, status=f"waiting ({type(e).__name__})", ts=time.time())
        time.sleep(3)


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Wordle SLM — live</title>
<style>
 *{box-sizing:border-box}
 body{background:#0e0e0f;color:#d7dadc;font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;padding:20px}
 h1{font-size:20px;margin:0 0 3px;letter-spacing:.3px}
 .sub{color:#9296a0;font-size:12.5px;margin-bottom:16px}
 .sub b{color:#6aaa64}
 .wrap{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap}
 .panel{background:#1a1a1b;border:1px solid #343437;border-radius:12px;padding:16px}
 .ctitle{font-size:13px;color:#d7dadc;margin-bottom:8px}.ctitle b{color:#6aaa64}
 .games{display:grid;grid-template-columns:repeat(5,1fr);gap:18px 16px;flex:1;min-width:560px}
 .gh{font-size:12.5px;margin-bottom:5px;letter-spacing:1.5px;font-weight:600}
 .win .gh{color:#6aaa64}.lose .gh{color:#d16b66}.ongoing .gh{color:#9296a0}
 .row{display:flex;gap:4px;margin-bottom:4px}
 .t{width:30px;height:30px;border-radius:4px;display:flex;align-items:center;justify-content:center;
    font-weight:700;font-size:15px;text-transform:uppercase;color:#fff;background:#3a3a3c;
    transition:background .2s}
 .green{background:#538d4e}.yellow{background:#b59f3b}.gray{background:#3a3a3c}
 /* invalid word: dashed border (= not a real word) + FADED ghost color (what it would have scored) */
 .bad{background:#231f20;color:#d99;border:1.5px dashed #6a4a4a}
 .bad.green{background:rgba(83,141,78,.40);border-color:#538d4e;color:#dfeede}
 .bad.yellow{background:rgba(181,159,59,.40);border-color:#b59f3b;color:#f0e9cf}
 .bad.gray{background:#231f20;border-color:#5a565a;color:#b89}
 .legend{font-size:11.5px;color:#9296a0;margin-top:10px}
 .pill{display:inline-block;background:#26262a;border-radius:10px;padding:2px 9px;margin-right:6px;font-size:11.5px}
</style></head><body>
<h1>Wordle SLM — live</h1>
<div class=sub id=sub>connecting…</div>
<div class=wrap>
 <div class="panel" style="flex:0 0 auto">
   <div class=ctitle>learning curve · <b id=metric>win</b> <span id=logname style="color:#9296a0"></span></div>
   <canvas id=chart></canvas>
   <div class=legend id=curvelegend></div>
 </div>
 <div class="panel games" id=games></div>
</div>
<script>
const COL={green:'#538d4e',yellow:'#b59f3b',gray:'#3a3a3c'};
function board(g){
  let rows='';
  for(const t of g.turns){
    let r='<div class=row>';
    for(let i=0;i<5;i++){
      const l=t.guess[i]||'';
      if(t.fb===null){const gc=(t.ghost&&t.ghost[i])||'gray';r+=`<div class="t bad ${gc}" title="not a real word">${l}</div>`;}
      else r+=`<div class="t ${t.fb[i]}">${l}</div>`;
    }
    rows+=r+'</div>';
  }
  const cls=g.status==='win'?'win':(g.status==='lose'?'lose':'ongoing');
  const tag=g.status==='win'?`✓ ${g.used}`:(g.status==='lose'?'✗':'…');
  let grade='';  // RL rollouts carry a grade: reward (the score) + advantage (group-relative)
  if(g.reward!==undefined&&g.reward!==null){
    const c=g.reward>=0?'#6aaa64':'#d16b66';
    grade=` <span style="color:${c};font-weight:700" title="reward (grade)">r=${g.reward.toFixed(2)}</span>`;
    if(g.adv!==undefined&&g.adv!==null){const ac=g.adv>=0?'#7a9':'#a77';
      grade+=` <span style="color:${ac};font-size:11px" title="group-relative advantage">A=${g.adv.toFixed(2)}</span>`;}
  }
  return `<div class="${cls}"><div class=gh>${g.secret} ${tag}${grade}</div>${rows}</div>`;
}
const CV=document.getElementById('chart'),CTX=CV.getContext('2d'),DPR=window.devicePixelRatio||1,CW=540,CH=350;
CV.style.width=CW+'px';CV.style.height=CH+'px';CV.width=CW*DPR;CV.height=CH*DPR;CTX.scale(DPR,DPR);
function chart(curve,metric,refs,curve2){
  const ctx=CTX,W=CW,H=CH; ctx.clearRect(0,0,W,H);
  const Lm=46,Rm=W-16,Tm=16,Bm=H-26;
  if(!curve.length){ctx.fillStyle='#666';ctx.font='12px sans-serif';ctx.fillText('waiting for eval points…',Lm,H/2);return;}
  const xs=curve.map(p=>p.i),ys=curve.map(p=>p.v),refv=refs.map(r=>r[1]);
  const ys2=(curve2||[]).map(p=>p.v);  // validity series (shares the 0..1 axis)
  const xmax=Math.max(1,...xs);
  let lo=Math.min(...ys,...ys2,...refv),hi=Math.max(...ys,...ys2,...refv),pad=Math.max(0.02,(hi-lo)*0.2);
  let ymin=Math.max(0,lo-pad),ymax=Math.min(1,hi+pad);
  if(ymax-ymin<0.12){const c=(ymax+ymin)/2;ymin=Math.max(0,c-0.06);ymax=Math.min(1,c+0.06);}  // zoom in
  const X=i=>Lm+(Rm-Lm)*(i/xmax), Y=v=>Bm-(Bm-Tm)*((v-ymin)/(ymax-ymin));
  ctx.font='10px -apple-system,sans-serif';ctx.textBaseline='middle';ctx.textAlign='left';
  for(let k=0;k<=5;k++){const v=ymin+(ymax-ymin)*k/5,y=Y(v);
    ctx.strokeStyle='#27272a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(Lm,y);ctx.lineTo(Rm,y);ctx.stroke();
    ctx.fillStyle='#7a7d85';ctx.fillText(v.toFixed(2),6,y);}
  for(const [name,v] of refs){ if(v<ymin||v>ymax)continue; const y=Y(v);  // ref lines, left-labeled
    ctx.strokeStyle='#4a4750';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(Lm,y);ctx.lineTo(Rm,y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#8a8792';ctx.fillText(name,Lm+5,y-7);}
  ctx.fillStyle='#7a7d85';ctx.textBaseline='top';ctx.textAlign='center';
  const step=Math.max(1,Math.round(xmax/8));
  for(let i=0;i<=xmax;i+=step)ctx.fillText(i,X(i),Bm+6);
  ctx.textAlign='left';
  const g=ctx.createLinearGradient(0,Tm,0,Bm);g.addColorStop(0,'rgba(106,170,100,.30)');g.addColorStop(1,'rgba(106,170,100,0)');
  ctx.beginPath();curve.forEach((p,k)=>{const x=X(p.i),y=Y(p.v);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.lineTo(X(xs[xs.length-1]),Bm);ctx.lineTo(X(xs[0]),Bm);ctx.closePath();ctx.fillStyle=g;ctx.fill();
  ctx.strokeStyle='#6aaa64';ctx.lineWidth=2.5;ctx.lineJoin='round';ctx.beginPath();
  curve.forEach((p,k)=>{const x=X(p.i),y=Y(p.v);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle='#8fd47f';for(const p of curve){ctx.beginPath();ctx.arc(X(p.i),Y(p.v),3,0,7);ctx.fill();}
  if(curve2&&curve2.length){  // validity rate, second line (blue) — the spelling signal
    ctx.strokeStyle='#5a8fd4';ctx.lineWidth=2;ctx.lineJoin='round';ctx.beginPath();
    curve2.forEach((p,k)=>{const x=X(p.i),y=Y(p.v);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    ctx.fillStyle='#8fb8e8';for(const p of curve2){ctx.beginPath();ctx.arc(X(p.i),Y(p.v),2.5,0,7);ctx.fill();}
  }
  const last=ys[ys.length-1],lx=X(xmax),ly=Y(last),txt=last.toFixed(3);
  ctx.font='bold 13px -apple-system,sans-serif';const tw=ctx.measureText(txt).width;
  let bx=lx-tw-12,by=ly-9; if(bx<Lm+2)bx=lx+8; if(by<Tm)by=Tm; if(by>Bm-18)by=Bm-18;
  ctx.fillStyle='#173a17';ctx.fillRect(bx-4,by-2,tw+8,18);
  ctx.fillStyle='#8fd47f';ctx.textBaseline='top';ctx.fillText(txt,bx,by);
  const v2=(curve2&&curve2.length)?` <span class=pill>valid=<b style="color:#8fb8e8">${curve2[curve2.length-1].v.toFixed(3)}</b></span>`:'';
  document.getElementById('curvelegend').innerHTML=
    `<span class=pill>${curve.length} pts</span><span class=pill>${metric}=<b style="color:#8fd47f">${last.toFixed(3)}</b></span>${v2}`;
}
async function tick(){
  try{
    const d=await (await fetch('/data')).json();
    const won=(d.games||[]).filter(g=>g.status==='win').length, n=(d.games||[]).length;
    document.getElementById('sub').innerHTML=
      `checkpoint <b>${d.ckpt}</b> · <b>${won}/${n||10}</b> games won · curve from ${d.log} · updated ${new Date(d.ts*1000).toLocaleTimeString()}`;
    document.getElementById('metric').textContent=d.metric;
    document.getElementById('logname').textContent='· '+d.log;
    document.getElementById('games').innerHTML=(d.games||[]).map(board).join('');
    chart(d.curve||[],d.metric,d.refs||[],d.curve2||[]);
  }catch(e){document.getElementById('sub').textContent='waiting for server…';}
}
tick();setInterval(tick,2500);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            with LOCK:
                body = json.dumps(STATE).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    fixed_ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    fixed_log = sys.argv[2] if len(sys.argv) > 2 else None
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
    threading.Thread(target=worker, args=(fixed_ckpt, fixed_log), daemon=True).start()
    while True:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    print(f"\n  ►  Wordle SLM live viz:  http://127.0.0.1:{port}\n", flush=True)
    print("     auto-follows the active run (newest fresh ckpt+log); replays 10 held-out games on each new checkpoint\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
