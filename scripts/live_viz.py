"""Live Wordle visualizer (driver, not committed): watch the model play + the curve, in a browser.

A tiny stdlib HTTP server. A background thread auto-picks the newest checkpoint (runs/*.pt) and replays
a FIXED set of 10 held-out games (greedy ephemeral-CoT) whenever the checkpoint changes — so the boards
update as training saves new bests. The win-rate curve is parsed live from the newest run log. The page
polls /data and renders the 10 Wordle boards + an SVG curve. Honest: greedy, held-out, no inference rules.

    uv run python scripts/live_viz.py            # newest ckpt + newest log, port 8765
    uv run python scripts/live_viz.py runs/dpo.pt runs/dpogo.log 8800
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
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import load_checkpoint

DEV = "mps"
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1  # CoT models (vocab 35)
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
_NAME = {Color.GREEN: "green", Color.YELLOW: "yellow", Color.GRAY: "gray"}
ROWS = 10
_, HELD = split(seed=0)
VIZ_SECRETS = list(HELD[:10])  # fixed 10 held-out games — watch them get solved as training improves

STATE: dict = {"games": [], "curve": [], "ckpt": "", "log": "", "metric": "win", "ts": 0, "status": "starting…"}
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
        fb = None if t.feedback is None else [_NAME[c] for c in t.feedback]
        turns.append({"guess": t.guess, "fb": fb})
    return {"secret": g.secret, "status": g.status.value, "used": g.guesses_used, "turns": turns}


_METRIC_RE = re.compile(r"(held10|held6|win)\s*[=:]\s*([0-9]*\.?[0-9]+)")


def parse_curve(logpath):
    """Pull a win-rate series from a run log: prefer held10, else held6, else win=."""
    if not logpath or not os.path.exists(logpath):
        return [], "win"
    pref = None
    series_by = {"held10": [], "held6": [], "win": []}
    try:
        with open(logpath) as f:
            for line in f:
                for key, val in _METRIC_RE.findall(line):
                    series_by[key].append(float(val))
    except OSError:
        return [], "win"
    for key in ("held10", "held6", "win"):
        if series_by[key]:
            pref = key
            return [{"i": i, "v": v} for i, v in enumerate(series_by[key])], pref
    return [], "win"


def newest(patterns):
    files = []
    for p in patterns:
        files += glob.glob(p)
    files = [f for f in files if os.path.getsize(f) > 0]
    return max(files, key=os.path.getmtime) if files else None


def worker(fixed_ckpt, fixed_log):
    model, loaded = None, None
    while True:
        ckpt = fixed_ckpt or newest(["runs/*.pt"])
        log = fixed_log or newest(["runs/*.log"])
        curve, metric = parse_curve(log)
        try:
            sig = (ckpt, os.path.getmtime(ckpt)) if ckpt else None
            if ckpt and sig != loaded:
                m = WordleGenerator(CFG, VOCAB).to(DEV)
                load_checkpoint(ckpt, m)
                m.eval()
                model, loaded = m, sig
                games = [game_json(play(model, s)) for s in VIZ_SECRETS]
                wins = sum(g["status"] == "win" for g in games)
                with LOCK:
                    STATE.update(games=games, ckpt=os.path.basename(ckpt),
                                 status=f"{wins}/10 of these games won by the current model")
            with LOCK:
                STATE.update(curve=curve, metric=metric, log=os.path.basename(log) if log else "—",
                             ts=time.time())
        except Exception as e:  # mid-write checkpoint / transient — keep last good state
            with LOCK:
                STATE.update(curve=curve, metric=metric, status=f"waiting ({type(e).__name__})", ts=time.time())
        time.sleep(8)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Wordle SLM — live</title>
<style>
 body{background:#121213;color:#d7dadc;font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;padding:18px}
 h1{font-size:18px;margin:0 0 2px}.sub{color:#818384;font-size:12px;margin-bottom:14px}
 .wrap{display:flex;gap:24px;flex-wrap:wrap}
 .panel{background:#1a1a1b;border:1px solid #3a3a3c;border-radius:10px;padding:14px}
 .games{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;flex:1;min-width:520px}
 .game{}.gh{font-size:12px;margin-bottom:5px;letter-spacing:1px}
 .win .gh{color:#6aaa64}.lose .gh{color:#c9605b}
 .row{display:flex;gap:3px;margin-bottom:3px}
 .t{width:26px;height:26px;border-radius:3px;display:flex;align-items:center;justify-content:center;
    font-weight:700;font-size:13px;text-transform:uppercase;color:#fff;background:#3a3a3c}
 .green{background:#538d4e}.yellow{background:#b59f3b}.gray{background:#3a3a3c}
 .bad{background:#2a2a2b;color:#c9605b;border:1px dashed #c9605b}
 .legend{font-size:12px;color:#818384;margin-top:8px}
 svg{background:#0e0e0f;border-radius:6px}
</style></head><body>
<h1>Wordle SLM — live</h1>
<div class=sub id=sub>connecting…</div>
<div class=wrap>
 <div class="panel" style="flex:0 0 auto">
   <div style="font-size:13px;margin-bottom:6px">learning curve (<span id=metric>win</span>) — <span id=logname></span></div>
   <svg id=chart width=460 height=300></svg>
   <div class=legend id=curvelegend></div>
 </div>
 <div class="panel games" id=games></div>
</div>
<script>
const TILE=(l,c)=>`<div class="t ${c||'gray'}">${l||''}</div>`;
function board(g){
  let rows='';
  for(const t of g.turns){
    let r='<div class=row>';
    for(let i=0;i<5;i++){
      const l=t.guess[i]||'';
      if(t.fb===null) r+=`<div class="t bad">${l}</div>`;
      else r+=TILE(l, t.fb[i]);
    }
    rows+=r+'</div>';
  }
  const cls=g.status==='win'?'win':(g.status==='lose'?'lose':'');
  const tag=g.status==='win'?`✓ ${g.used}`:(g.status==='lose'?'✗':'…');
  return `<div class="game ${cls}"><div class=gh>${g.secret} ${tag}</div>${rows}</div>`;
}
function chart(curve,metric){
  const W=460,H=300,P=34;const svg=document.getElementById('chart');
  if(!curve.length){svg.innerHTML='';return;}
  const xs=curve.map(p=>p.i),ys=curve.map(p=>p.v);
  const xmax=Math.max(1,...xs),ymax=Math.max(0.1,...ys,...[0.7]);
  const X=i=>P+(W-2*P)*(i/xmax),Y=v=>H-P-(H-2*P)*(v/ymax);
  let grid='';
  for(let k=0;k<=5;k++){const v=ymax*k/5;grid+=`<line x1=${P} y1=${Y(v)} x2=${W-P} y2=${Y(v)} stroke=#2a2a2b/>`+
    `<text x=4 y=${Y(v)+4} fill=#818384 font-size=10>${v.toFixed(2)}</text>`;}
  let d='';curve.forEach((p,k)=>{d+=(k?'L':'M')+X(p.i)+' '+Y(p.v)+' ';});
  let dots=curve.map(p=>`<circle cx=${X(p.i)} cy=${Y(p.v)} r=2.5 fill=#6aaa64/>`).join('');
  const last=ys[ys.length-1];
  svg.innerHTML=grid+`<path d="${d}" fill=none stroke=#6aaa64 stroke-width=2/>`+dots+
    `<text x=${W-P} y=${Y(last)-8} fill=#6aaa64 font-size=12 text-anchor=end>${last.toFixed(3)}</text>`;
  document.getElementById('curvelegend').textContent=`${curve.length} eval points · latest ${metric}=${last.toFixed(3)}`;
}
async function tick(){
  try{
    const d=await (await fetch('/data')).json();
    document.getElementById('sub').textContent=
      `checkpoint: ${d.ckpt||'—'} · ${d.status} · curve from ${d.log} · updated ${new Date(d.ts*1000).toLocaleTimeString()}`;
    document.getElementById('metric').textContent=d.metric;
    document.getElementById('logname').textContent=d.log||'';
    document.getElementById('games').innerHTML=(d.games||[]).map(board).join('');
    chart(d.curve||[],d.metric);
  }catch(e){document.getElementById('sub').textContent='waiting for server…';}
}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
    print(f"     (newest ckpt + newest runs/*.log; replays 10 held-out games on each new checkpoint)\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
