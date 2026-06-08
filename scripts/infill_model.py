"""Green-conditioned infill model (user's idea, in-weights): pin the greens, generate only the blanks.

Each turn is serialized as  board → GREEN-TEMPLATE → think → <GUESS> → 5 letters.  The template is 5
tokens, one per position: the known green letter, or <gray> as a BLANK placeholder (e.g. HOIST greens
O@2/I@3/T@5 → `_ O I _ T`). The loss is masked to ONLY the blank guess-positions (greens are given, not
predicted), so the model spends capacity learning to FILL blanks conditioned on the whole template. At
inference we build the template from the board's greens, FORCE the greens, and greedily generate only the
blanks. This (a) makes green-violations impossible and (b) conditions position-1 on the full word shape
(`_OI_T`), biasing toward real completions (POINT/JOINT) over non-words (DOINT).

Honest: greens are the model's OWN earned clues (not a dictionary); generation is free over the blanks;
eval is the clean protocol (non-words counted as a turn, NOT fed back; no dictionary). Warm-start stage-1.
Env: IF_AUX (aux lambda, default 3.0), IF_EPOCHS (default 18), IF_LR (default 1.5e-4), IF_SECRETS (1200),
IF_TEACHER (passes, default 3), IF_BASE, IF_OUT. -> runs/infill.pt
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

from viz_progress import append_epoch
from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_valid_guesses, split
from wordle_slm.engine import Color
from wordle_slm.engine.game import Turn
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.engine.scoring import score
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(0)
tok = Tokenizer()
THINK = tok.vocab_size
BLANK = tok.vocab_size + 1  # dedicated template blank token (not the overloaded <gray>=absent)
VOCAB = tok.vocab_size + 2
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
OPENERS = ("salet", "crane", "slate", "trace", "stare", "raise", "crate")
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
VALID = load_valid_guesses()
K_CANDS = 3
AUX_LAMBDA = float(os.environ.get("IF_AUX", "6.0"))  # combine the proven validity lever (was 3)
EPOCHS = int(os.environ.get("IF_EPOCHS", "18"))
LR = float(os.environ.get("IF_LR", "1.5e-4"))
CAP = int(os.environ.get("IF_SECRETS", "1200"))
TEACHER_PASSES = int(os.environ.get("IF_TEACHER", "3"))
BASE = os.environ.get("IF_BASE", "runs/cot_eph_aux_fair.pt")
OUT = os.environ.get("IF_OUT", "runs/infill.pt")
SQ = {Color.GREEN: "🟩", Color.YELLOW: "🟨", Color.GRAY: "⬜"}


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.token_to_id(_COLOR[c]) for c in turn.feedback]  # only valid turns reach here


def board_only(turns):
    ids = [tok.bos_id]
    for turn in turns:
        ids += [tok.guess_id, *_letters(turn.guess), *_fb(turn), tok.sep_id]
    return ids


def known_greens(turns):
    g = {}
    for t in turns:
        if t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            if c is Color.GREEN:
                g[i] = t.guess[i]
    return g


def template_tokens(greens):
    """5 tokens: green letter where known, else BLANK placeholder."""
    return [tok.token_to_id(greens[p]) if p in greens else BLANK for p in range(5)]


def pick_cands(history, guess, rng):
    cands = [guess]
    for _ in range(300):
        if len(cands) >= K_CANDS:
            break
        w = rng.choice(VALID)
        if w != guess and w not in cands and is_consistent(w, history):
            cands.append(w)
    rng.shuffle(cands)
    return cands


def build_example(game, rng):
    """board -> TEMPLATE -> think -> <GUESS> -> 5 letters. Loss on think + BLANK guess-positions only."""
    exs = []
    for k, turn in enumerate(game.turns):
        greens = known_greens(game.turns[:k])
        ids = board_only(game.turns[:k])
        mask = [False] * len(ids)
        ids += template_tokens(greens)  # the green template (context, no loss)
        mask += [False] * 5
        for c in pick_cands(game.turns[:k], turn.guess, rng):  # think candidates (loss)
            ids.append(THINK)
            mask.append(True)
            ids += _letters(c)
            mask += [True] * 5
        ids.append(tok.guess_id)
        mask.append(False)
        for pos in range(5):  # the committed word: loss ONLY on blanks (greens are given/forced)
            ids.append(tok.token_to_id(turn.guess[pos]))
            mask.append(pos not in greens)
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


def cot_valid_mask(seqs):
    trie = _valid_trie()
    L = max(len(s) for s in seqs)
    vmask = torch.zeros((len(seqs), L, 26))
    for i, seq in enumerate(seqs):
        for t, tokid in enumerate(seq):
            if tokid not in WORD_STARTS or t + 5 >= len(seq):
                continue
            node = trie
            for j in range(5):
                for child in node:
                    vmask[i, t + j, child] = 1.0
                nxt = seq[t + 1 + j] - LETTER_LO
                node = node.get(nxt, {}) if 0 <= nxt < 26 else {}
    return vmask


@torch.no_grad()
def play_infill(model, secret):
    """Clean protocol: think+template, FORCE greens, generate blanks; non-words counted, not fed back."""
    visible = []
    for attempt in range(1, 7):
        greens = known_greens(visible)
        seq = board_only(visible) + template_tokens(greens)
        # think phase: free-gen until <GUESS>
        for _ in range(60):
            nxt = int(ALLOWED_GEN[int(torch.argmax(model.forward(torch.tensor([seq], device=DEV))[0, -1][ALLOWED_GEN]))])
            seq.append(nxt)
            if nxt == tok.guess_id:
                break
        else:
            seq.append(tok.guess_id)
        # commit: force greens, greedily fill blanks
        letters = []
        for pos in range(5):
            if pos in greens:
                tid = tok.token_to_id(greens[pos])
            else:
                logits = model.forward(torch.tensor([seq], device=DEV))[0, -1]
                tid = int(LETTER_IDS[int(torch.argmax(logits[LIDS]))])
            letters.append(tok.id_to_token(tid))
            seq.append(tid)
        word = "".join(letters)
        if len(word) != 5 or not is_valid(word):
            return {"won": False, "guesses": attempt, "reason": "nonword"}
        fb = score(word, secret)
        visible.append(Turn(guess=word, feedback=fb, valid=True))
        if all(c is Color.GREEN for c in fb):
            return {"won": True, "guesses": attempt, "reason": "win"}
    return {"won": False, "guesses": 6, "reason": "exhausted"}


def clean_metrics(model, secrets):
    res = [play_infill(model, s) for s in secrets]
    wins = [r for r in res if r["won"]]
    return {"win": len(wins) / len(res),
            "nonword": sum(1 for r in res if r["reason"] == "nonword"),
            "exhausted": sum(1 for r in res if r["reason"] == "exhausted"),
            "avg": statistics.mean(r["guesses"] for r in wins) if wins else float("nan")}


def main():
    train, held = split(seed=0)
    safe_openers = tuple(o for o in OPENERS if o not in set(held))
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    secrets = tuple(train[:CAP])
    PROG = "runs/infill_progress.jsonl"
    if os.path.exists(PROG):
        os.remove(PROG)
    print(f"[infill] aux={AUX_LAMBDA} epochs={EPOCHS} lr={LR} |secrets|={len(secrets)} base={BASE}", flush=True)
    model = WordleGenerator(CFG, VOCAB).to(DEV)
    if BASE == "scratch":  # FULL retrain: spell warm-up, then learn the template format natively
        from wordle_slm.sft import pretrain_lm, pretrain_words
        pre = int(os.environ.get("IF_PRETRAIN", "30"))
        print(f"[infill] scratch: spell warm-up (pretrain_lm {pre} ep) ...", flush=True)
        pretrain_lm(model, pretrain_words(), tok, SFTConfig(lr=1e-3), epochs=pre, batch_size=256, device=DEV, seed=0)
    else:
        load_checkpoint(BASE, model)
        base = clean_metrics(model, VAL)
        print(f"[infill] warm-start clean-VAL win={base['win']:.3f} (likely ~0 — template is OOD for stage-1)", flush=True)

    rng = Random(0)
    games = []
    for s in range(TEACHER_PASSES):
        games += [tr.game for tr in generate_transcripts(
            secrets, weak_frac=0.5, openers=safe_openers, seed=200 + s, valid_pool=VALID, answer_pool=secrets)]
    exs = [e for g in games for e in build_example(g, rng)]
    rng.shuffle(exs)
    print(f"[infill] teacher games={len(games)} -> examples={len(exs)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best, saved = -1.0, False
    VIZ = tuple(held[:24])
    model.train()
    print("[infill] epoch  clean-VAL win / nonword-losses", flush=True)
    for epoch in range(EPOCHS):
        for idx in _batches(len(exs), 128, rng2):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            vmask = cot_valid_mask([s for s, _ in bs]).to(DEV)
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vm = vmask[:, :-1]
            aux_pos = (vm.sum(-1) > 0).float() * tmask[:, 1:]
            valid_mass = (logp_let.exp() * vm).sum(-1).clamp_min(1e-9)
            aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:  # clean-VAL eval is slow; every 5 epochs
            model.eval()
            m = clean_metrics(model, VAL)
            flag = ""
            if m["win"] > best:
                best = m["win"]
                save_checkpoint(OUT, model, opt, epoch, SFTConfig())
                saved = True
                flag = "  <- best, saved"
            print(f"  epoch {epoch:>2}  loss={float(loss.detach()):.3f}  clean-VAL win {m['win']:.3f}  "
                  f"nonword-losses {m['nonword']}{flag}", flush=True)
            model.train()

    print("\n=== INFILL: honest clean-protocol TEST (greens pinned, blanks free-gen, no dict) ===", flush=True)
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(OUT if saved else BASE, b)
    mt = clean_metrics(b, TEST)
    print(f"  TEST clean win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) avg {mt['avg']:.2f}  "
          f"| losses: nonword {mt['nonword']}, ran-out {mt['exhausted']}", flush=True)
    print(f"  [ref] stage-1 plain clean 0.243 ; validity_max clean 0.281 ; constrained-mask(dict) 0.436", flush=True)
    print("\n[INFILL DONE]", flush=True)


if __name__ == "__main__":
    main()
