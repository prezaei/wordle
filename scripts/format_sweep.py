"""Context-format bake-off: same validity-max recipe (CoT + aux + clean play), swappable board encoding.

Today's best format puts each guess letter 5 tokens from its color clue. These variants make the
letter<->clue binding local/explicit, to recover the model's TRACKING errors (re-using gray letters,
mis-placing greens) — the "stupid" subset of losses (some of the size-1 misses the diagnostic found).
Honest: identical recipe, train-only secrets, free-gen greedy clean eval on disjoint TEST. From-scratch
(no warm-start) so the format comparison is fair. FMT in {baseline, interleaved, keyboard}.

  baseline:    bos (GUESS l0..l4 c0..c4 SEP)*            -- letters then colors (current best)
  interleaved: bos (GUESS l0 c0 l1 c1 .. l4 c4 SEP)*     -- each letter adjacent to its clue
  keyboard:    baseline + SEP + (letter,status)* for every clued letter (alphabet-state suffix)

Env: FMT(baseline) VM_SECRETS(1852) VM_EPOCHS(16) VM_PRE(15) VM_TEACHER(2) VM_AUX(3.0) VM_LR(3e-4) VM_OUT.
"""

from __future__ import annotations

import os
import statistics
from random import Random

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.data import is_valid, load_valid_guesses, split
from wordle_slm.engine import Color, Game, Status
from wordle_slm.engine.constraints import is_consistent
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.sft.train import _batches, _valid_trie, load_checkpoint, save_checkpoint
from wordle_slm.teacher import generate_transcripts

DEV = "mps"
torch.manual_seed(int(os.environ.get("VM_SEED", "0")))
tok = Tokenizer()
THINK = tok.vocab_size
VOCAB = tok.vocab_size + 1
CFG = ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=2048, context_len=256, dropout=0.1)
_COLOR = {Color.GREEN: "<green>", Color.YELLOW: "<yellow>", Color.GRAY: "<gray>"}
_RANK = {Color.GRAY: 1, Color.YELLOW: 2, Color.GREEN: 3}
LETTER_IDS = [tok.token_to_id(c) for c in "abcdefghijklmnopqrstuvwxyz"]
LETTER_SET = set(LETTER_IDS)
LIDS = torch.tensor(LETTER_IDS, device=DEV)
LETTER_LO = tok.token_to_id("a")
ALLOWED_GEN = torch.tensor(LETTER_IDS + [THINK, tok.guess_id], device=DEV)
WORD_STARTS = {THINK, tok.guess_id}
VALID = load_valid_guesses()
TRIE = _valid_trie()
K_CANDS = int(os.environ.get("VM_KCANDS", "3"))  # CoT candidate-search width (more CoT = wider search)
FMT = os.environ.get("FMT", "baseline")
AUX_LAMBDA = float(os.environ.get("VM_AUX", "3.0"))


def _letters(w):
    return [tok.token_to_id(c) for c in w]


def _fb(turn):
    return [tok.gray_id] * 5 if turn.feedback is None else [tok.token_to_id(_COLOR[c]) for c in turn.feedback]


def _board_baseline(turns):
    ids = [tok.bos_id]
    for t in turns:
        ids += [tok.guess_id, *_letters(t.guess), *_fb(t), tok.sep_id]
    return ids


def _board_interleaved(turns):
    ids = [tok.bos_id]
    for t in turns:
        ids.append(tok.guess_id)
        L, F = _letters(t.guess), _fb(t)
        for i in range(5):
            ids += [L[i], F[i]]
        ids.append(tok.sep_id)
    return ids


def _board_keyboard(turns):
    ids = _board_baseline(turns)
    status: dict[str, Color] = {}
    for t in turns:
        if t.feedback is None:
            continue
        for i, c in enumerate(t.feedback):
            ltr = t.guess[i]
            if ltr not in status or _RANK[c] > _RANK[status[ltr]]:
                status[ltr] = c
    if status:
        ids.append(tok.sep_id)  # marker before the alphabet-state suffix
        for ltr in sorted(status):
            ids += [tok.token_to_id(ltr), tok.token_to_id(_COLOR[status[ltr]])]
    return ids


_BOARDS = {"baseline": _board_baseline, "interleaved": _board_interleaved, "keyboard": _board_keyboard}
board = _BOARDS[FMT]


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


def game_examples(game, rng):
    exs = []
    for k, turn in enumerate(game.turns):
        ids = board(game.turns[:k])
        mask = [False] * len(ids)
        for c in pick_cands(game.turns[:k], turn.guess, rng):
            ids.append(THINK)
            mask.append(True)
            ids += _letters(c)
            mask += [True] * 5
        ids.append(tok.guess_id)
        mask.append(True)
        ids += _letters(turn.guess)
        mask += [True] * 5
        ids.append(tok.eos_id)
        mask.append(False)
        exs.append((ids, mask))
    return exs


def cot_valid_rows(seq):
    """Per-example [len,26] trie-valid mask. Deterministic per example -> precompute ONCE (not per batch
    per epoch), so the GPU never stalls on this single-threaded Python trie-walk during training."""
    L = len(seq)
    v = torch.zeros((L, 26))
    for t, tokid in enumerate(seq):
        if tokid not in WORD_STARTS or t + 5 >= L:
            continue
        node = TRIE
        for j in range(5):
            for child in node:
                v[t + j, child] = 1.0
            nxt = seq[t + 1 + j] - LETTER_LO
            node = node.get(nxt, {}) if 0 <= nxt < 26 else {}
    return v


def build_freq_trie():
    """Trie over VALID words; each node stores '_f' = total English frequency of words under it (wordfreq).
    The aux can then push generation toward COMMON valid words, not just any valid word — general English
    knowledge baked in-weights (training-only; no inference lookup), the 'which common word' prior."""
    import wordfreq
    root = {}
    for w in VALID:
        f = wordfreq.word_frequency(w, "en") + 1e-9  # smooth so rare valid words still register
        node = root
        for ch in w:
            i = ord(ch) - 97
            child = node.get(i)
            if child is None:
                child = {}
                node[i] = child
            child["_f"] = child.get("_f", 0.0) + f
            node = child
    return root


def cot_freq_rows(seq, ftrie):
    """[len,26] per-position target = normalized subtree-FREQUENCY over valid next letters (vs uniform 0/1)."""
    L = len(seq)
    v = torch.zeros((L, 26))
    for t, tokid in enumerate(seq):
        if tokid not in WORD_STARTS or t + 5 >= L:
            continue
        node = ftrie
        for j in range(5):
            kids = {c: node[c]["_f"] for c in node if isinstance(c, int)}
            tot = sum(kids.values())
            if tot > 0:
                for c, f in kids.items():
                    v[t + j, c] = f / tot
            nxt = seq[t + 1 + j] - LETTER_LO
            node = node.get(nxt, {}) if 0 <= nxt < 26 else {}
    return v


@torch.no_grad()
def play(model, secret):
    g = Game(secret)
    visible = []
    while g.status is Status.ONGOING:
        seq = board(visible)
        guess, committed = [], False
        for _ in range(60):
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
        turn = g.guess(word if len(word) == 5 else "zzzzz")
        if turn.valid:
            visible.append(turn)
        else:
            break
    return g


@torch.no_grad()
def batched_play(model, secrets, chunk=384):
    """GPU-saturating eval: play ALL games in lockstep, one padded forward per token-step (right-pad +
    gather at each game's real last position -> causal mask never attends to pad, so identical to single
    play). ~10-50x faster than batch-1. Greedy + deterministic, so it must match play() exactly."""
    model.eval()
    games = [Game(s) for s in secrets]
    visible: list[list] = [[] for _ in secrets]
    alive = [True] * len(secrets)
    for _turn in range(6):
        idx = [i for i in range(len(games)) if alive[i] and games[i].status is Status.ONGOING]
        if not idx:
            break
        for cs in range(0, len(idx), chunk):
            sub = idx[cs : cs + chunk]
            seqs = [board(visible[i]) for i in sub]
            committed = [False] * len(sub)
            guess: list[list[int]] = [[] for _ in sub]
            tdone = [False] * len(sub)
            for _step in range(60):
                act = [j for j in range(len(sub)) if not tdone[j]]
                if not act:
                    break
                L = max(len(seqs[j]) for j in act)
                ids = torch.full((len(act), L), tok.pad_id, dtype=torch.long, device=DEV)
                last = torch.empty(len(act), dtype=torch.long, device=DEV)
                for a, j in enumerate(act):
                    ids[a, : len(seqs[j])] = torch.tensor(seqs[j], device=DEV)
                    last[a] = len(seqs[j]) - 1
                logits = model.forward(ids)[torch.arange(len(act), device=DEV), last]
                pick = ALLOWED_GEN[torch.argmax(logits[:, ALLOWED_GEN], dim=-1)].tolist()
                for a, j in enumerate(act):
                    t = int(pick[a])
                    seqs[j].append(t)
                    if committed[j]:
                        if t in LETTER_SET:
                            guess[j].append(t)
                        if len(guess[j]) >= 5:
                            tdone[j] = True
                    elif t == tok.guess_id:
                        committed[j] = True
            for a, i in enumerate(sub):
                gl = guess[a]
                word = "".join(tok.id_to_token(t) for t in gl[:5]) if len(gl) >= 5 else "zzzzz"
                turn = games[i].guess(word if len(word) == 5 else "zzzzz")
                if turn.valid:
                    visible[i].append(turn)
                else:
                    alive[i] = False
    return games


def evaluate(model, secrets):
    games = batched_play(model, list(secrets))
    wins = [g for g in games if g.won]
    n = sum(len(g.turns) for g in games)
    # clue-respect: of VALID guesses, fraction CONSISTENT with the clues so far (the format-tracking metric,
    # measurable even when win~0). Higher = the format helps the model track its own clues.
    vg = cg = 0
    for g in games:
        for k, t in enumerate(g.turns):
            if not is_valid(t.guess):
                continue
            vg += 1
            cg += is_consistent(t.guess, g.turns[:k])
    return {"win": len(wins) / len(games), "valid": sum(is_valid(t.guess) for g in games for t in g.turns) / n if n else 0.0,
            "respect": cg / vg if vg else 0.0,
            "avg": statistics.mean(g.guesses_used for g in wins) if wins else float("nan")}


def warmup(model, epochs):
    base = [tok.bos_id, tok.guess_id]
    exs = [(base + _letters(w) + [tok.eos_id], [False] * 2 + [True] * 5 + [False]) for w in VALID]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rng = Random(1)
    model.train()
    for epoch in range(epochs):
        for idx in _batches(len(exs), 256, rng):
            bs = [exs[i] for i in idx]
            L = max(len(s) for s, _ in bs)
            ids = torch.full((len(bs), L), tok.pad_id, dtype=torch.long)
            tmask = torch.zeros((len(bs), L))
            for i, (s, m) in enumerate(bs):
                ids[i, : len(s)] = torch.tensor(s)
                tmask[i, : len(m)] = torch.tensor([float(x) for x in m])
            ids, tmask = ids.to(DEV), tmask.to(DEV)
            logp = torch.log_softmax(model.forward(ids)[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            loss = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def main():
    train, held = split(seed=0)
    VAL, TEST = tuple(held[:96]), tuple(held[96:])
    safe = tuple(o for o in ("salet", "crane", "slate", "trace", "stare", "raise", "crate") if o not in set(held))
    cap = int(os.environ.get("VM_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    if os.environ.get("VM_EXPAND") == "1":  # DATA LEVER: more answer-LIKE (common) secrets, held-out EXCLUDED
        import wordfreq
        common = [w for w in wordfreq.top_n_list("en", 100000) if len(w) == 5 and w.isalpha() and w.isascii()]
        valid_set, held_set = set(VALID), set(held)
        secrets = tuple(w for w in common if w in valid_set and w not in held_set)[:cap]
        assert not (set(secrets) & held_set), "HELD-OUT LEAKED INTO SECRETS"  # honesty guard
        print(f"[fmt={FMT}] EXPAND: {len(secrets)} common+valid secrets (held-out {len(held)} excluded); "
              f"train-answers in set: {len(set(secrets) & set(train))}", flush=True)
    OUT = os.environ.get("VM_OUT", f"runs/fmt_{FMT}.pt")
    EPOCHS = int(os.environ.get("VM_EPOCHS", "16"))
    LR = float(os.environ.get("VM_LR", "3e-4"))
    PRE = int(os.environ.get("VM_PRE", "15"))
    TEACHER = int(os.environ.get("VM_TEACHER", "2"))
    print(f"[fmt={FMT}] VOCAB={VOCAB} aux={AUX_LAMBDA} epochs={EPOCHS} pre={PRE} |secrets|={len(secrets)}", flush=True)

    model = WordleGenerator(CFG, VOCAB).to(DEV)
    BASE = os.environ.get("VM_BASE", "")
    if BASE:  # warm-start from a strong model (e.g. validity_max_v4) -> reaches the 0.34 regime; adapt to FMT
        load_checkpoint(BASE, model)
        b0 = evaluate(model, VAL)
        print(f"[fmt={FMT}] warm-start from {BASE}: VAL win {b0['win']:.3f} valid {b0['valid']:.3f} (pre-adapt)", flush=True)
    else:
        print(f"[fmt={FMT}] spell warm-up ({PRE} ep)", flush=True)
        warmup(model, PRE)

    rng = Random(0)
    import pickle as _pkl
    games_pkl = os.environ.get("VM_GAMES_PKL", "")
    if games_pkl and os.path.exists(games_pkl):  # load pre-generated games (parallel CPU pre-step keeps GPU fed)
        with open(games_pkl, "rb") as f:
            games = _pkl.load(f)
        print(f"[fmt={FMT}] loaded {len(games)} teacher games from {games_pkl}", flush=True)
    else:
        games = []
        for s in range(TEACHER):
            games += [tr.game for tr in generate_transcripts(secrets, weak_frac=0.5, openers=safe, seed=300 + s, valid_pool=VALID, answer_pool=secrets)]
    exs = [e for g in games for e in game_examples(g, rng)]
    rng.shuffle(exs)
    # PRE-TENSORIZE everything ONCE into big GPU tensors (pad to global Lmax). Per-batch is then a pure
    # vectorized GPU gather — NO per-batch Python loop (that single-threaded loop was starving the GPU).
    N = len(exs)
    Lmax = max(len(s) for s, _ in exs)
    FREQAUX = os.environ.get("VM_FREQAUX") == "1"  # aux pushes toward COMMON valid words (vs any valid word)
    ftrie = build_freq_trie() if FREQAUX else None
    ids_all = torch.full((N, Lmax), tok.pad_id, dtype=torch.long)
    tmask_all = torch.zeros((N, Lmax))
    vmask_all = torch.zeros((N, Lmax, 26))
    for i, (s, m) in enumerate(exs):
        ids_all[i, : len(s)] = torch.tensor(s)
        tmask_all[i, : len(m)] = torch.tensor([float(x) for x in m])
        r = cot_freq_rows(s, ftrie) if FREQAUX else cot_valid_rows(s)
        vmask_all[i, : r.shape[0]] = r
    ids_all, tmask_all, vmask_all = ids_all.to(DEV), tmask_all.to(DEV), vmask_all.to(DEV)
    print(f"[fmt={FMT}] games={len(games)} examples={N} pre-tensorized (Lmax={Lmax}) freqaux={FREQAUX} on {DEV}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    rng2 = Random(0)
    best, saved = -1.0, False
    model.train()
    print(f"[fmt={FMT}] epoch  clean-VAL win / valid / clue-RESPECT (the format metric)", flush=True)
    for epoch in range(EPOCHS):
        for idx in _batches(N, int(os.environ.get("VM_BATCH", "512")), rng2):  # 128GB -> big batches saturate the GPU
            bidx = torch.tensor(idx, device=DEV)
            ids, tmask, vmask = ids_all[bidx], tmask_all[bidx], vmask_all[bidx]  # vectorized GPU gather
            logits = model.forward(ids)
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            imit = (nll * tmask[:, 1:]).sum() / tmask[:, 1:].sum()
            logp_let = torch.log_softmax(logits[:, :-1][:, :, LIDS], dim=-1)
            vm = vmask[:, :-1]
            aux_pos = (vm.sum(-1) > 0).float() * tmask[:, 1:]
            if FREQAUX:  # CE toward the freq-weighted-valid distribution -> bias generation to COMMON words
                aux = (-(vm * logp_let).sum(-1) * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            else:  # maximize probability mass on ANY valid next-letter (uniform)
                valid_mass = (logp_let.exp() * vm).sum(-1).clamp_min(1e-9)
                aux = (-valid_mass.log() * aux_pos).sum() / aux_pos.sum().clamp_min(1.0)
            loss = imit + AUX_LAMBDA * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if epoch % 4 == 0 or epoch == EPOCHS - 1:
            m = evaluate(model, VAL)
            flag = ""
            if m["respect"] > best:  # select by clue-RESPECT (win stays 0 from-scratch -> would save epoch-0 garbage)
                best, saved = m["respect"], True
                save_checkpoint(OUT, model, opt, epoch, SFTConfig())
                flag = "  <- best-respect, saved"
            print(f"  epoch {epoch:>2}  win {m['win']:.3f} / valid {m['valid']:.3f} / respect {m['respect']:.3f}{flag}", flush=True)
            model.train()

    print(f"\n=== fmt={FMT}: honest clean TEST ===", flush=True)
    b = WordleGenerator(CFG, VOCAB).to(DEV)
    load_checkpoint(OUT, b)
    mt = evaluate(b, TEST)
    print(f"  TEST win {mt['win']:.3f} ({int(round(mt['win'] * len(TEST)))}/{len(TEST)}) valid {mt['valid']:.3f} respect {mt['respect']:.3f} avg {mt['avg']:.2f}", flush=True)
    print(f"  [bar] validity-max v4 (warm-started) 0.338 ; compare fmt deltas from-scratch", flush=True)
    print(f"\n[FMT {FMT} DONE]", flush=True)


if __name__ == "__main__":
    main()
