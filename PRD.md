# PRD: Wordle SLM — Learning the *Strategy* by Trial and Error

**Status:** Draft v0.2 (hardened via adversarial review)
**Owner:** Pedram
**Last updated:** 2026-06-02

---

## 1. One-liner

Build a small language model **from scratch** and teach it — through trial and
error (reinforcement learning) — to play Wordle by **learning the strategy**, not
by memorizing answers. The point is the journey: a hands-on way to understand how
small models and RL actually work, with a fun, bounded game as the proof.

## 2. Guiding principle: strategy, not memorization

This is the north star, and it shapes every decision below.

The goal is a model that learns the **method** of Wordle — use the green / yellow
/ gray feedback to narrow down the possibilities — **not** one that has memorized
which words are answers. A model could "win" by memorizing the ~2,300-word answer
list and learn nothing interesting. That is a failure, not a success.

**How we keep ourselves honest:** the model is always judged on **words it never
practiced on** (a held-out set). If it plays nearly as well on unseen words as on
practiced ones, it learned the strategy. If it only does well on words it trained
on, it memorized — and we fix that. This single test runs through the whole plan.

## 3. Why this project

The real goal is **learning**, not winning Wordle. Wordle is just an unusually
good teacher:

- **Short games.** Every game ends in at most 6 guesses — fast practice rounds.
- **Crystal-clear feedback.** After each guess you get colors. The model is told,
  immediately, how it's doing — exactly the signal RL needs.
- **A real strategy to discover.** Good play is about *using* the clues, so there
  is a genuine method to learn (not just facts to store).
- **Small, fixed world.** Known lists of valid 5-letter words. No ambiguity.
- **Runs on a laptop.** Small enough to train and iterate on this Mac, with
  feedback in minutes, not days.

"From scratch" is deliberate: building the whole thing end-to-end — including
training the model ourselves — is where the learning is.

## 4. Goals

1. **Understand SLMs end-to-end** — build and train a small model myself.
2. **Understand RL end-to-end** — design rewards, run training loops, and watch a
   model improve from experience.
3. **Produce a model that learns the *strategy*** — it plays well on **unseen
   words**, with a visible "it's getting better" curve. (See §2.)
4. **Keep it all runnable on this Mac** — no cloud dependency.
5. **Build transferable intuition** — lessons I can carry to bigger models and
   harder problems.

## 5. Non-goals (this round)

- **Not** a model that wins by memorizing answers (see §2 — that's an anti-goal).
- **Not** a polished app or UI.
- **Not** a general chatbot — it only needs to play Wordle.
- **Not** multi-machine or cloud training.

## 6. What success looks like

The bar is: **the model clearly learns the strategy, and I can explain why it got
better.** "Learns the strategy" is measured on **held-out words it never trained
on.**

| Signal | What we want to see |
| --- | --- |
| **Win rate on unseen words** | % of *held-out* games solved within 6 guesses — far above the random floor and rising over training. **This is the headline metric.** |
| **Generalization gap** | Win rate on unseen words is *close to* win rate on practiced words. A big gap = memorization, and is a fail. |
| **Learning curve** | Win rate goes *up* over training — visible improvement, not noise. |
| **Guess efficiency** | Average guesses-per-win trends *down* over time. |
| **Legal play** | Model mostly produces real 5-letter words, not gibberish. |

**The three reference points (provisional numbers, calibrated in Phase 0):**

- **Random floor ≈ 0.26%.** A guesser that ignores feedback essentially never
  wins (6 random picks from ~2,300 words). Beating this is almost free, so it is a
  *sanity check, not the success bar.*
- **Feedback-using yardstick ≈ 96%.** A guesser that just picks any word still
  consistent with the clues wins ~96–99% of games. This is the *honest, hard*
  yardstick — our stretch reference, not the pass/fail gate.
- **Success target: ≥ 80% win rate on unseen words**, with a small generalization
  gap and a rising curve. This is the goal we're aiming for; Phase 0 will tell us
  how hard a reach 80% is for a model this small. The **method** of measuring
  (held-out words, the gap, the curve) is fixed regardless.

We declare this round a win when the model **comfortably beats the random floor on
unseen words, with a small generalization gap and an explainable learning curve.**

## 7. Who it's for

- **Primary user:** me — the builder and learner. Audience of one.
- **Secondary:** anyone who later reads the write-up of what I built and learned.

## 8. What we're building (in plain terms)

Five pieces:

1. **The game** — a Wordle engine the model plays against automatically, thousands
   of times, so it can practice.
2. **The player** — the small model. Given the game so far (past guesses + colors),
   it picks the next 5-letter word.
3. **The head start** — before the coach (step 4) turns on, we let the model first
   *watch and copy* a few thousand example games played by a simple, sensible
   strategy ("open with a common word, then only guess words that fit the clues so
   far"). **Why this matters:** a from-scratch model starts out guessing gibberish
   and would essentially *never* win by luck — so a reward-based coach would have
   nothing to reward and would never get going. Copying example games first makes
   the model good enough to *sometimes* win, which finally gives the coach
   something to work with. (The technical name is an "imitation warm-up." The
   example games come from *our own* engine — not a big outside model — and they
   teach the *method* of using clues, not which words are the answers. So it stays
   "from scratch" and supports learning the strategy, not memorizing it.)
4. **The coach** — the RL part. It rewards good play and nudges the player to
   repeat what worked. Crucially, it rewards **partial progress** (getting letters
   right, narrowing the field) — not just full wins — so the model gets useful
   signal early instead of almost never.
5. **The scoreboard** — telemetry so I can *watch* it learn (win rate on unseen
   vs. practiced words, the gap, guesses, legal-word rate, all over time).

### How a game flows
1. The game picks a secret word.
2. The player guesses a word.
3. The game returns colors: **green** = right letter, right spot; **yellow** =
   right letter, wrong spot; **gray** = not in the word.
4. The player guesses again using that info — up to 6 times.
5. Win if it guesses the word; lose otherwise.
6. The coach scores the game (full + partial progress) and updates the player so
   next time it's a little better.

## 9. The approach (committed plan + the genuinely open experiments)

We commit the *shape* of the plan up front, because the prior art is clear about
what goes wrong otherwise. We deliberately leave the *interesting unknowns* open —
discovering those is the learning.

**Committed (because cold trial-and-error on a fresh model usually learns nothing):**
- **Head start first, then RL.** Let the model learn the basics by copying example
  games *before* turning on the reward-based coach — otherwise a from-scratch model
  almost never wins, so there's nothing to reward and it never learns (see §8,
  "The head start").
- **Reward partial progress, not just wins.** Rewarding only full wins starves a
  fresh model of signal; we also reward getting letters right and narrowing the
  field, so it gets useful feedback early.
- **Start small, then widen.** Begin practicing on a small set of words and expand
  — easier to get traction, then generalize.
- **Always test on unseen words.** The held-out split (see §2) is non-negotiable.

**Open experiments (this is where the learning happens — see §13):**
- How small the model can be and still learn.
- The exact recipe for rewarding partial progress.
- How fast to widen the word set.

## 10. Milestones

Each phase **teaches something**, **produces a visible result**, and has a
**concrete "done" line.**

- **Phase 0 — The playground.** Build the Wordle engine, a random player, the
  scoreboard, and the held-out split.
  *Done when:* the engine plays full games correctly on a set of known examples;
  the random floor is measured over 1,000 games and recorded; and we've measured
  how many practice games/second the Mac can run, confirming a full training cycle
  fits the **~1-hour budget** (or identified what to shrink so it does).
  *Learn: the environment, the feedback signal, and our speed budget.*

- **Phase 1 — The model can play (legally and sensibly).** Build the small model
  and give it the head start (imitation).
  *Done when:* the model outputs a valid 5-letter word ≥95% of the time over 1,000
  tries, and mostly respects obvious clues.
  *Learn: building, training, and running an SLM.*

- **Phase 2 — Teach it with reward (the RL core).** Turn on the coach (with partial
  progress), start on the small word set, widen over time.
  *Done when:* on **unseen** words, win rate is well above the random floor and
  rising over training, with a small practiced-vs-unseen gap.
  *Learn: how RL actually works — rewards, training loop, real improvement.*

- **Phase 3 — Make it better, and understand why.** Tune the reward, inspect
  failures, watch for memorization and reward loopholes.
  *Done when:* at least one change is shown — with before/after numbers — to
  improve win rate on unseen words or average guesses.
  *Learn: debugging RL and building intuition.*

- **Phase 4 (optional) — Show and tell.** A script that plays a live game in the
  terminal, plus a polished public write-up.
  *Done when:* it plays one full live game and a public write-up exists.

## 11. Decisions we're committing now

(Pulled out of "open questions" because they're load-bearing — they decide the
baseline and whether the project works at all. Plain-language defaults; revisable.)

- **Word list:** start with the curated answer list (~2,300 words), split into a
  practice set and a held-out test set; widen to the larger valid-guess list later
  if useful.
- **Approach:** head start (imitation) first, then RL with partial-progress
  rewards — see §9.
- **Framework:** **PyTorch** (using its Metal/MPS backend to run on this Mac's
  GPU). Chosen for its large ecosystem and abundance of tutorials, and because the
  skills transfer beyond Apple hardware. (Apple's own MLX is the fallback if we hit
  a building block PyTorch's Mac support is missing.)

## 12. Constraints & assumptions

- **Hardware:** Runs entirely on this MacBook Pro (Apple M5 Max, 40-core GPU,
  128 GB unified memory). We expect memory to be ample for a deliberately small
  model; the **real limits are iteration speed and my own learning pace** — which
  is why Phase 0 measures the speed budget explicitly.
- **Time budget:** One full training cycle — the head start plus the RL run —
  should finish in about **an hour** on this Mac. This is the loop we iterate on,
  so it **bounds the model size and how many practice games we run**: if a run
  blows past an hour, we shrink the model or the word set rather than wait. (An
  hour is generous for a deliberately small model on this chip.)
- **People:** Single developer, part-time, learning pace.
- **Rules & data:** Standard Wordle rules and well-known public 5-letter word
  lists. Practice data (example games for the head start) is generated by **our own**
  engine — not copied from a big outside model — to keep it honestly "from scratch."
- **Offline:** No external services required.

## 13. Risks & how we de-risk

| Risk | Mitigation |
| --- | --- |
| **Cold trial-and-error never gets off the ground** (a fresh model almost never wins, so there's no signal to learn from) | The committed plan (§9): head start by imitation, reward partial progress, start on a small word set. Validate the training loop on a trivial task first. |
| **It memorizes answers instead of learning strategy** | Always measure on **unseen** words and watch the practiced-vs-unseen gap (§2, §6). A big gap triggers a fix (more variety, held-out enforcement). |
| **The coach gets gamed** (model finds a reward loophole) | Inspect actual games regularly; adjust the reward when it's being exploited. |
| **Apple-Silicon GPU rough edges** | Use PyTorch's Metal (MPS) backend, which is mature and widely used; fall back to CPU for any unsupported operation, or to Apple's MLX if needed. Validate the training loop early. |
| **Reward design is subtle** | Start simple (win/lose + partial progress), change one thing at a time, measure each change. |
| **Scope creep toward a "perfect solver"** | Explicitly out of scope this round (§5). |
| **Losing motivation on slow progress** | Milestone structure guarantees a visible win at each phase; Phase 0's speed budget keeps runs tolerable. |

## 14. Open questions (the genuine unknowns — discovering these is the point)

- How small can "small" be and still learn? (Pick in Phase 1.)
- The exact recipe for rewarding partial progress. (Discover in Phase 2–3.)
- How fast to widen the word set.
- Do we bother with "hard mode" rules (must reuse revealed clues)? Probably later.

## 15. Learning deliverables (required — the journey is the point)

Because learning is the **primary** goal, the learning itself is a required output,
not an afterthought:

- **A running lab notebook**, updated each phase: the decisions made, each
  experiment as *hypothesis → result*, and what actually moved the needle.
- **A short "what I learned about RL" note** from Phase 3.

(The Phase 4 demo/write-up is still optional — but the lab notebook is not.)

## 16. Future ideas (out of scope now)

- Push toward near-optimal play (≈3.42 guesses).
- A web or app UI.
- Generalize to other word games or longer words.
- Compare different model sizes or RL methods head-to-head.

---

### Notes for the next revision
- Phase 0 **validates** the reference numbers in §6 (floor, yardstick) against real
  data and gauges how reachable the ≥80% target is for a model this small.
- Decide on Phase 4: write-up vs. live demo vs. both.
