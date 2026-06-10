"""CPU-only parallel teacher-game generation (NO torch / NO MPS) so the GPU stays fed.

The InfoMax/Consistent teacher is single-threaded and CPU-heavy (consistency scans), so generating games
for thousands of secrets serially leaves the GPU idle for many minutes. This module is imported by worker
processes (spawn) to generate transcripts for a chunk of secrets in parallel across all cores. It must NOT
import torch (workers would each init MPS) — only the engine/teacher.
"""

from __future__ import annotations

from wordle_slm.teacher import generate_transcripts


def gen_chunk(args):
    """One worker: generate games for a chunk of secrets. args is picklable (strings/ints/lists)."""
    chunk, seed, openers, valid, answer, weak_frac = args
    return [tr.game for tr in generate_transcripts(
        tuple(chunk), weak_frac=weak_frac, openers=tuple(openers), seed=seed,
        valid_pool=tuple(valid), answer_pool=tuple(answer))]
