"""用 LM 对 CTC beam search 的 n-best 做 rescoring（两阶段，避免 beam 剪枝丢候选）。

  score(y) = log P_ctc(y) + alpha * log P_lm(y) + beta * len(y)

先固定 n-best 候选池，再用 LM 重排。这比 shallow fusion 更能发挥 LM 的作用，
因为候选池里已经包含大量正确答案（top-20 命中率 93%）。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import argparse
import json
import math
import pickle
import sys

import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
from ctc_beam import NgramLM, ctc_prefix_beam_search  # noqa: E402
from eval_qwen3_asr_cer import (                     # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

M = MODEL_SV
LM_MAX_ID = 24883


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


def lm_logprob(lm, ids):
    s = 0.0
    for i, c in enumerate(ids):
        if c > LM_MAX_ID:
            continue
        s += lm.score(tuple(ids[:i]), c)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--lm", default=LM3_PKL)
    ap.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.4")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST_JSONL,
                                        encoding="utf-8") if l.strip()][: a.limit]
    m = AutoModel(model=M, disable_update=True, device="cuda:0")
    net = m.model
    tok = m.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    d = pickle.load(open(a.lm, "rb"))
    lm = NgramLM(order=d["order"], backoff=d["backoff"])
    lm.counts = d["counts"]
    lm._uni_total = sum(lm.counts[1].values())
    lm._uni_vocab = len(lm.counts[1])

    # 生成 n-best 并缓存（含 LM logprob）
    cache = []
    for r in rows:
        wav, _ = sf.read(r["audio"], dtype="float32")
        m.generate(input=wav, cache={}, language="zh", use_itn=False,
                   ban_emo_unk=False, batch_size_s=60, merge_vad=False)
        logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
        cands = ctc_prefix_beam_search(logp, BLANK, beam_size=a.beam, topk=a.topk, topn=a.beam)
        items = []
        for ids, ctc_sc in cands:
            txt = rich_transcription_postprocess(tok.tokens2text(ids))
            items.append((ctc_sc, lm_logprob(lm, ids), len(ids), txt))
        cache.append((r["ref"], items))

    def ev(alpha, beta=0.0):
        E = Nt = 0
        for ref, items in cache:
            best = max(items, key=lambda it: it[0] + alpha * it[1] + beta * it[2])
            e, n = cer_err(ref, best[3])
            E += e
            Nt += n
        return E / Nt * 100

    print(f"样本 {len(rows)} 条, beam={a.beam}")
    for alpha in [float(x) for x in a.alphas.split(",")]:
        print(f"  alpha={alpha:<5} CER={ev(alpha):6.2f}%")

    # oracle 上界对照
    E = Nt = 0
    for ref, items in cache:
        e, n = min((cer_err(ref, it[3])[0], it[3]) for it in items)[0], None
        E += e
        Nt += n
    print(f"\n  oracle 上限 CER={E/Nt*100:.2f}%")


if __name__ == "__main__":
    main()
