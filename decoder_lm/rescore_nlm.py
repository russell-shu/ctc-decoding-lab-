"""用领域神经 LM（GPT2）对 CTC beam search 的 n-best 做 rescoring。

  score(y) = log P_ctc(y) + alpha * log P_nlm(y) + beta * len(y)

log P_nlm 由领域 LM 前向得到（teacher forcing），跳过特殊 token。
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
import sys

import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from transformers import GPT2LMHeadModel

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
from ctc_beam import ctc_prefix_beam_search  # noqa: E402
from eval_qwen3_asr_cer import (             # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

M = MODEL_SV
LM_MAX_ID = 24883
NLM = NLM_DIR


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


@torch.no_grad()
def nlm_logprob(model, ids, dev):
    seq = [c for c in ids if c <= LM_MAX_ID]
    if len(seq) < 2:
        return -1e9
    x = torch.tensor([seq], device=dev)
    out = model(input_ids=x, labels=x)
    return -out.loss.item() * (len(seq) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--betas", default="0")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST_JSONL,
                                        encoding="utf-8") if l.strip()][: a.limit]
    m = AutoModel(model=M, disable_update=True, device="cuda:0")
    net = m.model
    tok = m.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    dev = "cuda:0"
    nlm = GPT2LMHeadModel.from_pretrained(NLM).to(dev).eval()
    print(f"样本 {len(rows)}, beam={a.beam}")

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
            items.append((ctc_sc, nlm_logprob(nlm, ids, dev), len(ids), txt))
        cache.append((r["ref"], items))

    alphas = [float(x) for x in a.alphas.split(",")]
    betas = [float(x) for x in a.betas.split(",")]
    best = None
    for alpha in alphas:
        for beta in betas:
            E = Nt = 0
            for ref, items in cache:
                b = max(items, key=lambda it: it[0] + alpha * it[1] + beta * it[2])
                e, n = cer_err(ref, b[3])
                E += e
                Nt += n
            cer = E / Nt * 100
            print(f"  alpha={alpha:<5} beta={beta:<4} CER={cer:6.2f}%")
            if best is None or cer < best[2]:
                best = (alpha, beta, cer)

    # oracle 上界
    E = Nt = 0
    for ref, items in cache:
        e_best = min(cer_err(ref, it[3])[0] for it in items)
        _, n = cer_err(ref, items[0][3])
        E += e_best
        Nt += n
    print(f"\n  oracle 上限 CER={E/Nt*100:.2f}%   best: alpha={best[0]} beta={best[1]} CER={best[2]:.2f}%")


if __name__ == "__main__":
    main()
