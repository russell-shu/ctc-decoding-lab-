"""逐条诊断：CTC+LM 浅融合（alpha>0）相对纯 CTC（alpha=0）到底改了什么。

看净收益是真是假：统计改动条数、改对/改错，并列出样例。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=TEST_JSONL)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--lm", default=LM3_PKL)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--topk", type=int, default=12)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    m = AutoModel(model=M, disable_update=True, device="cuda:0")
    net = m.model
    tok = m.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    cache = []
    for r in rows:
        wav, _ = sf.read(r["audio"], dtype="float32")
        m.generate(input=wav, cache={}, language="zh", use_itn=False,
                   ban_emo_unk=False, batch_size_s=60, merge_vad=False)
        logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
        cache.append((logp, r["ref"]))

    d = pickle.load(open(a.lm, "rb"))
    lm = NgramLM(order=d["order"], backoff=d["backoff"])
    lm.counts = d["counts"]
    lm._uni_total = sum(lm.counts[1].values())
    lm._uni_vocab = len(lm.counts[1])

    def run(alpha):
        out = []
        for logp, ref in cache:
            ids, _ = ctc_prefix_beam_search(logp, BLANK, beam_size=a.beam, topk=a.topk,
                                            lm=(lm if alpha > 0 else None),
                                            alpha=alpha, lm_max_id=LM_MAX_ID)
            out.append(rich_transcription_postprocess(tok.tokens2text(ids)))
        return out

    h0 = run(0.0)
    h1 = run(a.alpha)

    E0 = N0 = E1 = N1 = 0
    changed = good = bad = same = 0
    examples = []
    for (logp, ref), x0, x1 in zip(cache, h0, h1):
        e0, n = cer_err(ref, x0)
        e1, _ = cer_err(ref, x1)
        E0 += e0; N0 += n
        E1 += e1; N1 += n
        if N(x0) != N(x1):
            changed += 1
            if e1 < e0: good += 1
            elif e1 > e0: bad += 1
            else: same += 1
            if len(examples) < 25:
                examples.append((good, ref, x0, x1, e0, e1))

    print(f"样本 {len(cache)} 条")
    print(f"  alpha=0    CER={E0/N0*100:.2f}%")
    print(f"  alpha={a.alpha}  CER={E1/N1*100:.2f}%   (净收益 {(E0-E1)/N0*100:+.2f}pt)")
    print(f"  改动 {changed} 条: 改对 {good} / 改错 {bad} / 无变化 {same}")
    print()
    print("改动样例:")
    for _, ref, x0, x1, e0, e1 in examples:
        mark = "✓" if e1 < e0 else ("✗" if e1 > e0 else "=")
        print(f"  {mark} ref={ref!r}")
        print(f"      a=0  : {x0!r}")
        print(f"      a={a.alpha}: {x1!r}")


if __name__ == "__main__":
    main()
