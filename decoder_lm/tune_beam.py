"""在 val/test 子集上扫 alpha/beta（CTC+LM 浅融合权重），找 CER 最低的组合。

为省时间，先把 ctc log_probs 提取到内存，再对多组参数做 beam search。
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
import os
import pickle
import sys
import time

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
LM_MAX_ID = 24883   # 只对常规 token 用 LM 打分，特殊 token(<|zh|>等) 跳过


def load_lm(path):
    d = pickle.load(open(path, "rb"))
    lm = NgramLM(order=d["order"], backoff=d["backoff"])
    lm.counts = d["counts"]
    lm._uni_total = sum(lm.counts[1].values())
    lm._uni_vocab = len(lm.counts[1])
    return lm


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=TEST_JSONL)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--lm", default=LM3_PKL)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--betas", default="0")
    ap.add_argument("--out", default=os.path.join(RESULTS, "tune_result.json"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print(f"样本 {len(rows)} 条")

    m = AutoModel(model=M, disable_update=True, device="cuda:0")
    net = m.model
    tok = m.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    # 提取 log_probs
    t0 = time.time()
    cache = []
    for r in rows:
        wav, _ = sf.read(r["audio"], dtype="float32")
        m.generate(input=wav, cache={}, language="zh", use_itn=False,
                   ban_emo_unk=False, batch_size_s=60, merge_vad=False)
        logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
        cache.append((logp, r["ref"]))
    print(f"提取 log_probs 完成 {time.time()-t0:.0f}s  平均 T={np.mean([c[0].shape[0] for c in cache]):.1f}")

    lm = load_lm(a.lm)
    alphas = [float(x) for x in a.alphas.split(",")]
    betas = [float(x) for x in a.betas.split(",")]

    results = []
    # alpha=0 作为 baseline（=greedy）
    for alpha in alphas:
        for beta in betas:
            t0 = time.time()
            E = Nt = 0
            for logp, ref in cache:
                ids, _ = ctc_prefix_beam_search(logp, BLANK, beam_size=a.beam, topk=a.topk,
                                                lm=(lm if alpha > 0 else None),
                                                alpha=alpha, beta=beta, lm_max_id=LM_MAX_ID)
                hyp = rich_transcription_postprocess(tok.tokens2text(ids))
                e, n = cer_err(ref, hyp)
                E += e
                Nt += n
            cer = E / Nt * 100
            results.append({"alpha": alpha, "beta": beta, "cer": cer})
            print(f"  alpha={alpha:<4} beta={beta:<4} CER={cer:6.2f}%   ({time.time()-t0:.0f}s)")

    base = min(r["cer"] for r in results if r["alpha"] == 0)
    best = min(results, key=lambda r: r["cer"])
    print()
    print(f"baseline(alpha=0) CER = {base:.2f}%")
    print(f"best: alpha={best['alpha']} beta={best['beta']}  CER={best['cer']:.2f}%  "
          f"(净收益 {base-best['cer']:+.2f}pt)")
    json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
