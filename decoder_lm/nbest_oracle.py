"""n-best oracle 实验：beam search 的候选里到底有没有正确答案？

这给出"CTC beam search + 任何 LM"的理论上限：
  - 若 ref 在 top-N 里出现率高 -> beam 里有答案，只是排序问题，强 LM 有潜力
  - 若出现率低 -> beam 里根本没有正确答案（CTC 声学概率就没给对），LM 再强也没用
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import json
import sys

import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
from ctc_beam import ctc_prefix_beam_search  # noqa: E402
from eval_qwen3_asr_cer import (             # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

M = MODEL_SV
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BEAM = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


rows = [json.loads(l) for l in open(TEST_JSONL,
                                    encoding="utf-8") if l.strip()][:LIMIT]
m = AutoModel(model=M, disable_update=True, device="cuda:0")
net = m.model
tok = m.kwargs["tokenizer"]
buf = {}
net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
BLANK = net.blank_id

KS = [1, 3, 5, 10, BEAM]
hit = {k: 0 for k in KS}
oracle_err = 0
base_err = 0
Ntot = 0
n_cand = 0
for r in rows:
    wav, _ = sf.read(r["audio"], dtype="float32")
    m.generate(input=wav, cache={}, language="zh", use_itn=False,
               ban_emo_unk=False, batch_size_s=60, merge_vad=False)
    logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
    cands = ctc_prefix_beam_search(logp, BLANK, beam_size=BEAM, topk=12, topn=BEAM)
    texts = [rich_transcription_postprocess(tok.tokens2text(ids)) for ids, _ in cands]
    n_cand += len(texts)
    ref_n = N(r["ref"])
    texts_n = [N(x) for x in texts]
    for k in KS:
        if ref_n in texts_n[:k]:
            hit[k] += 1
    e_base, n = cer_err(r["ref"], texts[0])
    e_best = min(cer_err(r["ref"], t)[0] for t in texts)
    base_err += e_base
    oracle_err += e_best
    Ntot += n

print(f"样本 {len(rows)} 条, 每条候选 {n_cand/len(rows):.1f} 个")
print()
print("=== ref 出现在 top-N 候选中的比例（句级完全匹配，去标点归一化后）===")
for k in KS:
    print(f"  top-{k:2d}: {hit[k]:4d}/{len(rows)} = {hit[k]/len(rows)*100:5.1f}%")
print()
print(f"baseline(beam best)  CER = {base_err/Ntot*100:.2f}%")
print(f"oracle(候选里挑最优) CER = {oracle_err/Ntot*100:.2f}%   "
      f"-> 完美 rescoring 的上限 {base_err/Ntot*100 - oracle_err/Ntot*100:+.2f}pt")
