"""诊断：oracle 能改进但 LM rescoring 没救的句子，正确答案(hyp)和候选(ref)在语言上差多少？

决定性问题：
  - 若两者语言上等价（如"哎/喂"）-> 纯语言 LM 无望，需要声学
  - 若 ref 明显更合理    -> LM 太弱，换强 LM 有希望
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
from transformers import GPT2LMHeadModel

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
from ctc_beam import ctc_prefix_beam_search  # noqa: E402
from eval_qwen3_asr_cer import (             # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

M = MODEL_SV
LM_MAX_ID = 24883
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


@torch.no_grad()
def nlm_ppl(model, ids, dev):
    seq = [c for c in ids if c <= LM_MAX_ID]
    if len(seq) < 2:
        return 1e9
    x = torch.tensor([seq], device=dev)
    out = model(input_ids=x, labels=x)
    return float(np.exp(out.loss.item()))


rows = [json.loads(l) for l in open(TEST_JSONL,
                                    encoding="utf-8") if l.strip()][:LIMIT]
m = AutoModel(model=M, disable_update=True, device="cuda:0")
net = m.model
tok = m.kwargs["tokenizer"]
buf = {}
net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
BLANK = net.blank_id
dev = "cuda:0"
nlm = GPT2LMHeadModel.from_pretrained(NLM_DIR).to(dev).eval()

n_fixable = 0
better_lm = 0     # 正确答案的 LM PPL 更低（LM 本应能救）
worse_lm = 0      # 正确答案的 LM PPL 反而更高（LM 有害）
shown = 0
for r in rows:
    wav, _ = sf.read(r["audio"], dtype="float32")
    m.generate(input=wav, cache={}, language="zh", use_itn=False,
               ban_emo_unk=False, batch_size_s=60, merge_vad=False)
    logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
    cands = ctc_prefix_beam_search(logp, BLANK, beam_size=20, topk=12, topn=20)
    top1_err, _ = cer_err(r["ref"], rich_transcription_postprocess(tok.tokens2text(cands[0][0])))
    # 找 oracle 最优
    best = None
    for ids, _ in cands:
        txt = rich_transcription_postprocess(tok.tokens2text(ids))
        e, _ = cer_err(r["ref"], txt)
        if best is None or e < best[0]:
            best = (e, txt, ids)
    if top1_err > 0 and best[0] < top1_err:
        n_fixable += 1
        ids_top = cands[0][0]
        ppl_top = nlm_ppl(nlm, ids_top, dev)
        ppl_best = nlm_ppl(nlm, best[2], dev)
        if ppl_best < ppl_top:
            better_lm += 1
        else:
            worse_lm += 1
        if shown < 18:
            shown += 1
            txt_top = rich_transcription_postprocess(tok.tokens2text(ids_top))
            tag = "LM能救" if ppl_best < ppl_top else "LM有害"
            print(f"[{tag}] ref={r['ref']!r}")
            print(f"    top1(错): {txt_top!r}   PPL={ppl_top:.1f}")
            print(f"    oracle  : {best[1]!r}   PPL={ppl_best:.1f}")

print()
print(f"可被候选池修复的句子: {n_fixable}")
print(f"  正确答案 LM PPL 更低(可救): {better_lm}")
print(f"  正确答案 LM PPL 更高(有害): {worse_lm}")
