"""冒烟测试：CTC prefix beam search（无 LM）vs greedy。"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import os
import sys
import json
import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
from ctc_beam import ctc_prefix_beam_search          # noqa: E402
from eval_qwen3_asr_cer import (                     # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

M = MODEL_SV
m = AutoModel(model=M, disable_update=True, device="cuda:0")
net = m.model
tok = m.kwargs["tokenizer"]
buf = {}
net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
BLANK = net.blank_id
V = net.vocab_size
print(f"blank_id={BLANK}  vocab={V}")

rows = [json.loads(l) for l in open(TEST_JSONL,
                                    encoding="utf-8") if l.strip()][:20]


def cer(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


tot_g = tot_b = tot_n = 0
for r in rows:
    wav, sr = sf.read(r["audio"], dtype="float32")
    res = m.generate(input=wav, cache={}, language="zh", use_itn=False,
                     ban_emo_unk=False, batch_size_s=60, merge_vad=False)
    greedy = rich_transcription_postprocess(res[0]["text"])

    logits = buf["x"][0]                              # [T,V]
    logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
    ids, sc = ctc_prefix_beam_search(logp, blank_id=BLANK, beam_size=10, topk=12)
    beam = rich_transcription_postprocess(tok.tokens2text(ids))

    eg, n = cer(r["ref"], greedy)
    eb, _ = cer(r["ref"], beam)
    tot_g += eg
    tot_b += eb
    tot_n += n
    mark = "" if eg == eb else ("   beam better" if eb < eg else "   beam worse")
    print(f"[T={logp.shape[0]:3d}] ref={r['ref']!r}")
    print(f"        greedy={greedy!r}")
    print(f"        beam  ={beam!r}{mark}")

print()
print(f"greedy CER = {tot_g/tot_n*100:.2f}%   beam(no LM) CER = {tot_b/tot_n*100:.2f}%   N={tot_n}")
