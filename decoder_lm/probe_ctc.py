"""探测 SenseVoice 的 CTC 帧级概率与词表，为 CTC prefix beam search 做准备。"""
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

M = MODEL_SV
WAV = TEST_WAV

m = AutoModel(model=M, disable_update=True, device="cuda:0")
net = m.model
print("net type:", type(net))
print()
print("=== 顶层属性 ===")
print([a for a in dir(net) if not a.startswith("_")])

ctc = getattr(net, "ctc", None)
print()
print("=== ctc 模块 ===")
print("type:", type(ctc))
if ctc is not None:
    for a in ["ctc_lo", "blank_id", "odim", "vocab_size"]:
        print(f"  {a}: {getattr(ctc, a, '(无)') if a!='ctc_lo' else type(getattr(ctc,a,None))}")
    lo = getattr(ctc, "ctc_lo", None)
    if lo is not None:
        print("  ctc_lo weight shape:", tuple(lo.weight.shape))
print()
print("=== net 的 blank_id / vocab ===")
for a in ["blank_id", "vocab_size", "odim"]:
    print(f"  {a}: {getattr(net, a, '(无)')}")

# hook ctc_lo 抓 raw logits
buf = {}
h = ctc.ctc_lo.register_forward_hook(lambda mod, inp, out: buf.__setitem__("x", out.detach()))
# hook encoder 拿 frame 数
enc_buf = {}
if hasattr(net, "encoder"):
    net.encoder.register_forward_hook(lambda mod, inp, out: enc_buf.__setitem__("o", out[0].detach() if isinstance(out, tuple) else out.detach()))

# 挑一条短音频
cands = sorted(os.listdir(WAV))[:1]
p = os.path.join(WAV, cands[0])
wav, sr = sf.read(p, dtype="float32")
print()
print(f"=== 测试音频 {cands[0]}  sr={sr} dur={len(wav)/sr:.2f}s ===")

res = m.generate(input=wav, cache={}, language="zh", use_itn=False, ban_emo_unk=False,
                 batch_size_s=60, merge_vad=False)
print("官方输出:", res[0]["text"])

raw = buf.get("x")
if raw is None:
    print("!! 没抓到 ctc logits"); sys.exit(1)
print()
print("ctc_lo 输出 shape:", tuple(raw.shape), " dtype:", raw.dtype)
logp = torch.log_softmax(raw.float(), dim=-1)[0]  # [T, V]
T, V = logp.shape
print(f"T={T} 帧, V={V} 词表")
print("每帧时长(假设 subsample 4x): %.3f s" % (len(wav)/sr/T))

# 手工 greedy
yseq = logp.argmax(dim=-1)
uniq = torch.unique_consecutive(yseq)
blank = getattr(net, "blank_id", 0)
tok = uniq[uniq != blank].tolist()
print()
print(f"手工 greedy token 数: {len(tok)}  (blank_id={blank})")

# 尝试用 tokenizer 解码
tk = getattr(net, "tokenizer", None)
print("tokenizer:", type(tk))
if tk is not None and hasattr(tk, "tokens2text"):
    try:
        text = tk.tokens2text(tok)
        print("手工 greedy 文本:", repr(text))
    except Exception as e:
        print("tokens2text 失败:", e)
print()
print("=== 前 20 帧的 argmax 分布 ===")
for t in range(min(20, T)):
    v = yseq[t].item()
    print(f"  frame {t:3d}: argmax={v}  logp={logp[t,v].item():.3f}  (blank logp={logp[t,blank].item():.3f})")
