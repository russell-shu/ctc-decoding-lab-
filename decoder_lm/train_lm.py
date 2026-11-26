"""用训练文本训 token 级 n-gram LM（供 CTC beam search 浅融合）。"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import json
import pickle
import sys

sys.path.insert(0, DECODER_LM)
from ctc_beam import NgramLM  # noqa: E402
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer  # noqa: E402

ORDER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = os.path.join(DECODER_LM, f"lm{ORDER}.pkl")
BPE = BPE_MODEL

tok = SentencepiecesTokenizer(bpemodel=BPE, unk_symbol="<unk>", split_with_space=True)

seqs = []
for l in open(TRAIN_JSONL, encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    try:
        # tokens2ids 返回 list[list[int]]（BPE 可能把一个 token 拆成多个 subword），展平
        ids = [x for sub in tok.tokens2ids(tok.text2tokens(r["target"])) for x in sub]
    except Exception:
        continue
    if ids:
        seqs.append(ids)
n_tok = sum(len(s) for s in seqs)
print(f"序列 {len(seqs)} 条, token 总数 {n_tok}, 平均 {n_tok/len(seqs):.1f}")

lm = NgramLM(order=ORDER, backoff=0.4)
lm.train(seqs)
print(f"n-gram 计数: " + ", ".join(f"{n}gram={len(lm.counts[n])}" for n in range(1, ORDER + 1)))

with open(OUT, "wb") as f:
    pickle.dump({"order": ORDER, "backoff": 0.4, "counts": lm.counts}, f)
import os
print(f"已保存 {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
