"""探测 SenseVoice 的 tokenizer 接口，确认 text -> token ids 的可靠路径。"""
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

from funasr import AutoModel

M = MODEL_SV

m = AutoModel(model=M, disable_update=True, device="cpu")
tok = m.kwargs["tokenizer"]
net = m.model

print("tokenizer type:", type(tok))
print("attrs:", [a for a in dir(tok) if not a.startswith("_")])
print()
print("blank_id:", getattr(net, "blank_id", None))
print("vocab_size:", getattr(net, "vocab_size", None))
print()

# 尝试各种 text -> ids 的入口
probe_texts = ["机主", "我要办宜享花", "你好"]
for t in probe_texts:
    print(f"--- {t!r}")
    for meth in ["text2tokens", "encode", "text2ids", "tokenize", "text2token"]:
        fn = getattr(tok, meth, None)
        if fn is None:
            continue
        try:
            out = fn(t)
            print(f"  {meth}({t!r}) = {out!r}")
        except Exception as e:
            print(f"  {meth} 失败: {type(e).__name__}: {e}")

# 反向验证：ids -> text
print()
for ids in [[1, 2, 3], [100, 200]]:
    try:
        print(f"  tokens2text({ids}) = {tok.tokens2text(ids)!r}")
    except Exception as e:
        print(f"  tokens2text 失败: {e}")

# 打印词表前若干项，观察特殊 token 与文本 token 的边界
print()
for attr in ["tokens", "vocab", "token_list", "id2token"]:
    v = getattr(tok, attr, None)
    if v is not None:
        print(f"{attr}: type={type(v)} len={len(v)}")
        if isinstance(v, (list, tuple)):
            for i in list(range(0, 5)) + list(range(24880, 24890)) + list(range(len(v) - 5, len(v))):
                if 0 <= i < len(v):
                    print(f"   [{i}] {v[i]!r}")
        break
