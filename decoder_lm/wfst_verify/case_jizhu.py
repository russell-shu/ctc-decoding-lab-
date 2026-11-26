"""针对含"机主"的样本做逐条对比，并统计 baseline 输出的 OOV 情况。"""
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

import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from verify_wfst import build_lexicon, ctc_beam, cer_err  # noqa: E402

M = MODEL_SV
TEST = TEST_JSONL

rows = [json.loads(l) for l in open(TEST, encoding="utf-8") if l.strip()]
target = [r for r in rows if "机主" in r["ref"]]

model = AutoModel(model=M, disable_update=True, device="cuda:0")
net = model.model
tok = model.kwargs["tokenizer"]
buf = {}
net.ctc.ctc_lo.register_forward_hook(lambda m, i, o: buf.__setitem__("x", o.detach()))
BLANK = net.blank_id

trie, words, singles = build_lexicon(tok)
print("\n词表检查:")
for w in ["机主", "机础", "基础", "助理", "转达"]:
    ids = tok.encode(w)
    in_trie = ids in []  # placeholder
    # 手工走一遍 trie
    st = trie.root
    ok = True
    for c in ids:
        st, pen = trie.step(st, c)
        if pen > 0:
            ok = False
    print(f"  {w!r}: ids={ids}  完整路径={'在' if ok else '不在'}词表")

print("\n" + "=" * 70)
for r in target:
    wav, _ = sf.read(r["audio"], dtype="float32")
    model.generate(input=wav, cache={}, language="zh", use_itn=False,
                   ban_emo_unk=False, batch_size_s=60, merge_vad=False)
    logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()

    print(f"\nref = {r['ref']!r}")
    for name, cfg in [
        ("baseline", dict(trie=None, boost=0.0, oov_pen=0.0)),
        ("soft 0.5", dict(trie=trie, boost=0.5, oov_pen=0.0)),
        ("hard 2", dict(trie=trie, boost=0.0, oov_pen=2.0)),
        ("hard 5", dict(trie=trie, boost=0.0, oov_pen=5.0)),
    ]:
        ids = ctc_beam(logp, BLANK, beam_size=10, topk=12, **cfg)
        hyp = rich_transcription_postprocess(tok.tokens2text(ids))
        e, n = cer_err(r["ref"], hyp)
        print(f"  {name:9s} CER={e/max(n,1)*100:6.2f}%  {hyp!r}")

    # 声学侧：看"主"与"础"的 token 概率
    id_zhu = tok.encode("主")[0]
    id_chu = tok.encode("础")[0]
    print(f"  token id: 主={id_zhu} 础={id_chu}")
    p_zhu = logp[:, id_zhu].max()
    p_chu = logp[:, id_chu].max()
    print(f"  整段音频上 主 的最高帧 logp={p_zhu:.3f}   础 的最高帧 logp={p_chu:.3f}")
