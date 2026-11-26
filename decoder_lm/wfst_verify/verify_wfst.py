"""验证 WFST 词典硬约束的收益上限。

三组对比，共用同一份声学后验：
  A. baseline   —— 纯 CTC beam（等价现有 greedy/beam）
  B. soft boost —— 命中词表加分，不剪枝（trie boost）
  C. hard       —— 离开词表加惩罚，惩罚越大越接近 WFST 的硬剪枝

用法:
  python verify_wfst.py --limit 400
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
import collections
import json
import sys

import jieba
import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, REPO_ROOT)
from eval_qwen3_asr_cer import (  # noqa: E402
    normalize_and_strip_punct_space as N,
    tokenize_mixed as T,
    calculate_cer_tokens as C,
)

M = MODEL_SV
TRAIN = TRAIN_JSONL
TEST = TEST_JSONL
NEG = -1e30

PUNCT = set("，。？！、；：“”‘’（）《》…—,.?!;:\"'()<>- ")


class Trie:
    def __init__(self):
        self.children = [{}]
        self.is_end = [False]
        self.root = 0

    def add(self, ids):
        n = self.root
        for c in ids:
            nxt = self.children[n].get(c)
            if nxt is None:
                nxt = len(self.children)
                self.children[n][c] = nxt
                self.children.append({})
                self.is_end.append(False)
            n = nxt
        self.is_end[n] = True

    def step(self, state, c):
        """返回 (next_state, penalty)。penalty=1.0 表示该 token 不在词表覆盖内。"""
        nxt = self.children[state].get(c)
        if nxt is not None:
            return nxt, 0.0
        if self.is_end[state]:
            nxt = self.children[self.root].get(c)
            if nxt is not None:
                return nxt, 0.0
        return self.root, 1.0


def ctc_beam(log_probs, blank_id, beam_size=10, topk=12, trie=None,
             boost=0.0, oov_pen=0.0):
    """CTC prefix beam search，可选 trie 软加分 / OOV 惩罚。"""
    Tn, V = log_probs.shape
    # key: (prefix, trie_state) -> [p_blank, p_nonblank]
    beams = {((), 0): [0.0, NEG]}

    for t in range(Tn):
        lp = log_probs[t]
        k = min(topk, V)
        idx = np.argpartition(lp, -k)[-k:]
        idx = np.unique(np.append(idx, blank_id))
        blank_lp = float(lp[blank_id])
        new = {}

        def put(key, slot, val):
            cur = new.get(key)
            if cur is None:
                cur = [NEG, NEG]
                new[key] = cur
            cur[slot] = np.logaddexp(cur[slot], val)

        for (prefix, state), (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)
            put((prefix, state), 0, p_total + blank_lp)

            for c in idx:
                c = int(c)
                if c == blank_id:
                    continue
                lpc = float(lp[c])
                if prefix and c == prefix[-1]:
                    # 重复 token：CTC 折叠，prefix 与 state 都不变
                    put((prefix, state), 1, pnb + lpc)
                    if trie is None:
                        ns, pen = state, 0.0
                    else:
                        ns, pen = trie.step(state, c)
                    put((prefix + (c,), ns), 1,
                        p_total + lpc + boost * (1 - pen) - oov_pen * pen)
                else:
                    if trie is None:
                        ns, pen = state, 0.0
                    else:
                        ns, pen = trie.step(state, c)
                    put((prefix + (c,), ns), 1,
                        p_total + lpc + boost * (1 - pen) - oov_pen * pen)

        if len(new) > beam_size * 3:
            # 先按 prefix 合并不同 trie 状态，避免 beam 被状态分裂占满
            merged = {}
            for (prefix, _st), (pb, pnb) in new.items():
                cur = merged.get(prefix)
                if cur is None:
                    merged[prefix] = [pb, pnb]
                else:
                    cur[0] = np.logaddexp(cur[0], pb)
                    cur[1] = np.logaddexp(cur[1], pnb)
            # 再按总分会剪枝，state 取能到达的最高分状态
            best = sorted(new.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))
            keep = {}
            seen = collections.Counter()
            for (prefix, st), v in best:
                if seen[prefix] >= 1:
                    continue
                seen[prefix] += 1
                keep[(prefix, st)] = merged[prefix]
                if len(keep) >= beam_size:
                    break
            new = keep
        beams = new

    prefix, _ = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
    return list(prefix)


def build_lexicon(tok, min_freq=3, big=False):
    cnt = collections.Counter()
    n_line = 0
    for line in open(TRAIN, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        txt = "".join(ch for ch in r["ref"] if ch not in PUNCT)
        for w in jieba.lcut(txt):
            w = w.strip()
            if w:
                cnt[w] += 1
        n_line += 1
    print(f"训练集 {n_line} 条，分词后 {len(cnt)} 个词")

    if big:
        words = set()
        with open(jieba.get_dict_file().name, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    words.add(parts[0])
        words |= {w for w, c in cnt.most_common() if c >= min_freq and len(w) >= 2}
        words = sorted(words)
        singles = []
        print(f"大词典模式：{len(words)} 个词")
    else:
        words = [w for w, c in cnt.most_common() if c >= min_freq and len(w) >= 2]
        singles = [w for w, c in cnt.most_common() if c >= min_freq and len(w) == 1]
        print(f"词表：多字词 {len(words)}，单字词 {len(singles)}")

    extra = ["机主", "宜享花"]
    for w in extra:
        if w not in words:
            words.append(w)

    trie = Trie()
    n_tok = 0
    for w in list(words) + list(singles):
        ids = tok.encode(w)
        if not ids:
            continue
        trie.add(ids)
        n_tok += 1
    print(f"trie 节点数 {len(trie.children)}，入表词 {n_tok}")
    return trie, words, singles


def oov_stats(hyp, trie, tok):
    """返回 (oov_token 数, 总 token 数)。"""
    txt = "".join(ch for ch in hyp if ch not in PUNCT)
    ids = tok.encode(txt)
    st = trie.root
    oov = 0
    for c in ids:
        st, pen = trie.step(st, c)
        oov += pen
    return oov, len(ids)


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--big", action="store_true", help="用 jieba 全量词典")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    model = AutoModel(model=M, disable_update=True, device="cuda:0")
    net = model.model
    tok = model.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    trie, words, singles = build_lexicon(tok, big=a.big)

    print(f"\n开始前向 {len(rows)} 条 ...")
    cache = []
    for i, r in enumerate(rows):
        wav, _ = sf.read(r["audio"], dtype="float32")
        model.generate(input=wav, cache={}, language="zh", use_itn=False,
                       ban_emo_unk=False, batch_size_s=60, merge_vad=False)
        logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
        cache.append((logp, r["ref"]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}")

    configs = [
        ("baseline", dict(trie=None, boost=0.0, oov_pen=0.0)),
        ("soft 0.5", dict(trie=trie, boost=0.5, oov_pen=0.0)),
        ("soft 1.0", dict(trie=trie, boost=1.0, oov_pen=0.0)),
        ("soft 2.0", dict(trie=trie, boost=2.0, oov_pen=0.0)),
        ("hard 2", dict(trie=trie, boost=0.0, oov_pen=2.0)),
        ("hard 5", dict(trie=trie, boost=0.0, oov_pen=5.0)),
        ("hard 10", dict(trie=trie, boost=0.0, oov_pen=10.0)),
        ("hard 20", dict(trie=trie, boost=0.0, oov_pen=20.0)),
    ]

    results = {}
    for name, cfg in configs:
        E = Nn = 0
        hyps = []
        for logp, ref in cache:
            ids = ctc_beam(logp, BLANK, beam_size=a.beam, topk=a.topk, **cfg)
            hyp = rich_transcription_postprocess(tok.tokens2text(ids))
            e, n = cer_err(ref, hyp)
            E += e
            Nn += n
            hyps.append((ref, hyp, e))
        results[name] = (E / Nn * 100, hyps)
        print(f"{name:10s} CER = {E/Nn*100:.3f}%")
        if name == "baseline":
            ov = tot = 0
            for _, hyp, _ in hyps:
                o, t = oov_stats(hyp, trie, tok)
                ov += o
                tot += t
            print(f"           输出 OOV token 占比 {ov}/{tot} = {ov/max(tot,1)*100:.2f}%")

    # 相对 baseline 的改动统计
    base = results["baseline"][1]
    print("\n改动统计（相对 baseline）:")
    for name, (_, hyps) in results.items():
        if name == "baseline":
            continue
        ch = good = bad = 0
        for (_, h0, e0), (_, h1, e1) in zip(base, hyps):
            if N(h0) != N(h1):
                ch += 1
                if e1 < e0:
                    good += 1
                elif e1 > e0:
                    bad += 1
        print(f"  {name:10s} 改动 {ch:4d} 条  改对 {good:4d} / 改错 {bad:4d}")

    # 含"机主"的分层
    n_sub = sum(1 for (r, _, _) in base if "机主" in r)
    if n_sub:
        print(f'\n含"机主"样本 {n_sub} 条:')
        for name, (_, hyps) in results.items():
            hit = sum(1 for (ref, hyp, _) in hyps if "机主" in ref and "机主" in hyp)
            print(f"  {name:10s} 召回 {hit}/{n_sub}")


if __name__ == "__main__":
    main()
