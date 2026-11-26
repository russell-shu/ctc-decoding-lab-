"""CTC prefix beam search + n-gram LM 浅融合（解码器内部语言模型）。

相比"后置纯文本纠错"，这里的 LM 打分发生在解码过程中，能同时利用
声学的帧级概率和语言先验（shallow fusion）。

  score = log P_ctc(prefix) + alpha * log P_lm(token | prefix) + beta

- beam search: 标准 CTC prefix beam search（Graves），log 域
- LM: token 级 n-gram + stupid backoff（无外部依赖）
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import collections
import math

import numpy as np

NEG = -1e30


class NgramLM:
    """token 级 n-gram LM，带 stupid backoff。

    score(prefix, c) 返回 log P(c | prefix)，在 log 域（<=0）。
    """

    def __init__(self, order=3, backoff=0.4):
        self.order = order
        self.backoff = backoff
        self.counts = [collections.Counter() for _ in range(order + 1)]
        self.start_id = None  # 序列起始标记

    def train(self, sequences):
        for seq in sequences:
            s = list(seq)
            for n in range(1, self.order + 1):
                for i in range(len(s) - n + 1):
                    self.counts[n][tuple(s[i:i + n])] += 1
        # 预计算 unigram 总数（含 <unk> 平滑）
        self._uni_total = sum(self.counts[1].values())
        self._uni_vocab = len(self.counts[1])

    def score(self, prefix, c):
        """log P(c | prefix)，stupid backoff。"""
        for n in range(self.order, 1, -1):
            if n - 1 > len(prefix):
                continue
            ctx = tuple(prefix[-(n - 1):])
            ng = ctx + (c,)
            denom = self.counts[n - 1].get(ctx, 0)
            num = self.counts[n].get(ng, 0)
            if denom > 0 and num > 0:
                return math.log(num / denom)
        # unigram（加一点平滑，避免 log(0)）
        num = self.counts[1].get((c,), 0)
        return math.log((num + 1e-3) / (self._uni_total + self._uni_vocab * 1e-3))


def ctc_prefix_beam_search(log_probs, blank_id, beam_size=10, topk=12,
                           lm=None, alpha=0.0, beta=0.0, lm_min_id=0,
                           lm_max_id=None, max_len=None, topn=1):
    """标准 CTC prefix beam search（log 域）+ 可选 LM 浅融合。

    log_probs: [T, V] numpy array（已 log_softmax）
    返回: (best_prefix_ids, best_prefix_score)
    """
    T, V = log_probs.shape
    if max_len is None:
        max_len = T
    # beams: prefix(tuple) -> [p_blank, p_nonblank]（log 域）
    beams = {(): [0.0, NEG]}

    for t in range(T):
        lp = log_probs[t]
        k = min(topk, V)
        idx = np.argpartition(lp, -k)[-k:]
        idx = np.unique(np.append(idx, blank_id))
        blank_lp = float(lp[blank_id])
        new = {}

        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)
            # 1) blank：prefix 不变
            cur = new.get(prefix)
            if cur is None:
                cur = [NEG, NEG]
                new[prefix] = cur
            cur[0] = np.logaddexp(cur[0], p_total + blank_lp)

            if len(prefix) >= max_len:
                continue
            # 2) 非 blank
            for c in idx:
                c = int(c)
                if c == blank_id:
                    continue
                lpc = float(lp[c])
                use_lm = (lm is not None and alpha != 0.0 and c >= lm_min_id
                          and (lm_max_id is None or c <= lm_max_id))
                if prefix and c == prefix[-1]:
                    # repeat：不产生新 token，只在 pnb 上延续
                    cur = new.get(prefix)
                    if cur is None:
                        cur = [NEG, NEG]
                        new[prefix] = cur
                    cur[1] = np.logaddexp(cur[1], pnb + lpc)
                    # 从 p_blank 延展成新 prefix（CTC 的 "重复字" 路径）
                    np_ = prefix + (c,)
                    sc = p_total + lpc
                    if use_lm:
                        sc += alpha * lm.score(prefix, c) + beta
                    cur2 = new.get(np_)
                    if cur2 is None:
                        cur2 = [NEG, NEG]
                        new[np_] = cur2
                    cur2[1] = np.logaddexp(cur2[1], sc)
                else:
                    np_ = prefix + (c,)
                    sc = p_total + lpc
                    if use_lm:
                        sc += alpha * lm.score(prefix, c) + beta
                    cur = new.get(np_)
                    if cur is None:
                        cur = [NEG, NEG]
                        new[np_] = cur
                    cur[1] = np.logaddexp(cur[1], sc)

        if len(new) > beam_size:
            scored = sorted(new.items(),
                            key=lambda kv: -(np.logaddexp(kv[1][0], kv[1][1])))
            new = dict(scored[:beam_size])
        beams = new

    scored = sorted(beams.items(), key=lambda kv: -(np.logaddexp(kv[1][0], kv[1][1])))
    if topn <= 1:
        p, (pb, pnb) = scored[0]
        return list(p), float(np.logaddexp(pb, pnb))
    return [(list(p), float(np.logaddexp(pb, pnb))) for p, (pb, pnb) in scored[:topn]]
