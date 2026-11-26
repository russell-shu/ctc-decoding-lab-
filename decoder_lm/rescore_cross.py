"""跨模型 rescoring：SenseVoice 的 CTC n-best + Qwen3-ASR 的声学似然。

两个模型的声学误判点互补（SenseVoice 的错里 34.5% 是 Qwen3-ASR 能答对的），
用 Qwen3-ASR 的 P(y|audio) 给 SenseVoice 的候选重排。

  score(y) = log P_sv(y) + alpha * mean_log P_qw(y|audio) + beta * len(y)

P_qw 用 teacher forcing 对候选文本前向得到（prompt 已含 "language Chinese<asr_text>"）。
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
import sys

import torch
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

sys.path.insert(0, DECODER_LM)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, QWEN_ROOT)
from ctc_beam import ctc_prefix_beam_search        # noqa: E402
from eval_qwen3_asr_cer import (                  # noqa: E402
    normalize_and_strip_punct_space as N, tokenize_mixed as T, calculate_cer_tokens as C,
)

SV_M = MODEL_SV
QW_M = MODEL_QWEN


def cer_err(ref, hyp):
    _, n, _, s, d, i = C(T(N(ref)), T(N(hyp)))
    return s + d + i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--alphas", default="0,0.5,1,2,3")
    ap.add_argument("--betas", default="0")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST_JSONL,
                                        encoding="utf-8") if l.strip()]
    if a.limit > 0:
        rows = rows[: a.limit]
    print(f"样本 {len(rows)}", flush=True)

    # ---- SenseVoice ----
    sv = AutoModel(model=SV_M, disable_update=True, device="cuda:0")
    net = sv.model
    stok = sv.kwargs["tokenizer"]
    buf = {}
    net.ctc.ctc_lo.register_forward_hook(lambda mod, i, o: buf.__setitem__("x", o.detach()))
    BLANK = net.blank_id

    # ---- Qwen3-ASR ----
    from qwen_asr import Qwen3ASRModel
    qw = Qwen3ASRModel.from_pretrained(QW_M, dtype=torch.bfloat16, device_map="cuda:0")
    qproc = qw.processor
    qmodel = qw.model.eval()
    qdev = next(qmodel.parameters()).device
    qdtype = next(qmodel.parameters()).dtype
    prompt = qw._build_text_prompt(context="", force_language="Chinese")

    @torch.no_grad()
    def qw_score(inputs, y_text):
        y_ids = qproc.tokenizer(y_text, add_special_tokens=False)["input_ids"]
        if not y_ids:
            return -30.0, 1
        L = inputs["input_ids"].shape[1]
        ids = torch.cat([inputs["input_ids"],
                         torch.tensor([y_ids], device=qdev)], dim=1)
        am = torch.cat([inputs["attention_mask"],
                        torch.ones((1, len(y_ids)), dtype=inputs["attention_mask"].dtype,
                                   device=qdev)], dim=1)
        # 顶层 Qwen3ASRForConditionalGeneration 的 forward 是未实现的封装，真正实现在 .thinker
        out = qmodel.thinker(input_ids=ids, attention_mask=am,
                             input_features=inputs["input_features"],
                             feature_attention_mask=inputs["feature_attention_mask"])
        logits = out.logits[0, L - 1: L - 1 + len(y_ids)]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tot = sum(lp[i, y_ids[i]].item() for i in range(len(y_ids)))
        return tot / len(y_ids), len(y_ids)

    cache = []
    for r in rows:
        wav, _ = sf.read(r["audio"], dtype="float32")
        sv.generate(input=wav, cache={}, language="zh", use_itn=False,
                    ban_emo_unk=False, batch_size_s=60, merge_vad=False)
        logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()
        cands = ctc_prefix_beam_search(logp, BLANK, beam_size=a.beam, topk=a.topk, topn=a.beam)

        qin = qproc(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        qin = {k: v.to(qdev) for k, v in qin.items()}
        qin["input_features"] = qin["input_features"].to(qdtype)

        items = []
        for ids, ctc_sc in cands:
            txt = rich_transcription_postprocess(stok.tokens2text(ids))
            qlp, _ = qw_score(qin, txt)
            items.append((ctc_sc, qlp, len(ids), txt))
        cache.append((r["ref"], items))

    alphas = [float(x) for x in a.alphas.split(",")]
    betas = [float(x) for x in a.betas.split(",")]
    best = None
    for alpha in alphas:
        for beta in betas:
            E = Nt = 0
            for ref, items in cache:
                b = max(items, key=lambda it: it[0] + alpha * it[1] + beta * it[2])
                e, n = cer_err(ref, b[3])
                E += e
                Nt += n
            cer = E / Nt * 100
            print(f"  alpha={alpha:<5} beta={beta:<4} CER={cer:6.2f}%", flush=True)
            if best is None or cer < best[2]:
                best = (alpha, beta, cer)

    E = Nt = 0
    for ref, items in cache:
        e_best = min(cer_err(ref, it[3])[0] for it in items)
        _, n = cer_err(ref, items[0][3])
        E += e_best
        Nt += n
    print(f"\n  oracle 上限 CER={E/Nt*100:.2f}%   best: alpha={best[0]} CER={best[2]:.2f}%")

    # 逐条诊断（alpha=0 vs best）
    if best[0] != 0:
        changed = good = bad = 0
        ex = []
        for ref, items in cache:
            b0 = max(items, key=lambda it: it[0])
            b1 = max(items, key=lambda it: it[0] + best[0] * it[1] + best[1] * it[2])
            if N(b0[3]) != N(b1[3]):
                changed += 1
                e0 = cer_err(ref, b0[3])[0]
                e1 = cer_err(ref, b1[3])[0]
                if e1 < e0:
                    good += 1
                elif e1 > e0:
                    bad += 1
                if len(ex) < 20:
                    ex.append((ref, b0[3], b1[3], e0, e1))
        print(f"  改动 {changed} 条: 改对 {good} / 改错 {bad}")
        for ref, t0, t1, e0, e1 in ex:
            mark = "✓" if e1 < e0 else ("✗" if e1 > e0 else "=")
            print(f"    {mark} ref={ref!r}")
            print(f"        a=0 : {t0!r}")
            print(f"        a={best[0]}: {t1!r}")


if __name__ == "__main__":
    main()
