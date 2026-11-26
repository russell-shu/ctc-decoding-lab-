# -*- coding: utf-8 -*-
"""在 pcm1229 测试集上用 Qwen3-ASR 评测 CER。

与 eval_backends_cer.py / evalute_sensevoice_sft_onnx.py 保持同一口径：
- 参考文本、音频路径解析复用同一份 parse_cer_report 逻辑（针对 cer_report.txt）
- 文本清洗 normalize_and_strip_punct_space / tokenize_mixed
- CER 计算 calculate_cer_tokens / levenshtein_tokens

上面这几个函数是从 evalute_sensevoice_sft_onnx.py 原样复制过来的，原因是该模块
顶层 import 了 kaldi_native_fbank（sensevoice 的 frontend 依赖），在 Qwen3-ASR 的
环境里不可用。复制而非 import，既避免引入 onnxruntime/funasr 依赖，又保证口径一致。

用法:
  python eval_qwen3_asr_cer.py --gpu 2                       # 默认全量, 自动语种
  python eval_qwen3_asr_cer.py --gpu 2 --language Chinese    # 强制中文
  python eval_qwen3_asr_cer.py --gpu 2 --limit 20            # 冒烟测试
  python eval_qwen3_asr_cer.py --gpu 2 --context "宜享花"     # 注入热词
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Tuple

SV = os.environ.get("SV_ROOT", "/data/russell/SenseVoice")
QWEN_ASR_REPO = os.environ.get("QWEN_ROOT", "/data/russell/Qwen3-ASR")
DEFAULT_MODEL = f"{QWEN_ASR_REPO}/Qwen3-ASR-1.7B"


# ---------- 文本清洗 / CER（复制自 evalute_sensevoice_sft_onnx.py） ----------

def normalize_and_strip_punct_space(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    out_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "Z", "C"):
            continue
        out_chars.append(ch.lower())
    return "".join(out_chars)


def tokenize_mixed(s: str) -> List[str]:
    if not s:
        return []
    tokens = []
    current_word = []
    for ch in s:
        if ch.isascii() and ch.isalnum():
            current_word.append(ch)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            if ch.strip():
                tokens.append(ch)
    if current_word:
        tokens.append("".join(current_word))
    return tokens


def levenshtein_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int, int]:
    if ref_tokens == hyp_tokens:
        return 0, len(ref_tokens), 0, 0, 0
    if len(ref_tokens) == 0:
        return len(hyp_tokens), 0, 0, 0, len(hyp_tokens)
    if len(hyp_tokens) == 0:
        return len(ref_tokens), 0, 0, len(ref_tokens), 0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = m, n
    correct = substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            correct += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            break
    return dp[m][n], correct, substitutions, deletions, insertions


def calculate_cer_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, int, int, int, int, int]:
    ref_len = len(ref_tokens)
    if ref_len == 0:
        if len(hyp_tokens) == 0:
            return 0.0, 0, 0, 0, 0, 0
        return float("inf"), 0, 0, 0, 0, len(hyp_tokens)
    dist, correct, subs, dels, ins = levenshtein_tokens(ref_tokens, hyp_tokens)
    return dist / ref_len, ref_len, correct, subs, dels, ins


def parse_cer_report(filepath: str) -> List[Dict]:
    """解析 cer_report.txt，音频名 .pcm -> pcm1229/*.wav"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    sections = content.split("-" * 80)
    results = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        audio_match = re.search(r"音频:\s*(.+)", section)
        ref_match = re.search(r"^参考:\s*(.*)$", section, re.MULTILINE)
        if audio_match and ref_match:
            audio_name = audio_match.group(1).strip()
            results.append({
                "audio": audio_name,
                "wav_path": f"pcm1229/{audio_name.replace('.pcm', '.wav')}",
                "ref_raw": ref_match.group(1).strip(),
            })
    return results


# ---------- 主流程 ----------

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR CER 评测 (pcm1229)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="本地模型目录")
    p.add_argument("--gpu", default="2", help="物理 GPU 序号, 写入 CUDA_VISIBLE_DEVICES")
    p.add_argument("--language", default="", help="强制语种, 如 Chinese; 留空=自动语种识别")
    p.add_argument("--context", default="", help="system message 上下文/热词")
    p.add_argument("--batch-size", type=int, default=16, help="推理批大小")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条, 0=全量")
    p.add_argument("--tag", default="", help="输出文件名后缀, 留空则自动生成")
    p.add_argument("--manifest", default="",
                   help="改用 JSONL 清单(每行含 audio/ref), 如 qwen3_asr_dataset/test.jsonl; 不传则用 cer_report.txt")
    p.add_argument("--out_dir", default="", help="报告输出目录, 默认与清单同目录(否则 SenseVoice 根目录)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, QWEN_ASR_REPO)
    import torch
    from qwen_asr import Qwen3ASRModel

    out_dir = args.out_dir
    if args.manifest:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
        valid = [{"audio": os.path.basename(r["audio"]), "wav_path": r["audio"],
                  "ref_raw": r.get("ref") or r["text"].split("<asr_text>", 1)[-1]}
                 for r in rows]
        valid = [it for it in valid if os.path.exists(it["wav_path"])]
        if not out_dir:
            out_dir = os.path.dirname(os.path.abspath(args.manifest))
    else:
        items = parse_cer_report(f"{SV}/cer_report.txt")
        valid = [{"audio": it["audio"], "wav_path": os.path.join(SV, it["wav_path"]),
                  "ref_raw": it["ref_raw"]}
                 for it in items if os.path.exists(os.path.join(SV, it["wav_path"]))]
        if not out_dir:
            out_dir = SV
    if args.limit:
        valid = valid[: args.limit]
    os.makedirs(out_dir, exist_ok=True)

    tag = args.tag or ("auto" if not args.language else args.language.lower())
    if args.context and not args.tag:
        tag += "_ctx"
    if args.manifest and not args.tag:
        tag += "_manifest"
    report_path = f"{out_dir}/cer_report_qwen3_asr_{tag}.txt"
    json_path = f"{out_dir}/eval_qwen3_asr_{tag}.json"

    print("=" * 70, flush=True)
    print(f"Qwen3-ASR CER 评测   model={args.model}", flush=True)
    print(f"样本 {len(valid)}  清单={args.manifest or SV + '/cer_report.txt'}  batch={args.batch_size}  "
          f"language={args.language or 'auto'}  context={args.context!r}", flush=True)
    print("=" * 70, flush=True)

    t_load = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"模型加载 {time.time() - t_load:.1f}s", flush=True)

    stats = {"n": 0, "c": 0, "s": 0, "d": 0, "i": 0}
    empty = 0
    rows = []
    no_lang = 0          # 未识别出语种且非空(疑似未按 language X<asr_text> 格式输出)
    langs: Dict[str, int] = {}

    t0 = time.time()
    bs = args.batch_size
    for start in range(0, len(valid), bs):
        batch = valid[start : start + bs]
        paths = [it["wav_path"] for it in batch]
        try:
            outs = model.transcribe(
                audio=paths,
                context=args.context,
                language=(args.language or None),
                return_time_stamps=False,
            )
            hyps = [o.text if o is not None else "" for o in outs]
            langs_out = [o.language if o is not None else "" for o in outs]
        except Exception as e:
            print(f"  !! 批推理异常 start={start}: {str(e)[:200]}", flush=True)
            hyps = [""] * len(batch)
            langs_out = [""] * len(batch)

        for it, hyp, lg in zip(batch, hyps, langs_out):
            hyp = hyp or ""
            if not hyp.strip():
                empty += 1
            lg = (lg or "").strip()
            langs[lg or "(empty)"] = langs.get(lg or "(empty)", 0) + 1
            if not lg and hyp.strip():
                no_lang += 1

            cer, n, c, s, d, i = calculate_cer_tokens(
                tokenize_mixed(normalize_and_strip_punct_space(it["ref_raw"])),
                tokenize_mixed(normalize_and_strip_punct_space(hyp)),
            )
            if n > 0:
                for k, v in zip("ncsdi", (n, c, s, d, i)):
                    stats[k] += v
            cer_pct = 100.0 if cer == float("inf") else cer * 100
            rows.append({"audio": it["audio"], "ref": it["ref_raw"], "hyp": hyp,
                         "language": lg, "cer": round(cer_pct, 2)})

        done = start + len(batch)
        if done % 160 < bs or done >= len(valid):
            el = time.time() - t0
            print(f"  {done}/{len(valid)}  empty={empty}  no_lang={no_lang}  "
                  f"{el:.0f}s ({done / max(el, 1e-9):.1f} it/s)", flush=True)
        if done % 320 < bs:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"partial": True, "rows": rows}, f, ensure_ascii=False)

    elapsed = time.time() - t0
    overall = (stats["s"] + stats["d"] + stats["i"]) / stats["n"] * 100 if stats["n"] else 0.0

    summary = {
        "model": args.model, "language": args.language or "auto",
        "context": args.context, "samples": len(valid), "overall_cer": round(overall, 2),
        "empty": empty, "no_lang_text": no_lang, "languages": langs,
        "N": stats["n"], "C": stats["c"], "S": stats["s"], "D": stats["d"], "I": stats["i"],
        "elapsed_s": round(elapsed, 1),
    }

    lines = ["", "=" * 100, "Qwen3-ASR 评测报告",
             f"模型: {args.model}",
             f"language: {args.language or 'auto'}   context: {args.context!r}",
             f"样本数: {len(valid)}   总参考 token: {stats['n']}",
             f"正确: {stats['c']}  替换: {stats['s']}  删除: {stats['d']}  插入: {stats['i']}",
             f"空输出: {empty}   无语种标签但有文本: {no_lang}",
             f"总体 CER: {overall:.2f}%",
             "=" * 100, ""]
    for r in rows:
        lines.append(f"音频: {r['audio']}")
        lines.append(f"参考: {r['ref']}")
        lines.append(f"Qwen3_ASR: {r['hyp']}")
        lines.append(f"CER: {r['cer']:.2f}%  (lang={r['language']!r})")
        lines.append("-" * 100)
        lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)

    print("", flush=True)
    print(f"结果 [{tag}]: CER={overall:.2f}%  空输出={empty}/{len(valid)}  "
          f"无标签={no_lang}  N={stats['n']} S={stats['s']} D={stats['d']} I={stats['i']}  "
          f"({elapsed:.0f}s, {elapsed / max(len(valid), 1) * 1000:.0f} ms/条)", flush=True)
    print(f"语种分布: {langs}", flush=True)
    print(f"已保存 {report_path}", flush=True)
    print(f"已保存 {json_path}", flush=True)


if __name__ == "__main__":
    main()
