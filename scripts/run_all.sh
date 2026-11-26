#!/usr/bin/env bash
# 一键复现全部实验。路径可用环境变量覆盖：
#   SV_ROOT=/path/to/SenseVoice QWEN_ROOT=/path/to/Qwen3-ASR bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/data/miniconda3/envs/sensevoice_ft/bin/python}

echo "=== 1. 训练语言模型 ==="
$PY decoder_lm/train_lm.py 3
$PY decoder_lm/train_neural_lm.py

echo "=== 2. 基线：greedy vs beam ==="
$PY decoder_lm/smoke_beam.py
$PY decoder_lm/tune_beam.py

echo "=== 3. 语言模型路线（预期无收益）==="
$PY decoder_lm/compare_alpha.py --alpha 0.2
$PY decoder_lm/rescore_nbest.py --limit 400
$PY decoder_lm/rescore_nlm.py   --limit 400

echo "=== 4. 上界与根因诊断 ==="
$PY decoder_lm/nbest_oracle.py --limit 400
$PY decoder_lm/oracle_cases.py --limit 400

echo "=== 5. 词典约束验证（预期无收益）==="
$PY decoder_lm/wfst_verify/verify_wfst.py --limit 400
$PY decoder_lm/wfst_verify/verify_wfst.py --limit 400 --big

echo "=== 6. 有效方案：跨模型重排 ==="
$PY decoder_lm/rescore_cross.py --limit 400

echo "全部完成。"
