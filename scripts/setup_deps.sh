#!/usr/bin/env bash
# 补齐 Qwen3-ASR 的可选依赖（nagisa / soynlp），装到独立目录避免污染主环境。
#
# 背景：qwen_asr/inference/qwen3_forced_aligner.py:21 在顶层 import nagisa，
# 缺失会阻断整个 qwen_asr 包的导入。用 --target 安装时必须清掉 numpy 2.x，
# 否则会遮蔽环境里的 numpy 1.26，与 torch 2.3 不兼容。
set -e
DEPS=${DEPS:-/tmp/qwen_asr_deps}

pip install --target "$DEPS" nagisa soynlp
rm -rf "$DEPS"/{numpy,numpy.libs,numpy-*.dist-info,scipy*,sklearn,scikit_learn*,joblib,\
Cython,cython*,pyximport,bin,__pycache__,six*,threadpoolctl*,cloudpickle*,psutil-*.dist-info}

echo "已安装到 $DEPS"
echo "使用： export PYTHONPATH=$DEPS:\$PYTHONPATH"
