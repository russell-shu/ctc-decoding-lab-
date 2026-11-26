# 复现指南

## 1. 环境

推荐 conda 环境（已验证可用）：

```bash
conda create -n sensevoice_ft python=3.10 -y
conda activate sensevoice_ft
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.6 funasr==1.3.14 datasets soundfile numpy sentencepiece jieba
```

### 已知依赖坑：Qwen3-ASR 需要 `nagisa`

`qwen_asr/inference/qwen3_forced_aligner.py:21` 在**顶层** `import nagisa`，
缺了会阻断整个 `qwen_asr` 包的导入（即使你只用 ASR、不用强制对齐）。

**不要直接装进主环境**——`pip install --target` 会把 numpy 2.x 一起装进来并
遮蔽环境的 numpy 1.26（与 torch 2.3 不兼容）。正确做法：

```bash
pip install --target /tmp/qwen_asr_deps nagisa soynlp
rm -rf /tmp/qwen_asr_deps/{numpy,numpy.libs,numpy-*.dist-info,scipy*,sklearn,\
scikit_learn*,joblib,Cython,cython*,pyximport,bin,__pycache__,six*,threadpoolctl*,\
cloudpickle*,psutil-*.dist-info}
export PYTHONPATH=/tmp/qwen_asr_deps:$PYTHONPATH
```

或直接用仓库内的脚本：

```bash
bash scripts/setup_deps.sh
```

> `soynlp` 是懒加载、`qwen_omni_utils` 在代码里未被引用，中文评测都不会用到，
> 但一并装上可避免其他分支 import 失败。

## 2. 配置路径

所有路径集中在 `config.py`，可用环境变量覆盖：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SV_ROOT` | `/data/russell/SenseVoice` | SenseVoice 仓库根 |
| `QWEN_ROOT` | `/data/russell/Qwen3-ASR` | Qwen3-ASR 仓库根（需 `pip install -e`） |
| `MODEL_SV` | `$SV_ROOT/sv_nbest_v2_avg10` | 微调后的 SenseVoice |
| `MODEL_QWEN` | `$QWEN_ROOT/qwen3-asr-ft-out/merged` | 微调后的 Qwen3-ASR |
| `TEST_JSONL` | `$SV_ROOT/qwen3_asr_dataset/test.jsonl` | 测试集 2070 条 |
| `TRAIN_JSONL` | `$SV_ROOT/correction_data/train.jsonl` | 训练文本 69871 条 |

```bash
export SV_ROOT=/data/russell/SenseVoice
export QWEN_ROOT=/data/russell/Qwen3-ASR
export PYTHONPATH=/tmp/qwen_asr_deps:$PYTHONPATH
```

## 3. 数据与模型

本仓库**不含**数据与权重，需自备：

```
$SV_ROOT/
├── sv_nbest_v2_avg10/                     # SenseVoice 域内微调（nbest-10 权重平均）
├── models/SenseVoiceSmall/
│   └── chn_jpn_yue_eng_ko_spectok.bpe.model
├── qwen3_asr_dataset/
│   ├── train.jsonl                        # 训练集
│   ├── test.jsonl                         # 测试集 2070 条
│   └── test/wav/
└── correction_data/train.jsonl            # 用于训语言模型的文本
```

SenseVoice 微调脚本见 `$SV_ROOT/finetune_v2.sh`；
Qwen3-ASR 微调见 `qwen3-asr-ft-out/`。

## 4. 执行

```bash
bash scripts/run_all.sh
```

或分步执行（见 README §4）。

## 5. 预期输出

| 实验 | 预期关键输出 |
| --- | --- |
| `smoke_beam.py` | `greedy CER = 8.63%  beam(no LM) CER = 8.67%` |
| `compare_alpha.py` | α=0.1 最优（+0.02pt），α=0.3 恶化到 9.09% |
| `rescore_nbest.py` | +0.18pt；末尾打印 oracle 上界 |
| `rescore_nlm.py` | +0.06pt |
| `nbest_oracle.py` | ref 出现率 93.2%，oracle CER 3.12% |
| `oracle_cases.py` | 57% 的情况正确答案 PPL 更高 |
| `wfst_verify/verify_wfst.py` | baseline 8.594%，输出 OOV 占比 1.78%，各约束配置全部变差 |
| `rescore_cross.py` | CER 由 8.63% 降至 7.97% |

数字与 `docs/FINDINGS.md` 一致。**若结果显著偏离，先检查模型路径与数据集版本**。

## 6. 运行时长参考

单卡 RTX 4090：

| 脚本 | 400 条样本耗时 |
| --- | --- |
| `smoke_beam.py`（20 条） | ~20 s |
| `compare_alpha.py` | ~2 min |
| `rescore_nbest.py` | ~2 min |
| `rescore_nlm.py` | ~2 min |
| `rescore_cross.py` | ~3 min（需加载第二个模型） |
| `wfst_verify/verify_wfst.py` | ~2 min（大词典构建约 30 s） |

> `rescore_*` 系列都做了候选缓存：**声学前向只跑一次，扫 α/β 为纯 CPU 运算**。
