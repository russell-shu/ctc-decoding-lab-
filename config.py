"""集中管理所有外部路径，便于在任意机器上复现。

所有路径都可用环境变量覆盖：

    SV_ROOT=/path/to/SenseVoice \
    MODEL_SV=/path/to/sv_nbest_v2_avg10 \
    MODEL_QWEN=/path/to/qwen3-asr-ft-out/merged \
    TEST_JSONL=/path/to/test.jsonl \
    python decoder_lm/rescore_cross.py
"""
import os

# ---- 仓库内 ----
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = _REPO
DECODER_LM = os.path.join(_REPO, "decoder_lm")
COMMON = os.path.join(_REPO, "common")
RESULTS = os.path.join(_REPO, "results")

# ---- 外部资源 ----
SV_ROOT = os.environ.get("SV_ROOT", "/data/russell/SenseVoice")
QWEN_ROOT = os.environ.get("QWEN_ROOT", "/data/russell/Qwen3-ASR")

MODEL_SV = os.environ.get("MODEL_SV", os.path.join(SV_ROOT, "sv_nbest_v2_avg10"))
MODEL_QWEN = os.environ.get("MODEL_QWEN", os.path.join(QWEN_ROOT, "qwen3-asr-ft-out/merged"))
BPE_MODEL = os.environ.get(
    "BPE_MODEL",
    os.path.join(SV_ROOT, "models/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model"),
)
TEST_JSONL = os.environ.get("TEST_JSONL", os.path.join(SV_ROOT, "qwen3_asr_dataset/test.jsonl"))
TEST_WAV = os.environ.get("TEST_WAV", os.path.join(SV_ROOT, "qwen3_asr_dataset/test/wav"))
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", os.path.join(SV_ROOT, "correction_data/train.jsonl"))

# 训练产物（不入库，用脚本重新生成）
LM3_PKL = os.environ.get("LM3_PKL", os.path.join(DECODER_LM, "lm3.pkl"))
NLM_DIR = os.environ.get("NLM_DIR", os.path.join(DECODER_LM, "nlm"))

# SenseVoice 词表共 25055：前 24884 为文本 token，其后为
# language / emotion / event / itn 等特殊 token，LM 只应在文本 token 上生效。
LM_MAX_ID = 24883
