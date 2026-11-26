# ctc-decoding-lab

**SenseVoice CTC 解码与重排实验集**

一个自包含的实验仓库，围绕「CTC 输出如何与语言模型 / 第二意见结合」这条主线，
系统验证了 6 类 12 组方案，并用排除法定位到唯一产生正收益的路径。

主模型：`SenseVoice-Small`（234M，非自回归 / CTC），域内微调后 CER **8.63%**。

---

## 一、核心结论

| # | 方案 | 机制 | CER | 判定 |
|---|---|---|---|---|
| 0 | greedy（基线） | 帧级 argmax | 8.63% | — |
| 1 | CTC prefix beam search | 自实现，无 LM | 8.67% | 证伪 |
| 2 | n-gram 浅融合（α=0.1） | beam 内逐 token 加权 | 8.65% (+0.02) | 证伪 |
| 3 | n-gram N-best 重排 | 出候选后整句重打分 | 8.47% (+0.18)※ | 证伪 |
| 4 | 神经 LM（GPT2）N-best 重排 | 同上，LM 换神经 | 8.59% (+0.06)※ | 证伪 |
| 5 | WFST 词典硬约束 | 词典 trie 剪枝 | 8.65%+ | 证伪 |
| 6 | **跨模型声学重排** | 第二个 ASR 的 `log P(y\|audio)` | **7.97% (−0.66)** | ✅ **有效** |

※ 该组在 400 条子集上评测，其余为完整测试集（2070 条）。

### 为什么只有最后一个有效

三个独立实验互相印证，把错误的性质钉死了：

```
① 输出不合法的字符串？  → 不是
   证据：模型输出 OOV 率仅 1.78%（34.9 万词全量词典），
        词典约束的理论上界只有 1.78pt，实测全部变差

② 输出语言上不合理？    → 不是
   证据：N-best oracle 达 3.12%（比基线低 5.5pt），但逐条诊断显示
        57% 的情况「正确答案的语言 PPL 反而更高」

③ 合法且通顺，只是声学上选错了  → 是
   → 只有第二个声学模型能提供区分度
```

**结论**：错误既不是「说得不合法」，也不是「说得不通顺」，而是「听得不对」。
因此任何语言先验类手段都无效，唯一可行的是引入**另一份声学证据**。

### 互补性

两个模型的声学误判点有 **34.5% 不重叠**——SenseVoice 判错的样本里，
34.5% Qwen3-ASR 能答对。这是重排能带来收益的直接来源。

---

## 二、仓库结构

```
ctc-decoding-lab/
├── README.md                      本文件
├── config.py                      全部路径集中管理（可用环境变量覆盖）
├── requirements.txt
│
├── decoder_lm/                    核心：解码器与重排
│   ├── ctc_beam.py                ★ CTC prefix beam search + n-gram 浅融合
│   ├── probe_ctc.py               ★ 帧级后验探针（forward hook 取 ctc_lo 输出）
│   ├── smoke_beam.py              冒烟测试：greedy vs beam
│   ├── tune_beam.py               beam size / topk 扫描
│   ├── compare_alpha.py           浅融合 α 扫描 + 逐条改动统计
│   │
│   ├── train_lm.py                token 级 3-gram LM 训练（stupid backoff）
│   ├── train_neural_lm.py         GPT2 领域 LM 训练（38.2M，vocab 对齐 ASR）
│   ├── rescore_nbest.py           N-best + n-gram 重排
│   ├── rescore_nlm.py             N-best + 神经 LM 重排
│   ├── rescore_cross.py           ★★ 跨模型声学重排（唯一有效方案）
│   │
│   ├── nbest_oracle.py            N-best 候选池 oracle 上界
│   ├── oracle_cases.py            ★ 逐条诊断：为什么 LM 选不出来
│   │
│   └── wfst_verify/               词典硬约束验证
│       ├── verify_wfst.py         trie 硬/软约束对比 + OOV 统计
│       ├── case_jizhu.py          个案分析（含/不含某词的逐条对比）
│       └── probe.py               tokenizer 接口探针
│
├── common/
│   └── eval_qwen3_asr_cer.py      统一 CER 评测口径（S/D/I 分解）
│
├── docs/
│   ├── FINDINGS.md                全部实验结论与根因分析
│   ├── CTC_LM_INTEGRATION.md      CTC+LM 三种结合方式：原理 + 代码对照
│   └── REPRODUCE.md               复现指南
│
├── scripts/
│   └── run_all.sh                 一键复现全部实验
│
└── results/
    └── tune_result.json           beam/α 扫描结果
```

---

## 三、核心实现：三处关键改动

整个实验链路**没有修改任何模型代码**，全部靠 forward hook + 自写解码器。

### 1. 取帧级后验（`probe_ctc.py`）

SenseVoice 的 `generate()` 内部走 greedy CTC，但 `ctc_lo` 线性层照样输出
完整的 `[T, V]` 帧级 logits。hook 截下来即得原料：

```python
buf = {}
net.ctc.ctc_lo.register_forward_hook(lambda m, i, o: buf.__setitem__("x", o.detach()))

model.generate(input=wav, cache={}, language="zh", use_itn=False,
               ban_emo_unk=False, batch_size_s=60, merge_vad=False)

logp = torch.log_softmax(buf["x"][0].float(), dim=-1).cpu().numpy()   # [T, V]
```

### 2. 浅融合（`ctc_beam.py`）—— beam 扩展时加一行

```python
sc = p_total + lpc                     # CTC 声学分
if use_lm:
    sc += alpha * lm.score(prefix, c) + beta    # ←←← 浅融合的全部改动
```

LM 是独立的 n-gram（`stupid backoff`），只在文本 token 上生效
（`LM_MAX_ID = 24883`，其后为 language/emotion/event 等特殊 token）。

### 3. 跨模型重排（`rescore_cross.py`）—— 替换打分函数

```python
@torch.no_grad()
def qw_score(inputs, y_text):
    y_ids = qproc.tokenizer(y_text, add_special_tokens=False)["input_ids"]
    L = inputs["input_ids"].shape[1]
    ids = torch.cat([inputs["input_ids"], torch.tensor([y_ids], device=qdev)], dim=1)
    out = qmodel.thinker(input_ids=ids, attention_mask=am,
                         input_features=inputs["input_features"],
                         feature_attention_mask=inputs["feature_attention_mask"])
    logits = out.logits[0, L - 1: L - 1 + len(y_ids)]      # off-by-one：shift 后取
    lp = torch.log_softmax(logits.float(), dim=-1)
    tot = sum(lp[i, y_ids[i]].item() for i in range(len(y_ids)))
    return tot / len(y_ids), len(y_ids)                   # 长度归一化
```

两个模型 tokenizer 完全不同（SenseVoice: 25055 中文一字一 token；
Qwen3-ASR: 15 万+ byte-level BPE），**通过「文本」这一中间层解耦**：

```
SenseVoice token ids ──tokens2text()──→ 文本 ──Qwen tokenizer──→ Qwen token ids
```

要点：
- **teacher forcing**，一次前向拿到全部位置的预测分布，不用逐候选生成
- **长度归一化**：`log P` 是负数累加，不归一化会系统性偏好短句
- 走 `.thinker`：顶层 `Qwen3ASRForConditionalGeneration` 只是封装

最终打分：

```
score(y) = log P_ctc_sv(y) + α · mean_log P_qwen(y|audio) + β · len(y)
```

---

## 四、复现

```bash
pip install -r requirements.txt

# 路径可用环境变量覆盖
export SV_ROOT=/data/russell/SenseVoice
export QWEN_ROOT=/data/russell/Qwen3-ASR

# 1. 训练语言模型
python decoder_lm/train_lm.py 3              # -> decoder_lm/lm3.pkl
python decoder_lm/train_neural_lm.py         # -> decoder_lm/nlm/

# 2. 基础验证
python decoder_lm/smoke_beam.py              # greedy vs beam
python decoder_lm/tune_beam.py               # beam size 扫描

# 3. 语言模型路线（预期：无收益）
python decoder_lm/compare_alpha.py --alpha 0.2
python decoder_lm/rescore_nbest.py --limit 400
python decoder_lm/rescore_nlm.py   --limit 400

# 4. 上界与根因
python decoder_lm/nbest_oracle.py --limit 400
python decoder_lm/oracle_cases.py --limit 400

# 5. 有效方案
python decoder_lm/rescore_cross.py --limit 400

# 6. 词典约束验证
python decoder_lm/wfst_verify/verify_wfst.py --limit 400
python decoder_lm/wfst_verify/verify_wfst.py --limit 400 --big
```

一键复现：`bash scripts/run_all.sh`

---

## 五、依赖的外部资源

| 资源 | 默认路径 | 说明 |
|---|---|---|
| 微调后的 SenseVoice | `$SV_ROOT/sv_nbest_v2_avg10` | nbest-10 权重平均，CER 8.63% |
| Qwen3-ASR（微调） | `$QWEN_ROOT/qwen3-asr-ft-out/merged` | 跨模型重排的第二意见 |
| SenseVoice BPE | `$SV_ROOT/models/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model` | token 表 |
| 测试集 | `$SV_ROOT/qwen3_asr_dataset/test.jsonl` | 2070 条 |
| 训练文本 | `$SV_ROOT/correction_data/train.jsonl` | 69871 条，用于训 LM |

所有路径见 `config.py`，均可用同名环境变量覆盖。

---

## 六、说明

- 仓库**不含模型权重与数据集**，只含实现与实验脚本
- `decoder_lm/lm*.pkl` 与 `decoder_lm/nlm/` 为训练产物，已 gitignore，用脚本重新生成
- CER 口径统一来自 `common/eval_qwen3_asr_cer.py`（S/D/I 分解，中文按字）

## License

Apache-2.0
