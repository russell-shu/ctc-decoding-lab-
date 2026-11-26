# CTC 与语言模型结合：四种方式

CTC 的输出是**逐帧独立**的后验分布，没有标签间的依赖关系。
要引入语言先验，标准做法有三类，另有本仓库实现的第四类（跨模型声学重排）。

本文给出每种的**原理、公式、代码位置、实测结果与适用条件**。

---

## 方式一：WFST 图解码（Weighted Finite State Transducer）

### 原理

把三样东西编译成**一张有限状态转换器**，解码 = 在图上找最优路径。

传统 HMM 语音识别用四层组合：

| 层 | 含义 |
| --- | --- |
| **H** | HMM 拓扑（帧 → 状态） |
| **C** | 上下文依赖（triphone） |
| **L** | 词典（音素序列 → 词） |
| **G** | 语言模型（n-gram） |

即 **HCLG**。CTC 模型自带 blank 与重复折叠，不需要 H 与 C，简化成 **TLG**：

| 层 | 含义 | SenseVoice 上如何构建 |
| --- | --- | --- |
| **T** | token 层（compact-CTC topology） | `tokens.json` |
| **L** | 词典（token 序列 → 词） | `generate_lexicon_svs.py` + BPE 模型 |
| **G** | n-gram LM | srilm 在语料上训练 |

参考实现：`duj12/FunASR` 的 `svs-tlg` 分支。
**关键**：Stage 1 必须用 SenseVoice 自己的 `tokens.json` 和 BPE 模型生成词典，
否则 token id 对不上，图解码直接崩。

### 与前两种方式的本质差别

```
浅融合 / 重排：给非法候选「扣分」，但它仍是一条合法路径
WFST：         直接把它从搜索空间里「删掉」，走都走不到
```

| | 浅融合 | N-best 重排 | WFST |
| --- | --- | --- | --- |
| 约束时机 | 解码中逐 token | 解码后 | 解码中，结构性 |
| 约束类型 | 软（加权） | 软（加权） | **硬（剪枝）** |
| 词典约束 | 无 | 无 | **有** |
| 能排除非法串 | 弱 | 弱 | **强** |
| 能区分两个合法词 | 弱 | 中 | **不能** |
| OOV 词 | 能出 | 能出 | **出不来**（须改词典重编译） |
| 工程成本 | 低 | 低 | **高**（编译图 + C++ runtime） |
| 加新词成本 | 0 | 0 | **重编译图** |

### 本仓库的实测

在 Python 侧用 trie 模拟（`decoder_lm/wfst_verify/verify_wfst.py`）：

- 模型输出 OOV 率仅 **1.78%** → 作用空间上限 1.78pt
- soft boost 全部大幅变差（+2.8 ~ +8.7pt）
- hard 惩罚 λ=2 时 +0.06pt（≈0），λ=20 时 +9pt

**结论：词典约束在本域无效**，因为错误不是「输出了不存在的词」。

---

## 方式二：浅融合（Shallow Fusion）

### 原理

在 beam search 每一步扩展候选时，把声学分与语言分加权相加：

```
score(prefix + c) = log P_ctc(prefix + c) + α · log P_lm(c | prefix) + β
```

- 两个模型**完全独立**，只在打分时加权，不联合训练 —— 所以叫「浅」
- 深度融合（deep fusion）是把 LM 的隐藏状态接进声学解码器联合训练

### 代码

`decoder_lm/ctc_beam.py`，核心改动只有一行：

```python
sc = p_total + lpc                              # CTC 声学分
if use_lm:
    sc += alpha * lm.score(prefix, c) + beta    # ←←← 浅融合的全部改动
```

LM 实现（token 级 3-gram + stupid backoff）：

```python
class NgramLM:
    def score(self, prefix, c):
        for n in range(self.order, 1, -1):      # 3-gram → 2-gram → unigram
            ctx = tuple(prefix[-(n - 1):])
            ng = ctx + (c,)
            denom = self.counts[n - 1].get(ctx, 0)
            num = self.counts[n].get(ng, 0)
            if denom > 0 and num > 0:
                return math.log(num / denom)
        num = self.counts[1].get((c,), 0)
        return math.log((num + 1e-3) / (self._uni_total + self._uni_vocab * 1e-3))
```

### 必须注意

`LM_MAX_ID = 24883`：SenseVoice 词表 25055，前 24884 是文本 token，
其后是 language / emotion / event / itn 等特殊 token。
LM 只在文本 token 上生效，否则它会去预测「情感标签」，纯属噪声。

### 实测

| α | CER |
| --- | --- |
| 0.0 | 8.675% |
| **0.1** | **8.652%** |
| 0.2 | 8.744% |
| 0.3 | 9.089% |

最优 +0.02pt（噪声级）。**α 曲线非单调**是关键信号——
若 LM 携带有效信息，应存在稳定最优区间。

---

## 方式三：N-best 重排（Rescoring）

### 原理

两阶段：

```
阶段 1：声学模型 beam search → N-best 候选
阶段 2：对每个候选整句打分 → 重排
         score(y) = log P_ctc(y) + α · log P_lm(y) + β · len(y)
```

区别于浅融合：`P_lm(y)` 是**整句**概率而非逐 token 条件概率，
因此可以上更强的模型（神经 LM、BERT、LLM）。

### 代码

`decoder_lm/rescore_nbest.py`（n-gram）与 `rescore_nlm.py`（GPT2）。

整句 n-gram 分数：

```python
def lm_logprob(lm, ids):
    s = 0.0
    for i, c in enumerate(ids):
        if c > LM_MAX_ID:
            continue
        s += lm.score(tuple(ids[:i]), c)      # 逐位置取条件概率
    return s
```

神经 LM 分数（teacher forcing）：

```python
@torch.no_grad()
def nlm_logprob(model, ids, dev):
    seq = [c for c in ids if c <= LM_MAX_ID]
    if len(seq) < 2:
        return -1e9
    x = torch.tensor([seq], device=dev)
    out = model(input_ids=x, labels=x)         # HF 自动 shift + 交叉熵
    return -out.loss.item() * (len(seq) - 1)   # 平均 NLL → 总 log prob
```

### 适用条件

- 受限于 N-best 召回：正确答案不在候选池里就救不回来
- 无词典约束
- **可以换任意强模型**，工程上最灵活

### 实测

| 方法 | 样本 | CER | 净收益 |
| --- | --- | --- | --- |
| N-best + n-gram | 400 | 8.47% | +0.18pt（噪声） |
| N-best + 神经 LM | 400 | 8.59% | +0.06pt |

---

## 方式四：跨模型声学重排（本仓库实现，唯一有效）

### 原理

与 N-best 重排同构，但打分函数换成**第二个 ASR 的声学条件概率**：

```
score(y) = log P_ctc_sv(y) + α · mean_log P_qwen(y | audio) + β · len(y)
```

### tokenizer 不一致怎么解决

**不对齐，用「文本」做中间表示。**

```
SenseVoice token ids ──tokens2text()──→ 文本 ──Qwen tokenizer──→ Qwen token ids
```

SenseVoice: 25055，中文一字一 token；Qwen3-ASR: 15 万+，byte-level BPE。
两者 token 表完全独立，因为比较的基准是**字符序列本身**。

### 代码

`decoder_lm/rescore_cross.py`：

```python
@torch.no_grad()
def qw_score(inputs, y_text):
    y_ids = qproc.tokenizer(y_text, add_special_tokens=False)["input_ids"]
    L = inputs["input_ids"].shape[1]
    ids = torch.cat([inputs["input_ids"],
                     torch.tensor([y_ids], device=qdev)], dim=1)
    am = torch.cat([inputs["attention_mask"],
                     torch.ones((1, len(y_ids)),
                                dtype=inputs["attention_mask"].dtype,
                                device=qdev)], dim=1)
    out = qmodel.thinker(input_ids=ids, attention_mask=am,
                         input_features=inputs["input_features"],
                         feature_attention_mask=inputs["feature_attention_mask"])
    logits = out.logits[0, L - 1: L - 1 + len(y_ids)]   # off-by-one：shift 后取
    lp = torch.log_softmax(logits.float(), dim=-1)
    tot = sum(lp[i, y_ids[i]].item() for i in range(len(y_ids)))
    return tot / len(y_ids), len(y_ids)                  # 长度归一化
```

### 四个实现要点

| 要点 | 原因 |
| --- | --- |
| **teacher forcing**，不逐候选生成 | 10 个候选只需 1 次前向；且要的正是 `log P(y\|audio)` |
| **off-by-one** 取 logits | `logits[L-1]` 才是「预测候选第一个 token」的位置 |
| **长度归一化** | `log P` 是负数累加，不归一化会系统性偏好短句 |
| **走 `.thinker`** | 顶层 `Qwen3ASRForConditionalGeneration` 只是封装，`forward` 实现在 thinker |

### 为什么它有效而前三种不有效

```
n-gram / 神经 LM  打的是  P(y)      ← 语言维度
跨模型重排         打的是  P(y|audio) ← 声学维度
```

错误是**声学不确定性**（窄带下的近音混淆：`机主`/`机础`、`盘`/`弹`），
语言先验对它们没有区分度——`机础` 在中文里也挺通顺。

**互补性**：SenseVoice 判错的样本中，34.5% Qwen3-ASR 能答对。

### 实测

**CER 8.63% → 7.97%（−0.66pt）**

---

## 对照总表

| | 浅融合 | N-best 重排 | WFST | 跨模型重排 |
| --- | --- | --- | --- | --- |
| 打分/约束的量 | `P(y\|prefix)` | `P(y)` | 图路径 | **`P(y\|audio)`** |
| 维度 | 语言 | 语言 | 结构 | **声学** |
| 需要词典 | 否 | 否 | **是** | 否 |
| 工程成本 | 低 | 低 | 高 | 中 |
| 本域实测 | +0.02pt | +0.18 / +0.06pt | 全部变差 | **−0.66pt** |

**选择依据**：先诊断错误属于哪一维，再选对应维度的手段。
本域错误在声学维，所以只有跨模型重排有区分度。
