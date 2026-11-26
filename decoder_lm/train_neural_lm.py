"""训练领域神经语言模型（GPT2 架构），用于 CTC n-best rescoring。"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_here, "config.py")):
    _here = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "common"))
from config import *  # noqa: F401,F403
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer

BPE = BPE_MODEL
OUT = NLM_DIR
VOCAB = 25055
PAD = 0

tok = SentencepiecesTokenizer(bpemodel=BPE, unk_symbol="<unk>", split_with_space=True)

seqs = []
for l in open(TRAIN_JSONL, encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    try:
        ids = [x for sub in tok.tokens2ids(tok.text2tokens(r["target"])) for x in sub]
    except Exception:
        continue
    if ids:
        seqs.append(ids)
print(f"序列 {len(seqs)}, token {sum(len(s) for s in seqs)}")

ds = Dataset.from_list([{"input_ids": s} for s in seqs])


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ids, attn, lab = [], [], []
    for b in batch:
        x = b["input_ids"]
        ids.append(x + [PAD] * (m - len(x)))
        attn.append([1] * len(x) + [0] * (m - len(x)))
        lab.append(x + [-100] * (m - len(x)))
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(lab)}


cfg = GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=512, n_layer=8, n_head=8,
                 bos_token_id=0, eos_token_id=0)
model = GPT2LMHeadModel(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"LM 参数 {n/1e6:.1f}M")

args = TrainingArguments(
    output_dir=OUT, overwrite_output_dir=True,
    per_device_train_batch_size=64, num_train_epochs=8,
    learning_rate=3e-4, warmup_steps=200, logging_steps=100,
    save_strategy="no", bf16=torch.cuda.is_bf16_supported(),
    report_to="none", dataloader_num_workers=4,
)
tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
tr.train()
model.save_pretrained(OUT)
print("saved", OUT)
