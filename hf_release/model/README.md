---
base_model: Qwen/Qwen3-8B
library_name: peft
tags:
- text-to-sql
- lora
- qwen3
---

# Qwen3-8B BIRD LoRA（DB-dev4）

这是 Qwen3-8B 的 PEFT LoRA adapter，不包含约 19 GB 的基座模型。加载前请先获得基座模型 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)。

## 已发布文件

- `adapter_model.safetensors`：标准 PEFT LoRA 权重，174,655,536 bytes。
- `adapter_config.json`：PEFT adapter 配置。
- `additional_config.json`：训练框架附加配置。
- `training_config.yaml`：本项目记录的正式训练配置。
- `SHA256SUMS`：adapter 和配置文件的 SHA-256。
标准 adapter 已通过 Hugging Face LFS 发布，SHA-256 为 `8940b165798f62e7d13b978cc43646077db2a56e03c504c445e1b1e914dd4464`。

## 已核验结果

最佳 checkpoint 为 `checkpoint-969`。该 adapter 使用本项目构造的结构化 Mini-Dev prompt 生成 SQL，并使用 BIRD Mini-Dev 作者的 SQLite 集合型 EX 评测：**50.40%（252/500）**。基座 Qwen3-8B 在同一自定义 prompt 与评测口径下为 41.80%（209/500）。

预测原始文件有 498 条；`question_id` 119 和 120 缺失。作者格式适配器会将这两题补为空 SQL 后，以完整 500 题为分母评测，因此上述分数不因缺失预测而虚高。对应预测、作者格式输出、评测日志和哈希见 [结果数据集](https://huggingface.co/datasets/craboy4/text-to-sql-struct-distillation-results)。

## 加载

按标准 PEFT adapter 加载：

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-8B"
adapter_id = "craboy4/qwen3-8b-bird-lora-dbdev4"
tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

训练配置、提示词构造和评测脚本见 GitHub 工程：<https://github.com/craboy4/text-to-sql-struct-distillation>。
