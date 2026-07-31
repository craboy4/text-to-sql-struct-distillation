---
base_model: Qwen/Qwen3-8B
library_name: peft
tags:
- text-to-sql
- lora
- qwen3
---

# Qwen3-8B BIRD LoRA（DB-dev4）

该模型仓库用于发布 Qwen3-8B 的 LoRA adapter，不包含约 19 GB 的基座模型。基座模型为 `Qwen/Qwen3-8B`。

## 已核验结果

正式训练的最佳 checkpoint 是 `checkpoint-969`。在 BIRD Mini-Dev SQLite 500 题的作者集合型 EX 评测中，得分为 **50.40%（252/500）**；Qwen3-8B 基座为 41.80%（209/500）。

## 发布状态

`checkpoint-969` LoRA adapter（约 501 MB）、训练 manifest 与 SHA-256 目前只在远端训练机。等待取得远端访问后将原样上传。本卡不应被解读为模型权重已经在本仓库就绪。

训练配置与评测脚本见：<https://github.com/craboy4/text-to-sql-struct-distillation>。
