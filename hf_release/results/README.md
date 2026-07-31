---
license: cc-by-sa-4.0
tags:
- text-to-sql
- bird
- evaluation
---

# Qwen3-8B Text-to-SQL 评测结果

本仓库用于发布真实推理输出、作者 EX 评测输出和不可变清单。

## 已核验结论

在 BIRD Mini-Dev SQLite 的 500 题、作者集合型 EX 标准下：

| 模型 | EX | 正确数 |
| --- | ---: | ---: |
| Qwen3-8B 基座 | 41.80% | 209 / 500 |
| Qwen3-8B + LoRA checkpoint-969 | 50.40% | 252 / 500 |

## 发布状态

已发布的 `experiment_summaries/author_prompt_qwen3_8b_minidev_ex.json` 记录了完整 500 题的作者 prompt 对照实验：Qwen3-8B 基座为 49.40%（247/500），`checkpoint-969` 为 49.20%（246/500）。该文件只含评测汇总和运行元数据，不含预测 SQL、数据库或 gold SQL。

取得 Mini-Dev 50.40%（252/500）的正式 `predictions.jsonl`、对应作者 EX 输出和 SHA-256 清单目前仍仅位于远端训练机。它们会在取得远端产物后原样上传；在此之前本仓库不以任何本地 Mini-Dev 教师输出冒充 8B 微调结果。

来源：BIRD Mini-Dev <https://github.com/bird-bench/mini_dev>，派生发布遵守 CC BY-SA 4.0。
