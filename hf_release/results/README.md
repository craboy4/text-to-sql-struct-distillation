---
license: cc-by-sa-4.0
tags:
- text-to-sql
- bird
- evaluation
---

# Qwen3-8B Text-to-SQL 评测结果

本仓库发布 Qwen3-8B 真实推理输出、作者 EX 评测输出和不可变清单。

## 已核验结论

在 BIRD Mini-Dev SQLite 的 500 题、作者集合型 EX 标准下：

| 模型 | 推理 prompt | EX | 正确数 |
| --- | --- | ---: | ---: |
| Qwen3-8B 基座 | 本项目构造的结构化 prompt | 41.80% | 209 / 500 |
| Qwen3-8B + LoRA checkpoint-969 | 本项目构造的结构化 prompt | 50.40% | 252 / 500 |

## 正式 checkpoint-969 运行

已发布的 `experiment_summaries/author_prompt_qwen3_8b_minidev_ex.json` 记录了完整 500 题的作者 prompt 对照实验：Qwen3-8B 基座为 49.40%（247/500），`checkpoint-969` 为 49.20%（246/500）。该文件只含评测汇总和运行元数据，不含预测 SQL、数据库或 gold SQL。

`runs/qwen3_8b_dbdev4_checkpoint-969/` 包含取得 50.40% 的真实运行产物：

- `predictions.jsonl`：自定义结构化 prompt 的 498 条 `question_id/db_id/sql` 预测。
- `author_predictions.json`：按 BIRD 作者格式适配后的 500 题预测；缺失的 119、120 两题填入空 SQL。
- `author_ex.txt`：作者 `evaluation_ex.py` 的 SQLite 集合型 EX 输出，50.40%（252/500）。
- `SHA256SUMS`：上述三个文件的 SHA-256。

作者的 prompt 文件在此运行中仅用于题目顺序和输出格式适配，**不参与模型推理**。模型推理使用的是本项目构造的结构化 prompt。BIRD SQLite 数据库、gold SQL 和每题的 gold 执行结果不在本仓库发布。

来源：BIRD Mini-Dev <https://github.com/bird-bench/mini_dev>，派生发布遵守 CC BY-SA 4.0。
