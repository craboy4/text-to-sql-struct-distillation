---
license: cc-by-sa-4.0
language:
- zh
task_categories:
- text-generation
tags:
- text-to-sql
- sft
- bird
---

# 结构化 Text-to-SQL 蒸馏 SFT（本地通用导出）

这是本项目完整的结构化 Text-to-SQL SFT 导出，共 6,246 条 messages 格式 JSONL 样本；其中正式 Qwen3-8B 实验实际使用的是本仓库中明确标注的 5,160 条训练 split。

## 文件与划分

- `qwen_sft_messages.jsonl`：全量 SFT，6,246 条。
- `splits/dbdev4_train_5160.jsonl`：正式 Qwen3-8B LoRA 训练集，5,160 条。它是全量教师样本按数据库划分后的子集，对应训练机中的 `qwen_sft_messages_train_dbdev4.jsonl`。
- `splits/dbdev4_execution_dev_prompts_400.jsonl`：训练期 database-disjoint execution-dev 的 400 条提示词，只含 system/user messages，不含 gold SQL。
- `splits/dbdev4_split_manifest.json`：划分参数、样本数和 SHA-256。

划分从同一个 6,246 条教师样本源构造，随机种子为 `42`：

| 用途 | 数据库与规则 | 条数 |
| --- | --- | ---: |
| 正式训练 | 排除下方所有保留/剔除数据库后的样本 | 5,160 |
| execution-dev | `hockey`、`mondial_geo`、`movie_3`、`student_loan` 各按 `SHA-256("42:record_id")` 排序选取前 100 条 | 400 |
| 未参与训练的同库余量 | 上述四库未被抽入 execution-dev 的其余样本 | 318 |
| 长上下文剔除 | `works_cycles` 全部样本 | 368 |

四个 execution-dev 数据库总计 718 条，因此训练集与 execution-dev 在数据库层面完全不重叠。已完成的 8B 训练命令使用全部 5,160 条训练样本并显式设置 `split_dataset_ratio=0`；400 条 execution-dev 是独立的运行级开发集，不是训练器内部的 validation-loss split。

- 不包含 BIRD SQLite 数据库、gold SQL 或任何原始数据库内容。

## 样本格式

每行是 Hugging Face / Qwen 兼容的 messages JSON 对象，包含系统提示、用户问题及目标助手回复。样本由工程中的教师样本构造和导出脚本产生，相关代码见 GitHub 仓库：<https://github.com/craboy4/text-to-sql-struct-distillation>。

## 来源与许可

本数据包含基于 BIRD 任务构造的派生内容。BIRD Mini-Dev 上游材料采用 CC BY-SA 4.0；本仓库以 CC BY-SA 4.0 发布。使用者须保留署名、引用上游 BIRD 工作，并以兼容条款发布派生成果。

## 引用

```bibtex
@inproceedings{li2024bird,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author={Li, Jinyang and others},
  year={2024}
}
```
