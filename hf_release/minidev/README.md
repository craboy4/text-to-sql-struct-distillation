---
license: cc-by-sa-4.0
language:
- zh
task_categories:
- text-generation
tags:
- text-to-sql
- bird
- mini-dev
- sft
---

# 结构化 Text-to-SQL 蒸馏 Mini-Dev 派生 SFT

本仓库发布由 BIRD Mini-Dev 500 个样本构造的派生 SFT messages 数据，共 500 条。

## 文件与边界

- `minidev_sft_messages.jsonl`：500 条 messages 格式的派生 SFT 样本。
- 不含 Mini-Dev SQLite 数据库、gold SQL、原始 schema 文件或上游数据库内容。
- 评测中使用 BIRD 作者提供的 SQLite 集合型 Execution Accuracy (EX) 定义；该评测代码与复现说明在 GitHub 工程中维护。

## 来源、署名与许可证

本数据为 BIRD Mini-Dev 的派生文本内容。上游仓库：<https://github.com/bird-bench/mini_dev>。上游 README 标明 CC BY-SA 4.0，因此本仓库按 CC BY-SA 4.0 发布。使用或再发布时请保留对 BIRD 与 Mini-Dev 的署名，并遵守相同方式共享要求。

## 引用

```bibtex
@inproceedings{li2024bird,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author={Li, Jinyang and others},
  year={2024}
}
```
