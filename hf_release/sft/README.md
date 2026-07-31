---
license: cc-by-sa-4.0
language:
- zh
task_categories:
- text2text-generation
tags:
- text-to-sql
- sft
- bird
---

# 结构化 Text-to-SQL 蒸馏 SFT（本地通用导出）

这是本项目当前本地可核验的通用 SFT 导出，共 6,246 条 messages 格式 JSONL 样本。

## 重要版本说明

- 本版本的文件为 `qwen_sft_messages.jsonl`，用于保存本地通用导出，**不是**取得 Mini-Dev 50.40% EX 的正式 DB-dev4 训练版本。
- 正式训练版本为远端的 `qwen_sft_messages_train_dbdev4.jsonl`（5,160 条）；待获得远端产物后将作为明确命名的新版本发布。
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
