# 结构化 Text-to-SQL 蒸馏

本项目使用 BIRD 训练集和 Mini-Dev SQLite 基准，完成从教师样本构造、SFT 训练、推理到执行准确率评测的完整流程。工程组织参考 [Struct-SQL-Distillation](https://github.com/craterlabs/Struct-SQL-Distillation) 的“数据构造 - 训练 - 推理 - 结果”边界，并使用 [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) 作者提供的 SQLite EX 评测定义。

## 当前结果

评测使用 BIRD Mini-Dev SQLite 的 500 个样本，作者的集合型 Execution Accuracy (EX) 标准。当前远端推理产物中，最新完整微调 checkpoint 的结果如下：

| 模型 | 训练状态 | EX | 正确数 |
| --- | --- | ---: | ---: |
| Qwen3-8B | 初始模型 | 41.80% | 209 / 500 |
| Qwen3-8B + LoRA checkpoint-969 | 3 epoch SFT | 50.40% | 252 / 500 |

评测的详细口径见 [docs/评测说明.md](docs/评测说明.md)，每项资产的位置和版本见 [docs/资产清单.md](docs/资产清单.md)。

## Hugging Face 发布

- [通用 SFT 数据集（6,246 条）](https://huggingface.co/datasets/craboy4/text-to-sql-struct-distillation-sft)
- [Mini-Dev 派生 SFT 数据集（500 条）](https://huggingface.co/datasets/craboy4/text-to-sql-struct-distillation-minidev)

正式 DB-dev4 SFT、Qwen3-8B LoRA `checkpoint-969`、真实预测和评测输出会在远端资产核验后分别发布；仓库链接和不可变版本会同步登记到资产清单。

## 目录

```text
BIRD/
├── teacher_generation/       # 教师调用、SQL 执行校验、SFT 样本导出
├── training/                 # SFT、推理、提示词构造和本地评测
├── evaluation/author_minidev/# BIRD Mini-Dev 作者 EX 脚本副本
├── configs/sft/              # 可复现训练配置
├── scripts/                  # 远端运行和格式适配脚本
├── docs/                     # 中文文档、资产和评测记录
├── train*/                   # 原始/过滤训练集，不纳入 Git
├── mini_dev/                 # BIRD Mini-Dev 数据库，不纳入 Git
└── author_mini_dev/          # 上游代码快照，不纳入 Git
```

Git 只跟踪代码、配置、文档和小型结果元数据。原始数据、SFT JSONL、推理响应、SQLite 数据库、模型权重与检查点由远端算力平台或后续数据/模型仓库保存，路径和哈希记录在资产清单中。

## 工作流

1. **构造样本**：从 `teacher_generation/generate_teacher_data.py` 生成带 schema、evidence 与执行校验的教师样本；使用 `export_qwen_sft.py` 导出 SFT messages。
2. **准备训练集**：按数据库划分训练集，当前远端已使用 `qwen_sft_messages_train_dbdev4.jsonl`，共 5,160 条样本。
3. **训练**：远端执行 `training/start_sft_3epoch_dbdev.sh`。该实验使用 Qwen3-8B、LoRA rank 16、learning rate `5e-5`、3 epoch、最大长度 32,768。
4. **推理**：使用 `training/run_minidev_local.py` 或算力平台既有推理流程，生成 `question_id/sql` JSONL 预测。
5. **评测**：使用 `scripts/evaluate_author_minidev.sh` 适配预测格式并调用作者 `evaluation_ex.py`。

## 快速评测

远端环境中，准备好 `PROJECT_ROOT`、Mini-Dev SQLite 数据库与作者的 `mini_dev_prompt.jsonl` 后：

```bash
export PROJECT_ROOT=/root/autodl-tmp/text2sql_qwen3
bash scripts/evaluate_author_minidev.sh \
  "$PROJECT_ROOT/outputs/<run>/predictions.jsonl" \
  "$PROJECT_ROOT/evaluation_results/<run>"
```

脚本按作者要求以 500 题为分母；若输入预测缺少某题，则该题以空 SQL 计错，避免因缺失预测虚高结果。

## 复现边界

- BIRD Mini-Dev 的数据库和 gold SQL 需要从上游数据包获取，不能用本仓库的 Git 内容替代。
- Qwen3-8B 基座约 19 GB，LoRA checkpoint 约 501 MB，均不进入普通 Git；发布时应分别上传到模型仓库并在资产清单登记版本与哈希。
- 作者 EX 使用集合比较，忽略重复行；它与多重集比较的分数不可直接混用。
