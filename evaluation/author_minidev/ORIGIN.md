# 来源

本目录中的 `evaluation_ex.py` 与 `evaluation_utils.py` 从 `bird-bench/mini_dev` 提交 `b3d4bcbbae9a96934ad812551eb400c7a3b23c12` 原样复制。

本项目仅增加 `training/prepare_author_ex_predictions.py` 以适配本地 `question_id/sql` JSONL 预测格式；该适配器不改写 SQL。上游评测逻辑未经修改。
