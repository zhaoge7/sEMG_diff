# Ninapro DB2 跨个体实验实现进度

更新日期：2026-07-06

## 已完成

- 建立项目结构：`configs/`、`src/semg_diff/`、`scripts/`、`data/processed/`、`results/`、`docs/`。
- 添加 `requirements.txt` 和 `pyproject.toml`。
- 添加默认配置 `configs/default.yaml`，包含数据路径、窗口参数、模型参数、稳定学习参数和随机种子。
- 实现 DB2 文件发现、字段读取、完整性检查和统计表生成。
- 实现 EMG 带通滤波、工频陷波、可选整流、滑动窗口切片和 `.npz` 缓存。
- 窗口元信息包含 subject、exercise、repetition、label、start/end、env_id、class_id、sample_weight。
- 实现 LOSO 和个体内 repetition 划分工具。
- 实现传统时域特征：MAV、RMS、WL、ZC、SSC、VAR、WAMP。
- 实现传统模型入口：LDA、Linear SVM、RBF SVM、Random Forest。
- 实现类别条件受试者均衡样本权重，用于稳定重加权传统 baseline。
- 实现 PyTorch `EMGWindowDataset`、1D-CNN、early stopping、checkpoint 和训练日志。
- 实现 Stable-CNN 风险方差目标：`mean_s(Loss_s) + lambda * var_s(Loss_s)`。
- 实现 CORAL 无监督目标域特征对齐 baseline。
- 实现 Accuracy、Balanced Accuracy、Macro-F1、per-class F1、混淆矩阵 CSV、跨受试者均值/标准差/最差/最差 25% 指标。
- 实现结果图脚本：每名目标受试者柱状图、方法箱线图和混淆矩阵热图。

## 已验证

- `python -m compileall /workspace/sEMG_diff/src /workspace/sEMG_diff/scripts` 通过。
- `scripts/check_db2_integrity.py` 检查 E1 的 40 个受试者文件，结果为 `All files OK: True`。
- `scripts/build_db2_windows.py --subjects 1 2 3 --exercises 1` 生成 S1-S3 E1 冒烟缓存：`(12011, 400, 12)`，17 个类别。
- `scripts/run_traditional_baseline.py` 在 S1-S3 冒烟缓存上跑通 LDA 和 Linear SVM LOSO。
- `scripts/run_traditional_baseline.py --stable-weights` 在 S1-S3 冒烟缓存上跑通稳定重加权 LDA。
- `scripts/run_cnn_baseline.py --target-subjects 1 --epochs 1` 跑通 CNN 训练、checkpoint 和测试输出。
- `scripts/run_cnn_baseline.py --stable-risk --target-subjects 1 --epochs 1` 跑通 Stable-CNN 分支。
- `scripts/run_coral_baseline.py --classifier lda` 跑通 CORAL 对比。
- `scripts/plot_results.py` 跑通指标图和混淆矩阵热图生成。

## 待完成

- 全量 E1 的 40-fold LOSO 结果尚未运行。
- E2/E3 与 DB2 全手势扩展尚未运行。
- 少样本个体自适应实验尚未实现完整脚本。
- MMD、DANN、对比学习等额外跨个体方法尚未实现。
- 窗口长度、归一化策略、特征组合、模型结构和稳定系数消融尚未全量运行。
- 论文相关章节尚未撰写。
- 文献调研仍需补充正式引用和笔记。
