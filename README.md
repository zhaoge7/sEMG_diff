# sEMG_diff

Ninapro DB2 跨个体 sEMG 手势识别实验代码。当前已落地一个可复现实验管线，覆盖任务清单中的最小可完成实验版本：

- DB2 `.mat` 读取、完整性检查和统计表生成。
- 带通/陷波预处理、滑动窗口切片和窗口缓存。
- 时域特征：MAV、RMS、WL、ZC、SSC、VAR、WAMP。
- 个体内与 LOSO 跨个体传统机器学习 baseline：LDA、SVM、Random Forest。
- 类别条件受试者均衡样本权重 baseline。
- 1D-CNN 与 Stable-CNN，Stable-CNN 使用 `mean subject loss + lambda * loss variance`。
- CORAL 无监督目标域特征对齐 baseline。
- Accuracy、Balanced Accuracy、Macro-F1、per-class F1、混淆矩阵和跨受试者稳定性指标。

## 数据

默认配置读取：

```text
/workspace/Ninapro/db2/DB2_s{subject}/S{subject}_E{exercise}_A1.mat
```

原始数据只读，处理后的缓存写入 `data/processed/db2/`，结果写入 `results/`。

## 环境

```bash
pip install -e .
```

或：

```bash
pip install -r requirements.txt
export PYTHONPATH=/workspace/sEMG_diff/src
```

## 推荐运行顺序

检查 E1 文件完整性并生成统计表：

```bash
python scripts/check_db2_integrity.py --config configs/default.yaml
```

先用少量受试者构建窗口缓存做冒烟测试：

```bash
python scripts/build_db2_windows.py --config configs/default.yaml --subjects 1 2 3 --exercises 1
```

全量 E1 窗口缓存：

```bash
python scripts/build_db2_windows.py --config configs/default.yaml --exercises 1
```

传统 LOSO baseline：

```bash
python scripts/run_traditional_baseline.py \
  --config configs/default.yaml \
  --windows data/processed/db2/db2_E1_S1-2-3_all_200ms.npz
```

稳定样本重加权传统 baseline：

```bash
python scripts/run_traditional_baseline.py \
  --config configs/default.yaml \
  --windows data/processed/db2/db2_E1_S1-2-3_all_200ms.npz \
  --stable-weights
```

1D-CNN 或 Stable-CNN：

```bash
python scripts/run_cnn_baseline.py \
  --config configs/default.yaml \
  --windows data/processed/db2/db2_E1_S1-2-3_all_200ms.npz \
  --target-subjects 1 2 \
  --epochs 3

python scripts/run_cnn_baseline.py \
  --config configs/default.yaml \
  --windows data/processed/db2/db2_E1_S1-2-3_all_200ms.npz \
  --stable-risk
```

CORAL 对比：

```bash
python scripts/run_coral_baseline.py \
  --config configs/default.yaml \
  --windows data/processed/db2/db2_E1_S1-2-3_all_200ms.npz \
  --classifier linear_svm
```

生成图表：

```bash
python scripts/plot_results.py \
  --fold-results results/loso_traditional/fold_results.csv results/loso_traditional_stable_weights/fold_results.csv \
  --out-dir results/figures
```

## 协议约束

- LOSO 中目标受试者只用于测试。
- 标准化参数只在源域训练窗口上拟合，再应用到验证和测试受试者。
- 稳定样本权重只用源域受试者标签和环境 ID 估计。
- Stable-CNN 的风险方差只在源域训练受试者上计算。
- CORAL 脚本使用未标注目标受试者特征，输出中明确标注为无监督域适应，不应与零校准 source-only 直接混为同一设置。
