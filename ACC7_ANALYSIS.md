# Acc-7 混淆矩阵与分等级 Recall

`train_dual.py` 会在 validation MAE 创新低时保存：

- `best_validation_model.pth`：最佳 validation 轮模型；
- `best_test_predictions.npz`：测试集连续预测与标签；
- `best_test_predictions.csv`：包含 ID、原文和增强文本的可读结果。

默认目录为：

```text
ckpt/<project_name>/
```

训练完成后运行：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4/best_test_predictions.npz \
  --title "MOSI Dual-C4"
```

也可以直接分析 CSV：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4/best_test_predictions.csv
```

输出目录包含：

- `acc7_confusion_matrix.png`：计数矩阵和行归一化百分比矩阵；
- `confusion_counts.csv`：每个真实等级到预测等级的样本数；
- `confusion_row_normalized.csv`：每行归一化后的矩阵；
- `per_class_metrics.csv`：七个等级各自的 recall、precision、F1、support；
- `sample_diagnostics.csv`：每条样本的连续误差和离散等级误差；
- `summary.json`：总体 Acc-7、macro recall、macro F1、MAE 和 Corr。

脚本严格复用 ALMT 的 Acc-7 定义：

```python
np.round(np.clip(value, -3.0, 3.0))
```

如果环境没有 Matplotlib，CSV 和 JSON 仍会正常生成，但 PNG 会跳过。安装方式：

```bash
pip install matplotlib
```

旧版 `train_dual.py` 没有保存逐样本预测，因此已经结束且没有 checkpoint/NPZ 的训练
无法仅从终端日志恢复混淆矩阵，需要使用更新后的训练入口重新运行一次。
