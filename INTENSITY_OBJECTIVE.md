# 回归 + 有序分类 + 情感强度对比学习

## 1. 目标

混淆矩阵显示 `-3` 大量被预测为 `-2`，`+3` 大量被预测为 `+2`，说明模型存在
向中心强度收缩的问题。本模块保留连续回归能力，同时直接监督七个有序情感等级，
并约束表示空间反映连续情感距离。

完整损失为：

```text
L = Lreg + lambda_ord * Lord + lambda_con * Lcon
```

前五轮对两个辅助损失进行线性 warm-up，降低训练初期随机有序头和对比投影对已
有效的 ALMT 回归主干造成的扰动。

## 2. 连续回归

最终连续预测由回归值和有序分布期望融合：

```text
y_final = (1-rho) * y_reg + rho * y_ord
```

默认 `rho=0.2`。`Lreg` 仍然是 `MSE(y_final, y)`，所以 MAE、Corr 和二分类所依赖
的连续预测没有被替换。

## 3. 单调有序分类

七个等级 `[-3,-2,-1,0,1,2,3]` 被转换成六个累计判断：

```text
P(class > -3), P(class > -2), ..., P(class > +2)
```

模型学习一个样本强度分数和六个阈值。相邻阈值的距离通过 `softplus` 参数化，
因此训练过程中始终保持严格单调，不会出现“超过 +2 的概率反而高于超过 +1”这类
无效有序分布。

有序损失使用训练集标签统计进行阈值级正负平衡。第一个阈值的少数负例主要对应
`-3`，最后一个阈值的少数正例主要对应 `+3`，因此这两个极端等级会获得更充分的
梯度。权重仅由 train split 计算，不读取 validation/test 分布。

## 4. 连续强度对比学习

对最终多模态特征增加仅在训练时使用的投影头。对每个样本 `i`，其余样本的目标
相似度为：

```text
q(i,j) proportional to exp(-|y_i-y_j| / tau_y)
```

投影特征的相似度分布通过交叉熵拟合该目标分布。因此标签距离接近的样本会被拉近，
强度差异大的样本会被相对推远。对比锚点还使用七等级的逆平方根频次权重，降低
中性/弱情感多数样本对表示空间的支配。

## 5. 运行

MOSI：

```bash
python train_dual.py \
  --config_file configs/mosi_dual_c4_intensity.yaml \
  --gpu_id 0
```

MOSEI：

```bash
python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --gpu_id 0
```

原来的 `configs/mosi_dual_c4.yaml` 没有启用强度模块，仍可作为 Dual-Gate 基线。

最佳 validation MAE 轮会额外保存：

- `prediction`：用于正式指标的融合连续预测；
- `regression_prediction`：纯回归头预测；
- `ordinal_prediction`：六个累计概率得到的有序期望值。

可以继续生成混淆矩阵：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/best_test_predictions.npz \
  --title "MOSI Dual-C4 + Intensity Objective"
```

还可以分别分析纯回归头与有序头，判断极端等级的改善来自哪里：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/best_test_predictions.npz \
  --prediction-key regression_predictions \
  --output-dir ckpt/ALMT_MOSI_Dual_C4_Intensity/acc7_regression

python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/best_test_predictions.npz \
  --prediction-key ordinal_predictions \
  --output-dir ckpt/ALMT_MOSI_Dual_C4_Intensity/acc7_ordinal
```

## 6. 建议消融

只需要复制 intensity 配置并修改以下参数：

### 回归基线

```yaml
model:
  ordinal_prediction_weight: 0.0
objective:
  ordinal_weight: 0.0
  contrastive_weight: 0.0
```

### 回归 + 有序分类

```yaml
model:
  ordinal_prediction_weight: 0.2
objective:
  ordinal_weight: 0.2
  contrastive_weight: 0.0
```

### 回归 + 强度对比

```yaml
model:
  ordinal_prediction_weight: 0.0
objective:
  ordinal_weight: 0.0
  contrastive_weight: 0.05
```

### 完整模型

```yaml
model:
  ordinal_prediction_weight: 0.2
objective:
  ordinal_weight: 0.2
  contrastive_weight: 0.05
```

第一轮不要立即提高辅助损失权重。先观察 `-3/+3 recall`、整体 Acc-7、MAE 和 Corr；
如果极端 recall 上升但 MAE 明显变差，优先把 `ordinal_prediction_weight` 从 0.2
降到 0.1，而不是继续增加类别平衡权重。

## 7. 最佳 epoch 的选择标准

强度配置默认使用 validation Acc-7 选择 checkpoint，并在 Acc-7 并列时选择
validation MAE 更小的 epoch：

```yaml
base:
  selection_metric: Mult_acc_7
  selection_mode: max
  selection_secondary_metric: MAE
  selection_secondary_mode: min
```

训练阶段不会逐 epoch 评估或比较 test 指标。全部 epoch 结束后，程序重新加载完全
由 validation 指标选中的 `best_validation_model.pth`，只运行一次 test，并保存：

```text
best_validation_model.pth
best_validation_selection.json
best_test_predictions.npz
best_test_predictions.csv
```

论文中应报告 `best_validation_selection.json` 里的 `test_results`，并说明 checkpoint
selection criterion 是 validation Acc-7。不要从多个 epoch 的 test Acc-7 中挑最大值。

仓库内的 Dual-C4 与 Dual-C4-Intensity 配置均使用相同的 Acc-7 选择标准，以保证
消融公平。其他没有填写 `selection_metric` 的旧配置或自定义配置保持向后兼容，默认
按照 validation MAE 最小选择 checkpoint。
