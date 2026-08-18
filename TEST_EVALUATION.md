# Validation 选模后的 Test 评测流程

## 1. 正确的“最佳 Test 结果”定义

本项目使用 validation Acc-7 选择唯一 checkpoint。对于当前实验，选择依据是
validation Acc-7 为 51.1% 的 epoch。参数和模型固定后，只在 test split 上运行一次，
得到的全部指标构成这套配置的最终 test 结果。

不能逐 epoch 评估 test 后挑选最高 Acc-7，也不能在多组超参数的 test 结果中挑最大值。
这两种做法都使用了 test 信息进行模型选择，会造成 test leakage，使论文结果偏高且无法
公平复现。

流程为：

```text
train epochs
    -> 用 validation Acc-7 选择 best_validation_model.pth
    -> 冻结参数与 checkpoint
    -> 在 test 上评估一次
    -> 报告这一轮的全部 test 指标
```

## 2. 51.1% 对应配置

`configs/mosi_dual_c4_intensity.yaml` 已固定为：

```yaml
model:
  ordinal_prediction_weight: 0.4

objective:
  regression_weight: 1.0
  ordinal_weight: 0.3
  contrastive_weight: 0.0
  contrastive_temperature: 0.1
  contrastive_label_temperature: 0.5
  auxiliary_warmup_epochs: 10
  max_balance_weight: 5.0
```

`ordinal_prediction_weight` 不在 checkpoint 的 `state_dict` 中，评测时必须保持为 0.4；
否则即使加载同一个模型参数，也会得到不同的融合预测。

## 3. 已经完成训练时直接评测 Test

默认读取配置中项目目录下的 `best_validation_model.pth`：

```bash
python scripts/evaluate_selected_test.py \
  --config_file configs/mosi_dual_c4_intensity.yaml \
  --gpu_id 0
```

也可以明确指定 51.1% 对应的 checkpoint，避免项目名或路径不一致：

```bash
python scripts/evaluate_selected_test.py \
  --config_file configs/mosi_dual_c4_intensity.yaml \
  --checkpoint ckpt/ALMT_MOSI_Dual_C4_Intensity/best_validation_model.pth \
  --gpu_id 0
```

脚本只接受一个 checkpoint，不遍历 epoch，也不根据 test 指标重新选模。启动时会输出
checkpoint 内保存的 validation 选择 epoch、选择指标及数值，请确认其中的
`Mult_acc_7` 约为 `0.511`。

## 4. 输出文件

默认写入 checkpoint 所在目录：

```text
selected_test_predictions.npz
selected_test_predictions.csv
selected_test_results.json
```

- NPZ：融合预测、标签、epoch，以及回归头和有序头的预测；
- CSV：逐样本 ID、标签、预测、原文和增强文本；
- JSON：validation 选模信息、test 样本数和最终 test 全部指标。

终端的 `Test Results` 与 `selected_test_results.json` 中的 `test_results` 是论文应记录的
同一组结果，不能把不同 epoch 各指标的最大值拼成一行。

## 5. Test Acc-7 混淆矩阵

分析最终融合预测：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/selected_test_predictions.npz \
  --title "MOSI test - validation-selected checkpoint" \
  --output-dir ckpt/ALMT_MOSI_Dual_C4_Intensity/acc7_selected_test
```

单独分析回归头与有序头：

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/selected_test_predictions.npz \
  --prediction-key regression_predictions \
  --output-dir ckpt/ALMT_MOSI_Dual_C4_Intensity/acc7_selected_test_regression

python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSI_Dual_C4_Intensity/selected_test_predictions.npz \
  --prediction-key ordinal_predictions \
  --output-dir ckpt/ALMT_MOSI_Dual_C4_Intensity/acc7_selected_test_ordinal
```

## 6. 多随机种子论文报告

如果运行多个随机种子，每个种子都应独立使用 validation Acc-7 选择 checkpoint，再各自
评估一次 test。论文报告这些 test 结果的均值和标准差，而不是报告其中最大的 test 值。
超参数搜索结束后不得再根据 test 结果返回修改权重。
