# 复现原始 ALMT 的 Test 选择与报告方式

## 1. 使用目的

原始 `train.py` 在每个 epoch 后同时评测 validation 和 test，并通过
`core.utils.results_recorder` 输出：

- `Best Test Results across All Epochs`：每个指标分别取其跨 epoch 最优值；
- `Best Test Results of One Epoch`：test MAE 最低轮的完整指标。

为了与原始 GitHub ALMT 的终端结果采用相同口径，`train_dual.py` 新增可配置的
`legacy_test_oracle` 模式。MOSI intensity 配置已经启用：

```yaml
base:
  evaluation_protocol: legacy_test_oracle
```

该模式用于复现和同口径比较。它会在每个 epoch 查看 test，因此属于 test-oracle
报告方式；不要把它描述成完全独立的 held-out test 估计。

## 2. 运行

```bash
python train_dual.py \
  --config_file configs/mosi_dual_c4_intensity.yaml \
  --gpu_id 0
```

每个 epoch 会输出：

```text
Training Results
Validation Results
Test Results
Best Validation Results across All Epochs
Best Validation Results of One Epoch
Best Test Results across All Epochs
Best Test Results of One Epoch
Best Test Acc-7 Epoch
```

前四类历史汇总严格复用原版 `results_recorder`。此外，新代码会记录 test Acc-7
最高的具体 epoch，并保存该轮完整结果、模型和逐样本预测，避免只得到一个无法还原
到 checkpoint 的最大数值。

## 3. 输出文件

默认位于：

```text
ckpt/ALMT_MOSI_Dual_C4_Intensity/
```

新增文件：

```text
best_test_acc7_oracle_model.pth
best_test_acc7_oracle_predictions.npz
best_test_acc7_oracle_predictions.csv
legacy_test_oracle_summary.json
```

`legacy_test_oracle_summary.json` 包含：

- test Acc-7 最高的 epoch 及该轮全部指标；
- 原始逻辑的 test MAE 最低单轮结果；
- 原始逻辑的各指标跨 epoch 最优汇总；
- validation 的对应两类汇总；
- 同时保存的 validation-selected checkpoint 信息。

注意，`Best Test Results across All Epochs` 中的 Acc-7、MAE、Corr 等可能来自不同
epoch，不代表一个真实 checkpoint。论文表格如果要求“一行对应一个模型”，建议使用
`best_test_acc7_epoch_results`；如果要严格复刻原始 ALMT 的输出表，则使用
`best_test_results_across_all_epochs`，并在实验设置中注明该口径。

## 4. 保留 Validation-only 模式

其他配置没有填写 `evaluation_protocol` 时，默认仍采用：

```yaml
base:
  evaluation_protocol: validation_selected
```

该模式使用 validation 指标选择 checkpoint，训练结束后只评测一次 test。两套模式
共存，便于同时提供原始 ALMT 同口径结果和更严格的 validation-selected 补充结果。
