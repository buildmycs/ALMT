# MOSEI：训练融合权重 0.30 对照实验

本轮默认配置为 `configs/mosei_dual_c4_intensity.yaml`。相对上一轮 `2e-5`
实验，只将训练融合权重 `ordinal_prediction_weight` 从 `0.45` 调整为 `0.30`。
学习率为 `2e-5`、有序损失权重为 `0.2`，训练预算为 100 epoch，LR warmup 为
10 epoch，有序损失 warmup 为 5 epoch，seed 为 0；对比学习关闭，有序损失不衰减。

## 验证集依据

使用各自保存的 `best_validation_predictions.npz`，对同一 checkpoint 的两个头
重新计算指标和验证集 rho 搜索，不使用测试集决定参数。样本数均为 1,871。

| 学习率 | 最优 epoch | 回归头 Acc-7 | 有序头 Acc-7 | 原融合 Acc-7（rho=0.45） | 验证最优推理 rho | 搜索后 Acc-7 | 搜索后 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3e-5 | 45 | 53.0198% | 50.4543% | 54.8904% | 0.44 | 54.9973% | 0.510315 |
| 2e-5 | 79 | 53.8215% | 48.7974% | 54.1956% | 0.32 | 54.9973% | 0.503101 |

因此 `2e-5` 并未提升固定 rho 下的融合验证 Acc-7；但使用相同验证搜索规则后，
Acc-7 与 `3e-5` 持平，MAE 更低。不能仅凭测试集上 rho=0.2 的结果认定前者更好。

在当前 `2e-5` checkpoint 上，推理 rho=0.30/0.31/0.32/0.33 的验证 Acc-7
分别为 54.8904% / 54.9439% / 54.9973% / 54.8904%，均高于原融合的 54.1956%。
因此选择 0.30 做下一次训练对照，验证降低训练中有序预测的占比能否改善融合输出。
这是实验假设：训练时 rho 会改变 MSE 的梯度及 checkpoint 选择，离线融合结果
不等价于重新训练后的效果，也不能保证重训练会提升 Acc-7。

原理：`prediction = (1-rho) * regression_prediction + rho * ordinal_prediction`，
训练中的 MSE 监督这个融合输出。改为 rho=0.30 后，两个头从这项损失得到的直接
梯度系数由 0.55/0.45 变为 0.70/0.30；独立的有序辅助损失权重仍为 0.2。
共享特征仍会受到两种损失的影响，这些系数不代表最终总梯度的固定比例。

这次最优轮为 79，末 11 轮融合验证 Acc-7 平均约 54.01%，没有明显持续上升的
趋势。暂时采用相同的 100 epoch 调度做可比实验，不同时增加训练预算或继续降低 LR。

## 保存的基线

旧实验配置为 `configs/mosei_dual_c4_intensity_rho045_lr2e-5.yaml`，旧结果目录为：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr2e-5_seed0/
```

该配置用于核对或评估旧 checkpoint。若重新运行基线训练，应复制配置并换用独立的
`project_name`，避免覆盖已经存在的结果。

## 新实验运行命令

以下命令在训练服务器的 Bash 中运行；在同一终端保留 `RUN_NAME` 和 `RUN_DIR`。
本文件中的路径对应新实验，不要混用其他实验的 checkpoint 或预测文件。

```bash
RUN_NAME=ALMT_MOSEI_Dual_C4_Intensity_rho030_ord020_fixed_e100_wu10_lr2e-5_seed0
RUN_DIR="ckpt/$RUN_NAME"

python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --gpu_id 0

python scripts/bestweight.py \
  --predictions "$RUN_DIR/best_validation_predictions.npz" \
  --step 0.01 \
  --top-k 10
```

从本轮验证集搜索结果读取推理 rho，不能直接沿用旧实验的 0.2 或 0.32：

```bash
RHO=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["rho"])' "$RUN_DIR/rho_search_validation/rho_search_summary.json")

python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --checkpoint "$RUN_DIR/best_validation_model.pth" \
  --ordinal-prediction-weight "$RHO" \
  --output-dir "$RUN_DIR/rho_${RHO}_test" \
  --gpu_id 0
```

训练脚本会先输出训练 rho=0.30 对应的测试指标，上述脚本再应用验证集选定的
推理 rho 做固定评估。不要在测试集上遍历 rho 后将最高值当作未参与调参的成绩。

## 比较标准与后续

相对旧的 `2e-5/rho=0.45` 基线，记录两个头、按训练 rho 融合的验证 Acc-7、
验证搜索后的 Acc-7 与 MAE，以及后期曲线。主要比较相同验证搜索规则下的融合
结果，不能把新实验搜索后指标与旧实验未搜索指标混在一起比较。

如果新结果仅比 54.9973% 多几个验证样本，给旧基线和新实验各补 seed=1、2
并使用不同项目名，比较均值与波动。若没有稳定收益，保留旧配置，不继续盲目
降低 rho 或 LR。更大改动（例如对回归头单独增加监督）应另作一项对照实验。
