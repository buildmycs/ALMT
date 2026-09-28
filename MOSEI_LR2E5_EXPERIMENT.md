# MOSEI：2e-5 学习率对照实验

本轮使用 `configs/mosei_dual_c4_intensity.yaml`，只将峰值学习率从 `3e-5`
降到 `2e-5`。训练融合权重为 `0.45`，有序损失权重为 `0.2`，训练预算为
100 epoch，学习率 warmup 为 10 epoch，有序损失 warmup 为 5 epoch，seed 为 0。
对比学习权重为 0，有序损失不衰减。采用独立实验目录保留上一轮结果。

## 选择依据

以下均为按验证集融合 Acc-7 选中的 checkpoint，推理 rho 固定为训练值 `0.45`：

| 学习率 | 最优 epoch | 回归头 Acc-7 | 有序头 Acc-7 | 融合 Acc-7 | 融合 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5e-5 | 77 | 46.9268% | 51.8974% | 54.6232% | 0.509349 |
| 3e-5 | 45 | 53.0198% | 50.4543% | 54.8904% | 0.510752 |

`3e-5` 的融合 Acc-7 提升了约 0.27 个百分点，即在 1,871 条验证样本上多答对
5 条；MAE 略有上升，因此不能视为所有指标都显著改善。其后 90–100 epoch 的
平均验证 Acc-7 约为 53.95%，未显示需要进一步延长训练的持续上升趋势。

这支持再做一次 `2e-5` 的单变量实验，但不保证继续提升。`3e-5` 的验证集搜索
最优推理 rho 为 `0.44`，Acc-7 为 54.9973%；rho 为 `0.2` 时验证 Acc-7 为
53.9818%。不能根据测试集上 rho=0.2 的较好表现，把下一轮训练融合权重改为 0.2。

`3e-5` 的原始依据位于：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr3e-5_seed0/
```

`5e-5` 的结果可在 Git 提交 `0a5371a` 对应的 checkpoint 目录中查看；两次的
逐轮 TensorBoard 日志均保存在各自实验名称的 `log/` 目录中。

## 运行命令

以下命令用于训练服务器的 Bash，按顺序在同一终端运行。新实验使用本文件中的
目录；`MOSEI_WORKFLOW.md` 中已有的其他实验命令不应直接用于本轮结果分析。

```bash
RUN_NAME=ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr2e-5_seed0
RUN_DIR="ckpt/$RUN_NAME"

python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --gpu_id 0
```

训练结束后，在该 checkpoint 的验证集预测上搜索推理 rho：

```bash
python scripts/bestweight.py \
  --predictions "$RUN_DIR/best_validation_predictions.npz" \
  --step 0.01 \
  --top-k 10
```

下面从本轮验证搜索结果自动读取 rho，避免沿用其他实验的数值。训练脚本已输出
训练 rho=0.45 对应的测试结果；以下是应用验证集选定推理 rho 后的一次固定评测。

```bash
RHO=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["rho"])' "$RUN_DIR/rho_search_validation/rho_search_summary.json")

python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --checkpoint "$RUN_DIR/best_validation_model.pth" \
  --ordinal-prediction-weight "$RHO" \
  --output-dir "$RUN_DIR/rho_${RHO}_test" \
  --gpu_id 0
```

## 比较标准

分别比较相同推理 rho=0.45 下的验证 Acc-7，以及各自按同一规则在验证集上搜索
rho 后的 Acc-7；同时记录 MAE、逐类别召回和后期曲线。测试集不参与参数选择。

若 `2e-5` 没有带来一致的验证收益，就保留 `3e-5`，不继续盲目降低学习率。
如果差距仍只有几个验证样本，给 `2e-5` 和 `3e-5` 各补 seed=1、2 的实验，
比较三个 seed 的均值和波动；每次使用独立的配置、项目名和输出目录。
