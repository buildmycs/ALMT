# MOSEI：回退到固定 rho 选择 checkpoint

此页记录 ordinal_weight=0.2 的固定选择基线，配置已归档为
`configs/mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml`。
当前默认的 0.3/0.15 单变量实验见 `MOSEI_ORD015_EXPERIMENT.md`。
本次只回退 checkpoint 选择方式，不调整模型结构、损失、学习率或训练轮数。

- `lr=2e-5`、训练 `ordinal_prediction_weight=0.30`、`ordinal_weight=0.2`。
- 100 epoch，LR warmup=10 epoch，有序损失 warmup=5 epoch，seed=0。
- `validation_rho_candidates: null`，不再每轮搜索推理 rho。
- 按固定 rho=0.3 的验证 Acc-7 最大保存 checkpoint；完全同分时 MAE 最小优先。
- 不设置最低选择 epoch，不保证一定选中后期 checkpoint，也不保证测试分数提升。

联合选择实验的 epoch 5/rho=0 验证 Acc-7 为 54.6766%、MAE 约 0.5170；
epoch 80/rho=0.1 的日志为 54.6232%、MAE 约 0.5044，前者只多判对一个验证样本。
单次实验不足以证明早期 checkpoint 必然差或回退必然更好。本轮恢复较简单的主流程，
同时保留联合选择代码、原配置及结果，不做整仓库 Git 回滚。

## 重新训练

```bash
python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml \
  --gpu_id 0
```

启动日志应显示 `Checkpoint selection uses fixed training rho=0.3; no rho grid.`。
新结果目录为：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho030_ord020_fixed_e100_wu10_lr2e-5_fixedselect_seed0/
```

旧的 `..._seed0/` 和 `..._valrho_seed0/` 不会被本配置覆盖。
回退配置不会把现有 epoch 5 checkpoint 变成后期权重；若服务器没有另存其他轮次，
需要重新训练。训练结束会按固定 rho=0.3 自动评估选定 checkpoint 的测试结果。

## 如需校准推理 rho：仅使用验证集

以下命令在训练服务器 Bash 中执行，并在同一终端保留变量。
只对已选定的一个 checkpoint 搜索 rho，不再改变 epoch。

```bash
RUN_DIR=ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho030_ord020_fixed_e100_wu10_lr2e-5_fixedselect_seed0

python scripts/bestweight.py \
  --predictions "$RUN_DIR/best_validation_predictions.npz" \
  --step 0.01 \
  --top-k 10

RHO=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["rho"])' "$RUN_DIR/rho_search_validation/rho_search_summary.json")

python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml \
  --checkpoint "$RUN_DIR/best_validation_model.pth" \
  --ordinal-prediction-weight "$RHO" \
  --output-dir "$RUN_DIR/rho_${RHO}_test" \
  --gpu_id 0
```

固定 rho 测试结果与验证校准后的测试结果应分开记录，不按测试成绩在两者中挑选。
不能直接沿用过去从测试集挑出的 rho=0.15。

## 保留的对照

联合选择配置归档在 `configs/mosei_dual_c4_intensity_rho030_valrho_lr2e-5.yaml`，
原始固定选择配置在 `configs/mosei_dual_c4_intensity_rho030_lr2e-5.yaml`。
归档配置用于核对旧实验；重跑应更换 project_name，避免覆盖已有结果。
旧联合选择 checkpoint 仍会由独立测试脚本读取其保存的推理 rho，
因此复核旧 checkpoint 应使用其对应归档配置，不要与本轮新结果混用。

本次不新增“同一次训练保存两种选择 checkpoint”的诊断功能，保持回退范围清晰；
联合选择仅保留为可显式启用的对照功能，不参与默认训练或测试。
