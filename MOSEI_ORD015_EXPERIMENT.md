# MOSEI：固定选择下的 0.3/0.15 单变量实验

默认配置为 `configs/mosei_dual_c4_intensity.yaml`。
相对刚回退的 0.3/0.2 配置，本次只把 `objective.ordinal_weight` 从 0.2 降为 0.15。
项目名同步更换，防止覆盖旧结果。没有修改模型或损失计算公式。

## 为什么先调整有序损失，而不是继续调 rho 或 epoch

目前本地没有 `..._fixedselect_seed0` 的回退后新结果；现有参考是之前固定选择的
0.3/0.2、epoch 70 实验，不能把未完成的回退实验视为已证明有效的新基线。
重新核对其 1,871 个验证预测：

- 回归头 Acc-7=54.3560%，有序头=50.0802%，rho=0.3 的融合头=54.1422%。
- 相对回归头，融合修正 82 个样本、损失 86 个样本，净少判对 4 个。
- 中性类净少判对 50 个；-2/+2 类分别净多判对 16/22 个。
- 有序头的 -2/+2 recall 为 49.09%/50.30%，回归头为 20.00%/17.75%。
- 当前 checkpoint 验证集选择的推理 rho 为 0.07，校准后 Acc-7=54.4094%。

有序头有助于强情绪识别，但它带来的常见类别损失抵消了部分收益。
训练代码使用融合预测计算 MSE，同时对有序 logits 施加类别平衡 BCE。
因此先测试降低辅助 BCE 的相对强度 25%，看看能否保留有序信息又改善整体 Acc-7。
这只是可检验的假设，现有结果不能证明 ordinal_weight=0.2 一定过大，
降低它也可能增加中心收缩、损失极端类别 recall。

用户报告的测试最高 54.4% 来自测试集上比较 rho，应记录为探索性结果，
不能据此保证本轮超过 54.4%，也不把测试最佳 rho=0.15 直接写入训练配置。
过去试过的 0.45/0.15 与本轮的训练 rho、学习率组合不同，不能视为这组对照的结论。

## 本轮固定项

- 学习率 2e-5，训练融合 `ordinal_prediction_weight=0.30`。
- 100 epoch，LR warmup=10 epoch，seed=0，batch size=32。
- 有序损失前 5 epoch warmup，此后固定 0.15，**不使用动态衰减**。
- 对比学习关闭，其他模型参数、weight decay、类别平衡权重保持不变。
- `validation_rho_candidates: null`：按固定 rho=0.3 的验证 Acc-7 选 checkpoint，
  完全同分时 MAE 最小优先；不恢复每轮联合 rho 搜索，不设置最低选择 epoch。

## 运行与评估

在训练服务器的 Bash 中运行：

```bash
python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --gpu_id 0

RUN_DIR=ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho030_ord015_fixed_e100_wu10_lr2e-5_fixedselect_seed0

python scripts/bestweight.py \
  --predictions "$RUN_DIR/best_validation_predictions.npz" \
  --step 0.01 \
  --top-k 10

RHO=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["rho"])' "$RUN_DIR/rho_search_validation/rho_search_summary.json")

python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --checkpoint "$RUN_DIR/best_validation_model.pth" \
  --ordinal-prediction-weight "$RHO" \
  --output-dir "$RUN_DIR/rho_${RHO}_test" \
  --gpu_id 0
```

训练结束会先自动评估训练 rho=0.3 的测试结果。上述校准只用已选 checkpoint 的
验证预测决定 rho，再应用到测试集；两种结果分开记录，不按测试成绩取高者。
不要修改 YAML 的训练 rho 来做推理，也不要加载旧的 epoch 5 checkpoint 混入本轮。

## 对照与停止规则

0.3/0.2 回退基线配置归档在
`configs/mosei_dual_c4_intensity_rho030_ord020_fixedselect_lr2e-5.yaml`，
其运行方式见 `MOSEI_FIXED_SELECTION_EXPERIMENT.md`。
如果该基线尚未重跑，保留它作为同环境、同 seed 的配对对照；不要覆盖旧结果。

比较时分别列出两个头、固定 rho 融合及验证校准后的 Acc-7/MAE，
并检查中性类、-2/+2、-3/+3 recall。旧 0.3/0.2 的对应参考为
固定融合 54.1422%、验证校准 54.4094%，不能与联合选择的 epoch 5 混作同协议对照。
也要保留更早 0.45/0.2 实验验证校准约 55.00% 的参考，避免只和较弱的一轮相比。

若只多判对几个验证样本，不宣布胜出：给 0.15 和 0.2 两组各补 seed=1、2，
同 seed 配对比较均值与波动，并为每次运行设置独立 project_name。
若验证集没有收益或强情绪识别明显恶化，保留 0.2 基线，不因某个测试最高值继续
下调权重。暂不同时加 epoch、改 LR 或启用对比学习；本轮不承诺分数提升。
