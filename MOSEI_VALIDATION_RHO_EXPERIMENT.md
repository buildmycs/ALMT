# MOSEI：先对齐 checkpoint 选择与最终推理方式

此页记录已完成的联合选择实验，不再是默认流程。配置已归档为
`configs/mosei_dual_c4_intensity_rho030_valrho_lr2e-5.yaml`。
当前默认已回退为固定 rho 选 checkpoint，见 `MOSEI_FIXED_SELECTION_EXPERIMENT.md`。
以下保留该联合选择实验的设计与命令；不要复跑旧项目名覆盖已有结果。
只改验证选择流程；训练保持 `lr=2e-5`、训练 rho=0.30、ordinal_weight=0.2、
100 epoch、LR warmup=10 epoch、有序损失 warmup=5 epoch、seed=0。
有序损失不衰减，对比学习仍关闭，不增加新的训练损失。

## 本地结果说明了什么

以下指标来自已保存的验证预测，每轮样本数均为 1,871。
“搜索后”都按旧的 `bestweight.py` 对各自已选 checkpoint 搜索 0～1、步长 0.01，
不是在测试集上选 rho。

| 训练配置 | 最优 epoch | 回归头 Acc-7 | 有序头 Acc-7 | 原融合 Acc-7 | 验证最优推理 rho | 搜索后 Acc-7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3e-5，0.45/0.2 | 45 | 53.0198% | 50.4543% | 54.8904% | 0.44 | 54.9973% |
| 2e-5，0.45/0.2 | 79 | 53.8215% | 48.7974% | 54.1956% | 0.32 | 54.9973% |
| 2e-5，0.30/0.2 | 70 | 54.3560% | 50.0802% | 54.1422% | 0.07 | 54.4094% |

因此新一轮的两个单头变好，但相同旧搜索协议下的融合验证 Acc-7 没有提升：
比上一轮少判对 11 个样本。这并不能证明总体泛化显著变差，也不能依据测试集
挑出的 rho=0.15/54.4% 认定总体泛化显著改善。当前验证集 rho=0.15 的 Acc-7
为 54.0353%；验证集选出的 0.07 也只比不融合多判对 1 个样本。

当前 epoch 70 的 rho=0.3 融合相对回归头，修正 82 个样本、损失 86 个样本，
净少判对 4 个；其中中性类净少判对 50 个，-2/+2 类分别净多判对 16/22 个。
回归头的 -2/+2 recall 为 20.0%/17.75%，有序头为 49.09%/50.30%。
这说明有序头有补充作用，中心收缩仍存在；但直接提高融合占比会牺牲中性类，
提高极端 recall 不等于提高整体 Acc-7。

更直接的发现来自训练日志：**epoch 21 回归头验证 Acc-7 约为 55.05%**，
超过最终保存的 epoch 70。原代码只按训练 rho=0.3 的融合分数保存权重，
随后才在这个 checkpoint 上改推理 rho，因而可能错过更适合低 rho 的 epoch。
日志只有标量，不能恢复 epoch 21 的模型，也不能由此推断其测试分数。
需要重新训练，或使用服务器另存的对应权重。

第 90～100 轮原融合验证 Acc-7 平均约 53.94%，第 100 轮约 53.87%，
后期没有持续增长证据。先保持 100 epoch；改总轮次会改变 cosine 调度，
暂不把“训练更久”作为本轮主要实验。

数据来源：两个旧实验及 `rho030_ord020_fixed_e100_wu10_lr2e-5_seed0` 目录中的
`best_validation_predictions.npz`、`best_validation_selection.json`、三个头的
`summary.json`，以及新实验对应的 TensorBoard event 日志。

## 新的选择规则

1. 训练始终使用 rho=0.30，训练梯度、损失权重及调度不变。
2. 每个 epoch 仅在验证集比较预先声明的 `[0.0, 0.1, 0.2, 0.3]`。
   这组粗网格包含纯回归与原训练融合，不包含测试集搜索出的 0.15。
3. 以未四舍五入的验证 Acc-7 最大为主，MAE 最小为次；同一 epoch 两项完全
   相同时选较小 rho，跨 epoch 完全相同时保留较早 checkpoint。
4. 一起保存选定 epoch、`inference_rho`、`training_rho` 和候选列表。
   `best_validation_predictions.npz` 的 `predictions` 是该推理 rho 的融合，
   同时保留两个单头预测和 `inference_rho`。
5. 训练结束载入该 checkpoint，自动使用保存的 rho 测试一次。
   独立测试脚本默认也读取 checkpoint 的 rho；旧 checkpoint 无此字段时回退
   到 YAML。显式命令行 rho 仍可覆盖，但本轮正常评估不应覆盖或再次搜索。

这是修正“按一种输出选模型、用另一种输出测试”的不一致，不是保证泛化提升的
新模型。更多验证候选也会增加选择偏差，不能把新的验证最高分直接当作模型能力
提升；要使用同样候选集合和选择规则比较后续实验，并做多 seed 复核。

## 运行

```bash
python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity_rho030_valrho_lr2e-5.yaml \
  --gpu_id 0
```

新结果目录，避免覆盖旧实验：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho030_ord020_fixed_e100_wu10_lr2e-5_valrho_seed0/
```

训练结束已自动测试。若需要单独复核相同 checkpoint，直接运行以下命令，
**不再运行 `bestweight.py` 密集扫描，不再从测试集选 rho**：

```bash
python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity_rho030_valrho_lr2e-5.yaml \
  --gpu_id 0
```

查看 `best_validation_selection.json` 中的 `selected_epoch`、`inference_rho`、
`primary_value`、`secondary_value`。TensorBoard 中：

- `valid/*`：训练 rho=0.3 的原融合指标，与旧训练曲线口径一致。
- `valid_regression/*`、`valid_ordinal/*`：单头指标。
- `valid_selection/*`：当轮候选网格选出的指标；`rho` 是当轮最佳，
  `best_rho` 是截至当前保存的 checkpoint 对应 rho。

验证 loss 仍按训练 rho 计算；选择依据是选定推理输出的 Acc-7 与 MAE，不是 loss。
最终 test loss 则对应测试时的选定 rho，不应与训练 rho 下的验证 loss 直接对比。

旧 rho=0.30 配置完整归档在 `configs/mosei_dual_c4_intensity_rho030_lr2e-5.yaml`，
旧 rho=0.45 配置在 `configs/mosei_dual_c4_intensity_rho045_lr2e-5.yaml`。
没有 `validation_rho_candidates` 时保持旧的固定 rho 选择逻辑；不要复跑旧项目名
覆盖已有结果。`MOSEI_RHO030_EXPERIMENT.md` 继续记录旧协议。

## 下一次如何决定是否继续调训练参数

先看这次选中的 epoch/rho 和单头指标，检查改善是否只来自更换选择规则。
若需要比较训练 rho=0.30 与 0.45，给两者使用同一个预先固定候选列表，
各补 seed=1、2，与 seed=0 一起记录均值和波动；不能用新协议结果直接宣称胜过
旧的“固定 rho 选 epoch，再密集搜 rho”协议。为每次运行设置独立 project_name，
单用 `--seed` 不会自动更改输出目录。

当前先不同时改学习率、ordinal_weight、对比学习参数或训练轮次。
如果统一选择协议后仍没有可靠改善，再单独比较 ordinal_weight=0.15 与 0.2，
验证减小辅助目标强度能否改善常见类别；这是待验证的假设，不是本轮已启用改动。
测试集已经被反复用于探索，现有最高测试分应注明这一点；后续冻结选择规则，
避免继续按照测试成绩修改参数。
