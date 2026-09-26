# MOSEI Dual-C4-Intensity 运行流程

MOSEI 可以继续使用 `train_dual.py`，不需要复制训练脚本。数据维度、项目目录和初始
损失参数由 `configs/mosei_dual_c4_intensity.yaml` 提供；训练、validation 融合权重搜索、
固定权重 test 评测和 Acc-7 分析与 MOSI 使用相同脚本。

## 1. 检查数据

配置默认读取：

```text
datasets/mosei/unaligned_50_dual_qwen25_c4.pkl
```

该文件必须包含 train、valid、test 三个 split，以及 `text_bert_llm` 和
`raw_text_llm`。不要把 MOSI 的 PKL 路径改到 MOSEI 配置中。

## 2. 训练并按 Validation Acc-7 选 checkpoint

本轮实验将学习率从 `1e-4` 降至 `5e-5`；训练融合权重为 `0.45`，有序损失权重为
`0.2`，训练预算为 100 epoch，学习率 warmup 为 10 epoch，seed 为 0。
使用独立的 `project_name` 保存本轮结果，避免覆盖原基线。

```bash
python train_dual.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --gpu_id 0
```

输出目录为：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/
```

训练完成后确认存在：

```text
best_validation_model.pth
best_validation_predictions.npz
best_validation_predictions.csv
best_test_predictions.npz
best_validation_selection.json
```

## 3. 只在 Validation 上搜索融合权重 rho

```bash
python scripts/bestweight.py \
  --predictions ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/best_validation_predictions.npz \
  --step 0.01 \
  --top-k 10
```

结果会保存到：

```text
ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/rho_search_validation/
├── rho_search_all.csv
└── rho_search_summary.json
```

记录 `rho_search_summary.json` 中的 `best.rho`。该数值是固定 checkpoint 上的
validation 后处理参数，不要把它写回 YAML 后重新训练，也不要在 test 上重新搜索。

## 4. 使用选定 rho 在 Test 上评测一次

以下以 validation 选出的 rho 为 `0.35` 举例；实际运行时替换为本轮验证集搜索出的值，
并同步调整输出目录名称。

```bash
python scripts/evaluate_selected_test.py \
  --config_file configs/mosei_dual_c4_intensity.yaml \
  --checkpoint ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/best_validation_model.pth \
  --ordinal-prediction-weight 0.35 \
  --output-dir ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/rho_035_test \
  --gpu_id 0
```

终端会同时打印 YAML 中的训练配置 rho 和本次推理 rho。输出 JSON 也会保存这两个值，
防止后续混淆 checkpoint、训练参数和后处理参数。

## 5. 生成 Test Acc-7 混淆矩阵

```bash
python scripts/analyze_acc7.py \
  --predictions ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/rho_035_test/selected_test_predictions.npz \
  --title "MOSEI test - validation-selected rho=0.35" \
  --output-dir ckpt/ALMT_MOSEI_Dual_C4_Intensity_rho045_ord020_fixed_e100_wu10_lr5e-5_seed0/rho_035_test/acc7
```

## 6. 与 MOSI 一致但不能直接复用的内容

以下流程保持一致：

```text
训练 -> validation 选 checkpoint -> validation 搜 rho -> 固定 rho 跑一次 test
     -> test 混淆矩阵和逐等级 Recall/F1
```

以下参数需要在 MOSEI validation 上单独确定，不能直接照搬 MOSI：

- `ordinal_prediction_weight`；
- `ordinal_weight` 和 `contrastive_weight`；
- 对比温度与类别平衡上限。

MOSEI 的音频/视觉特征维度、序列长度、样本量和标签分布都与 MOSI 不同，所以应保留
MOSEI YAML 中对应的数据结构参数，并从其当前初始权重开始做 validation 消融。
