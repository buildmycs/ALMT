# ALMT 双文本分支：共享 BERT + 门控交叉注意力

## 1. 设计目标

原始转录文本是可靠锚点，C4-Explicit 文本是带上下文语义的补充。模型不直接用
增强文本替换原文，而是让原文主动从增强文本中选择信息：

```text
原始 text_bert ─┐
                ├─ 共享 BERT ─ 共享语言投影 ─ H_original ─┐
增强 text_bert_llm ┘                         H_enhanced ─┤
                                                       ├─ 门控交叉注意力
                                                       ↓
                                           原 ALMT 的 l_encoder
                                                       ↓
                                      AHL + 多模态融合 + 回归
```

门控融合位于语言投影之后、原 ALMT `l_encoder` 之前。视觉分支、音频分支、AHL、
多模态融合层和回归头保持不变。

核心计算为：

```text
C = MHA(Q=H_original, K=H_enhanced, V=H_enhanced)
G = sigmoid(W[H_original; C; |H_original-C|; H_original*C])
s = sigmoid(alpha)
H_dual = LayerNorm(H_original + s * G * C)
```

`alpha` 默认初始化为 `-2`，所以初始 `s≈0.119`。这会限制训练初期增强文本的
影响；门控会在每个 token、每个特征维度上学习应当接收多少增强信息。

## 2. 文件

- `core/dataset_dual.py`：同时读取 `text_bert` 和 `text_bert_llm`，并检查形状。
- `models/dual_text_fusion.py`：原文作 Query 的单向门控交叉注意力。
- `models/almt_dual.py`：双文本 ALMT；BERT 和 `proj_l` 均严格共享参数。
- `train_dual.py`：独立训练入口，不修改已经复现成功的基线入口。
- `configs/mosi_dual_c4.yaml`：MOSI C4-Explicit 配置。
- `configs/mosei_dual_c4.yaml`：MOSEI C4-Explicit 配置模板。

## 3. 在服务器运行

拉取代码后，先确认配置里的 PKL 路径与实际文件一致：

```bash
git pull origin master
python train_dual.py --config_file configs/mosi_dual_c4.yaml --gpu_id 0
```

MOSEI：

```bash
python train_dual.py --config_file configs/mosei_dual_c4.yaml --gpu_id 0
```

双文本会让共享 BERT 一次处理 `2 * batch_size` 条序列。配置默认把 batch size
从 64 降为 32；RTX 5090 显存允许时可以再升高。若显存不足，先降到 16。

训练期间终端和 TensorBoard 会记录：

- `gate_mean`：增强信息门控的平均开放程度；
- `gate_std`：不同 token/维度之间的选择差异；
- `residual_scale`：增强残差的全局强度；
- `attention_entropy`：跨文本注意力是否过度集中。

## 4. 论文消融

只修改配置中的 `dual_fusion_mode` 即可复用相同训练代码和同一份双文本 PKL：

- `gated_cross`：完整模型；
- `mean`：两路语言表示直接平均；
- `original`：只使用原始文本分支；
- `enhanced`：只使用增强文本分支。

建议至少报告：

1. 原仓库 `train.py + configs/mosi.yaml` 的已复现基线；
2. `original`；
3. `enhanced`；
4. `mean`；
5. `gated_cross`；
6. C2-Explicit 与 C4-Explicit 对比（如果 C2 数据仍保留）。

`original` 模式仍会计算两路共享 BERT，适合控制训练代码差异；正式基线结果仍以
原 `train.py` 为准。所有实验应使用相同随机种子列表，并报告均值和标准差。

## 5. 第一轮训练建议

先只使用原 ALMT 的主回归 MSE，不加辅助损失。若完整模型稳定优于 `mean` 和
`enhanced`，再考虑原文/增强文本辅助头。这样可以先证明提升来自门控融合本身，
避免一次引入过多变量。
