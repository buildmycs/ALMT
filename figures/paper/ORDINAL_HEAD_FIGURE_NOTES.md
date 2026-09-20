# Ordinal-guided regression: prediction-module figures

## Files

- `ordinal_regression_heads_compact.svg` / `.png`: bottom-to-top inference module for inserting above an existing multimodal backbone.
- `ordinal_regression_heads_detail.svg` / `.png`: expanded prediction and training diagram.
- Vector PDFs: `output/pdf/ordinal_regression_heads_compact.pdf` and `output/pdf/ordinal_regression_heads_detail.pdf`.
- Rebuild with `python scripts/draw_ordinal_regression_heads.py` (Matplotlib).

SVG text remains text; shapes and connections are editable. PNGs are 300 DPI.

## Suggested English caption

**Ordinal-guided regression for fine-grained sentiment prediction.** The final fused multimodal representation is passed to a continuous regression head and a monotonic ordinal head. The ordinal head uses a shared latent score and six strictly ordered learnable thresholds to estimate cumulative probabilities for seven sentiment levels. Its expected sentiment score is combined with the regression output to obtain the final prediction. During training, threshold-balanced binary cross-entropy supervises the cumulative targets, while mean squared error is applied to the final blended prediction. Solid arrows denote prediction flow and dashed arrows denote training-only supervision. The auxiliary ordinal loss is gradually introduced using a warm-up factor.

## 中文图注

**面向细粒度情感预测的有序引导回归模块。** 多模态融合后的共享表征分别输入连续回归头和单调有序头。有序头利用共享潜在评分与六个严格有序的可学习阈值，估计七级情感标签对应的累积超越概率，并将得到的期望情感分值与回归分值融合。在训练阶段，阈值均衡二元交叉熵监督累积标签，均方误差监督融合后的最终预测。实线表示预测数据流，虚线表示仅训练阶段使用的监督；有序辅助损失通过预热系数逐步引入。

## Implementation consistency

1. Input `z` is the final fused 128-dimensional feature, not the pre-fusion AHL feature.
2. The two affine maps are distinct: the regression score is not reused as the ordinal score.
3. Class index `c = round(clip(y, -3, 3)) + 3` is in `{0,...,6}`. Cumulative targets are `q_k = 1[c > k]` for `k=0,...,5`.
4. The head estimates `p_k = P(c > k | z) = sigmoid(s - theta_k)`; these are not seven softmax class probabilities.
5. `theta_0` is free and `theta_k = theta_(k-1) + softplus(delta_k) + 1e-4` for `k=1,...,5`. These are learned latent-score thresholds, not permanently fixed sentiment cutoffs.
6. Increasing thresholds imply non-increasing cumulative probabilities. This is not a guarantee of error-free sample ranking or calibrated probabilities.
7. Ordinal expectation is `sum(p_k) - 3`; the model blends it with the continuous regression output using fixed hyperparameter `rho`, not a learned per-sample gate.
8. Threshold-specific positive/negative BCE weights come only from training labels and are capped. The implementation uses BCE-with-logits for stability; probability notation is the mathematically equivalent diagram abstraction.
9. MSE supervises the FINAL blended prediction. There is no extra MSE applied only to the regression head in this implementation.
10. `r(e)` denotes auxiliary-loss warm-up. It does not warm up the inference blending coefficient `rho`.
11. No contrastive branch is included because the selected paper route sets its weight to zero.
12. Figures leave `rho` and `lambda_ord` symbolic: the current MOSI config differs from the user's previously discussed 0.2/0.2 setting; use the final experiment configuration in the paper's implementation-details section.

## Literature used for conceptual framing

- ALMT: https://aclanthology.org/2023.emnlp-main.49/ (fused multimodal feature before prediction).
- TMSON: https://arxiv.org/abs/2404.08923 (explicit separation of multimodal representation and ordinal supervision; its actual ordinal mechanism differs from this implementation).
- CORAL: https://arxiv.org/abs/1901.07884 (cumulative ordinal decisions and consistency; do not claim ordinal learning itself is new).

These diagrams are newly drawn from the local implementation, not copied from the referenced papers.
