# Suggested figure captions

The three figures are exported as editable SVG/PDF files and 600-DPI PNG files.
For paper typesetting, use PDF (preferred) or SVG; use PNG only when the venue or
word-processing software requires a raster image.

## Figure 1 — Proposed model framework

**English caption.** Bottom-up architecture of the proposed multimodal sentiment
model. Original and C4-Explicit enhanced texts are processed by a parameter-shared
BERT and language projection, followed by original-anchored gated cross-attention.
The resulting multi-scale language features guide the three AHL stages to learn
audio-visual hyper-modality representations. Cross-modal fusion produces the final
feature, which is passed to regression and monotonic ordinal heads; an intensity
projection supports contrastive learning during training. Calibrated regression
and ordinal predictions yield the final sentiment decision.

**中文图注。** 本文提出的多模态情感分析模型自下而上结构。原始文本与 C4-Explicit
增强文本经过参数共享的 BERT 和语言投影层编码，并通过以原文为锚点的门控交叉注意力
进行融合。所得多尺度语言特征逐层引导三级 AHL 学习音视频超模态表征；跨模态融合产生
最终特征后，分别进入回归头与单调有序头，强度投影仅在训练阶段用于对比学习。校准后
的回归与有序预测共同得到最终情感判断。

## Figure 2 — Failure caused by text ambiguity

**English caption.** An illustrative failure case caused by textual ambiguity.
Although the local context implies a negative opinion, the unresolved pronoun and
the superficially positive word *great* conceal the sarcastic intent. When audio
and visual evidence is weak, the model over-relies on the surface lexical cue and
predicts the wrong sentiment polarity. This MOSI-style example is constructed for
illustration and is not a verbatim dataset sample.

**中文图注。** 文本歧义导致的示例性错误案例。尽管局部上下文表达了负面评价，未被
补全的指代和表面积极词 *great* 掩盖了讽刺意图；当音频与视觉证据较弱时，模型过度
依赖表层词汇线索，最终得到错误的情感极性。该案例为 MOSI 风格的说明性示例，并非
数据集原句。

## Figure 3 — Correction after LLM enhancement

**English caption.** The same example after C4-Explicit enhancement. The offline
LLM resolves the referent, disambiguates the sarcastic expression, and makes the
implicit negative evaluation explicit without replacing the raw-text branch.
Gated dual-text fusion therefore supplies clearer semantic evidence and recovers
the correct sentiment polarity.

**中文图注。** 同一案例经过 C4-Explicit 增强后的结果。离线大语言模型补全指代、
消解讽刺表达并显式化隐含的负面评价，同时保留原始文本分支；门控双文本融合因而获得
更清晰的语义证据，并恢复正确的情感极性判断。

## Source and editing note

- The AHL and cross-modal fusion backbone follows the architecture described in *Adaptive Language-
  guided Multimodal Transformer for Consistent Multimodal Sentiment Analysis*
  (EMNLP 2023).
- The illustrative utterance in Figures 2 and 3 was written specifically for the
  paper figure. It should not be cited as an actual MOSI/MOSEI sample.
- Regenerate all files with `python scripts/generate_paper_figures.py`.

## Figure 4 — Same polarity, incorrect intensity

**English caption.** An illustrative intensity-level error without a polarity
reversal. The original target utterance expresses mild dissatisfaction on its
surface, while the surrounding context implies a strongly negative evaluation.
The model therefore preserves the negative polarity but underestimates the
ordinal sentiment level from -3 to -1. This MOSI-style example is constructed
for illustration and is not a verbatim dataset sample.

**中文图注。** 不发生极性反转的情感强度等级错误示例。目标话语在表层仅表达轻度不满，
而上下文体现了强烈负面评价。因此模型虽然保持了负面极性，却将 -3 等级低估为 -1。
该案例为说明性 MOSI 风格示例，并非数据集原句。

## Figure 5 — Intensity correction after enhancement

**English caption.** The same example after C4-Explicit enhancement. Contextual
causes and evaluative severity are made explicit while the negative polarity is
preserved. Gated dual-text fusion consequently recovers the correct strongly
negative level.

**中文图注。** 同一案例经过 C4-Explicit 增强后的结果。增强文本在保持负面极性的同时，
显式表达上下文中的评价原因与强度；门控双文本融合因而恢复正确的强负面等级。
