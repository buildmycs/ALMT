# C4-Explicit v2

独立脚本：

```text
scripts/generate_c4_explicit.py
```

该脚本不导入其他生成脚本，也不使用 vLLM、FlashInfer 或 CUTLASS。固定采用：

- 目标语句之前 4 个同视频片段；
- 目标语句之后 2 个同视频片段；
- 自包含语义重写；
- 新增语义的原文证据约束；
- 情感极性和强度保持；
- Transformers + PyTorch SDPA。

## Prompt 干跑

```bash
python scripts/generate_c4_explicit.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --splits train valid \
  --limit 3 \
  --dry-run
```

## 生成 100 条

```bash
python scripts/generate_c4_explicit.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --output-path datasets/mosi/qwen25_c4_explicit.jsonl \
  --splits train valid \
  --batch-size 8 \
  --limit 100
```

## 生成全部数据

确认前 100 条后，删除 `--limit`，脚本自动续跑：

```bash
python scripts/generate_c4_explicit.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --output-path datasets/mosi/qwen25_c4_explicit.jsonl \
  --splits train valid test \
  --batch-size 8
```

## 校验和构建 PKL

```bash
python scripts/validate_llm_text.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl \
  --splits train valid test \
  --bert-path pretrained/bert-base-uncased \
  --report-path datasets/mosi/qwen25_c4_validation.json

python scripts/build_dual_text_pkl.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl \
  --output-path datasets/mosi/unaligned_50_dual_qwen25_c4.pkl \
  --bert-path pretrained/bert-base-uncased
```

原 C2 JSONL 不会被覆盖，可直接用于消融实验。
