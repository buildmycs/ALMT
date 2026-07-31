# 无 vLLM 的 Qwen2.5 增强后端

当 AutoDL 的 vLLM、FlashInfer 或 CUTLASS DSL 出现版本兼容错误时，使用：

```text
scripts/generate_llm_text_transformers.py
```

该入口使用 Hugging Face Transformers 与 PyTorch SDPA，不导入 vLLM、
FlashInfer 或 CUTLASS。它复用原生成脚本的 Prompt、上下文构造、JSON 解析、
失败回退及断点续传逻辑，可以继续写入同一个 JSONL。

## 单条测试

```bash
python scripts/generate_llm_text_transformers.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --output-path datasets/mosi/qwen25_enhanced.jsonl \
  --splits train valid \
  --context-window 2 \
  --batch-size 1 \
  --limit 1
```

启动日志中应出现：

```text
Loaded Transformers backend with PyTorch SDPA;
vLLM/FlashInfer/CUTLASS are not used.
```

## 生成 100 条

```bash
python scripts/generate_llm_text_transformers.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --output-path datasets/mosi/qwen25_enhanced.jsonl \
  --splits train valid \
  --context-window 2 \
  --batch-size 8 \
  --limit 100
```

如果显存不足，将 `--batch-size` 依次降为 `4`、`2` 或 `1`。不要指定
`--overwrite`，已有成功记录会自动跳过。

完成 train/valid 后，使用相同参数生成 test：

```bash
python scripts/generate_llm_text_transformers.py \
  --data-path datasets/mosi/unaligned_50.pkl \
  --model-path ../Qwen2.5-7B-Instruct \
  --output-path datasets/mosi/qwen25_enhanced.jsonl \
  --splits test \
  --context-window 2 \
  --batch-size 8
```

生成后的校验和双文本 PKL 构建命令与原流程完全相同。
