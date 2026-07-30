# Qwen2.5 离线文本增强

该流程使用冻结的 Qwen2.5-Instruct 对 MOSI/MOSEI 原始转录文本进行：

- 指代补全；
- 语义消歧；
- 隐含语义显式化。

生成过程不会读取情感标签、音频或视觉特征。原始 PKL 不会被覆盖。

## 1. AutoDL 环境

脚本使用 vLLM 批量推理。请使用与 AutoDL 当前 PyTorch/CUDA 镜像兼容的
vLLM 版本。ALMT 环境还需要仓库 `requirements.txt` 中的依赖。

确认模型是 Instruct 版本，例如：

```bash
ls ./pretrained/Qwen2.5-7B-Instruct/config.json
```

下文假定：

```text
模型：pretrained/Qwen2.5-7B-Instruct
MOSI：datasets/MOSI/unaligned_50.pkl
BERT：pretrained/bert-base-uncased
```

如果服务器路径不同，只需修改命令中的 `--model-path`。

## 2. Prompt 干跑

干跑不会加载 Qwen，也不会写文件：

```bash
python scripts/generate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --model-path pretrained/Qwen2.5-7B-Instruct \
  --splits train valid \
  --limit 3 \
  --dry-run
```

先人工检查打印出的上下文和 Prompt。上下文只会来自相同 split、相同视频的
前后片段。

## 3. 小规模试生成

先生成 100 条：

```bash
python scripts/generate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --model-path pretrained/Qwen2.5-7B-Instruct \
  --output-path datasets/MOSI/qwen25_enhanced.jsonl \
  --splits train valid \
  --context-window 2 \
  --batch-size 64 \
  --limit 100
```

输出采用 JSONL，并在每批结束后立即落盘。重复执行相同命令会跳过已经生成的
样本，因此可用于断点续传。不要在续跑时再次指定 `--limit 100`，否则每次只会
继续生成 100 条。

检查试生成结果：

```bash
python scripts/validate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --enhanced-path datasets/MOSI/qwen25_enhanced.jsonl \
  --splits train valid \
  --bert-path pretrained/bert-base-uncased
```

试生成未覆盖完整 split 时，校验脚本会报告缺失并返回非零退出码，这是预期行为。

## 4. 生成 train/valid

删除 `--limit` 后断点续跑：

```bash
python scripts/generate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --model-path pretrained/Qwen2.5-7B-Instruct \
  --output-path datasets/MOSI/qwen25_enhanced.jsonl \
  --splits train valid \
  --context-window 2 \
  --batch-size 64
```

在 train/valid 上人工抽查并确定 Prompt 后，应冻结
`qwen25-semantic-rewrite-v1`，不要根据 test 输出继续修改 Prompt。

## 5. 使用冻结 Prompt 生成 test

继续写入同一个 JSONL：

```bash
python scripts/generate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --model-path pretrained/Qwen2.5-7B-Instruct \
  --output-path datasets/MOSI/qwen25_enhanced.jsonl \
  --splits test \
  --context-window 2 \
  --batch-size 64
```

完整校验：

```bash
python scripts/validate_llm_text.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --enhanced-path datasets/MOSI/qwen25_enhanced.jsonl \
  --splits train valid test \
  --bert-path pretrained/bert-base-uncased \
  --report-path datasets/MOSI/qwen25_validation.json
```

## 6. 构建双文本 PKL

```bash
python scripts/build_dual_text_pkl.py \
  --data-path datasets/MOSI/unaligned_50.pkl \
  --enhanced-path datasets/MOSI/qwen25_enhanced.jsonl \
  --output-path datasets/MOSI/unaligned_50_dual_qwen25.pkl \
  --bert-path pretrained/bert-base-uncased
```

新 PKL 在每个 split 中新增：

```text
raw_text_llm   : Qwen 增强后的字符串
text_bert_llm  : shape 与 text_bert 相同的 (N, 3, 50) float32 数组
```

默认情况下，增强文本超过 50 个 BERT token 时回退为原文，并在
`.pkl.meta.json` 中记录数量；不会静默截断生成文本。原始文本自身超过 50 token
时仍沿用基线的截断行为。

输出文件已经被仓库的 `.gitignore` 排除，不会意外上传数据或模型权重。

## 7. 重要参数

- `temperature=0`：固定使用贪心解码。
- `seed=42`：记录推理种子。
- `context-window=2`：目标语句前后各两个同视频片段。
- `max-model-len=2048`：该任务上下文很短，无需占用超长 KV cache。
- `gpu-memory-utilization=0.90`：适合单张 5090；若 OOM 可降至 `0.85`。
- `batch-size` 只控制提交给 vLLM 的请求数，不等同于神经网络训练 batch。

所有 JSONL 记录均包含 Prompt 哈希、模型配置哈希、生成参数、原文哈希和原始
响应，便于论文审计与复现。
