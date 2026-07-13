# MiniGPT

MiniGPT is a small GPT-style language model training pipeline designed for local smoke tests and Docker-based Vertex AI custom training on GCP.

The first milestone is a reproducible proof of concept:

- GPT-2 BPE tokenization
- A compact PyTorch GPT implementation
- Dataset ingestion from curated Hugging Face datasets
- Causal language-model pretraining
- Supervised fine-tuning with response-only label masking
- Local generation/evaluation from checkpoints
- GCS-compatible checkpoint and dataset paths
- Docker and Vertex AI submission scripts

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run local tests:

```bash
python -m pytest
```

Run a tiny local pretraining smoke test:

```bash
python -m src.train_pretrain --config configs/debug.yaml
```

Generate from a checkpoint:

```bash
python -m src.generate \
  --checkpoint outputs/debug/checkpoint_last.pt \
  --prompt "The history of computing" \
  --max-new-tokens 80
```

Run SFT from a pretrained checkpoint:

```bash
python -m src.train_sft \
  --config configs/sft.yaml \
  --checkpoint gs://YOUR_BUCKET/checkpoints/pretrain/checkpoint_last.pt
```

Evaluate prompts:

```bash
python -m src.eval \
  --checkpoint outputs/debug/checkpoint_last.pt \
  --prompts prompts/eval.jsonl
```

## GCP Flow

1. Edit `configs/gcp.yaml` with your project, region, bucket, Artifact Registry repo, image URI, and service account.
2. Run `scripts/gcp_setup.sh configs/gcp.yaml`.
3. Build and push the Docker image with `scripts/build_image.sh configs/gcp.yaml`.
4. Submit jobs:

```bash
scripts/submit_vertex_pretrain.sh configs/gcp.yaml configs/pretrain.yaml
scripts/submit_vertex_sft.sh configs/gcp.yaml configs/sft.yaml gs://YOUR_BUCKET/checkpoints/pretrain/checkpoint_last.pt
```

Check Vertex AI job status:

```bash
gcloud ai custom-jobs list --region us-central1

gcloud ai custom-jobs describe \
  projects/997581983602/locations/us-central1/customJobs/JOB_ID \
  --region us-central1 \
  --format="value(state,startTime,endTime,updateTime)"

gcloud ai custom-jobs stream-logs \
  projects/997581983602/locations/us-central1/customJobs/JOB_ID \
  --region us-central1
```

Check checkpoint outputs:

```bash
gcloud storage ls gs://minigpt-wzhang-2026/checkpoints/
gcloud storage ls gs://minigpt-wzhang-2026/checkpoints/pretrain-medium/
```

## How It Works

MiniGPT is a GPT-style causal language model. Text is tokenized with the GPT-2 tokenizer, converted into fixed-length token blocks, and the model learns to predict the next token at every position.

The transformer stack uses causal self-attention plus a Mixture-of-Experts feed-forward layer. Each token is routed to the top active experts, their outputs are combined, and a small load-balancing loss encourages the router to use experts more evenly.

Training happens in stages:

1. **Pretraining** learns general next-token prediction from raw text, such as Wikipedia.
2. **Domain pretraining** continues from a general checkpoint on domain data, such as Python code.
3. **SFT** fine-tunes on prompt/response examples. Prompt tokens are masked out, so loss is focused on the response.
4. **Generation** loads a checkpoint, encodes a prompt, samples new tokens, and stops when the tokenizer EOS token is produced.

Checkpoints are saved to GCS and include model weights, optimizer state, step, model config, and training config. Resume pretraining keeps the old step and optimizer state; SFT loads pretrained weights but resets step and optimizer for a fresh fine-tuning run.

## Config Reference

### GCP configs

- `configs/gcp.yaml`: Main Vertex AI GPU config for project `minigpt-123`, bucket `minigpt-wzhang-2026`, Artifact Registry image, service account, and one T4 worker.
- `configs/gcp_cpu.yaml`: CPU-only Vertex config used for pipeline smoke tests when GPU quota is unavailable or not needed.

### Smoke and local configs

- `configs/debug.yaml`: Tiny local config using fallback inline text from `src/data.py`. Use this for local CPU/MPS smoke tests.
- `configs/vertex_smoke.yaml`: Tiny Vertex config that writes to `gs://minigpt-wzhang-2026/checkpoints/vertex-smoke`. It validates the container, Vertex job startup, GPU access, and GCS checkpoint writes without downloading a dataset.

### General pretraining configs

- `configs/pretrain_tiny.yaml`: First real-data Wikipedia GPU test. Small model/run, writes to `checkpoints/pretrain-tiny`.
- `configs/pretrain_small.yaml`: Intermediate Wikipedia run, about 51M parameters, writes to `checkpoints/pretrain-small`.
- `configs/pretrain_medium.yaml`: Main medium Wikipedia run, about 204M parameters, writes to `checkpoints/pretrain-medium`.
- `configs/pretrain_medium_continue.yaml`: Continuation config for `pretrain_medium`, same model/data paths, with `max_steps: 20000`. Use it with `--resume-checkpoint` via `scripts/submit_vertex_pretrain.sh`.
- `configs/pretrain.yaml`: Full experimental config, about 521M parameters. This is expensive and likely too large for a single T4 without tuning.

### Code pretraining configs

- `configs/pretrain_code_python.yaml`: Python code-domain pretraining using `code_search_net` / `python`, starting from the medium pretrain checkpoint. Writes to `checkpoints/pretrain-code-python`.

### SFT configs

- `configs/sft_medium.yaml`: General instruction SFT using `databricks/databricks-dolly-15k`, matching the medium model shape. Writes to `checkpoints/sft-medium`.
- `configs/sft_code_python.yaml`: Python coding instruction SFT using `iamtarun/python_code_instructions_18k_alpaca`, matching the code-pretrained medium model. Writes to `checkpoints/sft-code-python`.
- `configs/sft.yaml`: Full-model SFT config matching `configs/pretrain.yaml`. Use only with checkpoints from the full model shape.

Typical code-assistant sequence:

```bash
scripts/submit_vertex_pretrain.sh \
  configs/gcp.yaml \
  configs/pretrain_code_python.yaml \
  gs://minigpt-wzhang-2026/checkpoints/pretrain-medium/checkpoint_last.pt

scripts/submit_vertex_sft.sh \
  configs/gcp.yaml \
  configs/sft_code_python.yaml \
  gs://minigpt-wzhang-2026/checkpoints/pretrain-code-python/checkpoint_last.pt
```

## Calling the Model

Use `src.generate` to call a trained checkpoint. For instruction-tuned checkpoints, match the SFT prompt format:

```text
### Instruction:
...

### Response:
```

General instruction model:

```bash
python -m src.generate \
  --checkpoint gs://minigpt-wzhang-2026/checkpoints/sft-medium/checkpoint_last.pt \
  --prompt $'### Instruction:\nExplain what Wikipedia is in two sentences.\n\n### Response:\n' \
  --max-new-tokens 100 \
  --temperature 0.4 \
  --top-k 20
```

Code instruction model:

```bash
python -m src.generate \
  --checkpoint gs://minigpt-wzhang-2026/checkpoints/sft-code-python/checkpoint_last.pt \
  --prompt $'### Instruction:\nWrite a Python function that returns the factorial of a non-negative integer.\n\n### Response:\n' \
  --max-new-tokens 120 \
  --temperature 0.3 \
  --top-k 10
```

Pretrained-only completion model:

```bash
python -m src.generate \
  --checkpoint gs://minigpt-wzhang-2026/checkpoints/pretrain-code-python/checkpoint_last.pt \
  --prompt $'def factorial(n):\n' \
  --max-new-tokens 120 \
  --temperature 0.4 \
  --top-k 20
```

If Python GCS downloads time out for large checkpoints, copy the checkpoint locally first:

```bash
mkdir -p checkpoints
gcloud storage cp \
  gs://minigpt-wzhang-2026/checkpoints/sft-code-python/checkpoint_last.pt \
  checkpoints/sft-code-python-checkpoint_last.pt

python -m src.generate \
  --checkpoint checkpoints/sft-code-python-checkpoint_last.pt \
  --prompt $'### Instruction:\nWrite a Python function that returns the factorial of a non-negative integer.\n\n### Response:\n' \
  --max-new-tokens 120 \
  --temperature 0.3 \
  --top-k 10
```

## Notes

- `configs/debug.yaml` uses a tiny model and inline fallback data so it can run as a smoke test.
- `configs/pretrain.yaml` is sized for a small quality-focused run, not a full production language model.
- `configs/sft.yaml` formats instruction data and masks prompt tokens so the loss focuses on assistant responses.
- GCS paths are supported for checkpoint save/load and tokenized shard upload/download.
