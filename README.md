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

## Notes

- `configs/debug.yaml` uses a tiny model and inline fallback data so it can run as a smoke test.
- `configs/pretrain.yaml` is sized for a small quality-focused run, not a full production language model.
- `configs/sft.yaml` formats instruction data and masks prompt tokens so the loss focuses on assistant responses.
- GCS paths are supported for checkpoint save/load and tokenized shard upload/download.

