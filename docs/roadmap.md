# MiniGPT Feedback Execution Roadmap

This roadmap tracks the repo feedback execution plan. Work should be completed one major item at a time, with tests run after each item before moving to the next.

## Scope

- Keep MiniGPT MoE-only for now.
- Defer the dense GPT/FFN baseline and any `ffn_type: dense | moe` option.
- Treat a public Hugging Face checkpoint as a required release deliverable before `v1.0`.

## Execution Order

1. Completed: Save this roadmap with dense FFN marked deferred and public Hugging Face checkpoint release marked required.
2. Completed: Implement reproducibility and optimizer grouping.
3. Completed: Implement RoPE positional embeddings.
4. Implement PyTorch SDPA attention.
5. Completed: Implement KV-cache decoding.
6. Implement GQA/MQA support.
7. Fix pretraining split and padded-block handling.
8. Add structured training logs and metric evaluation.
9. Expand tests and CI.
10. Polish README with architecture, results placeholders, design notes, sanitized cloud docs, Hugging Face checkpoint release instructions, and `v1.0` release-tag checklist.
11. Publish the selected trained checkpoint publicly on Hugging Face and verify the documented loading and generation commands.

## Architecture Tasks

- Add `position_embedding_type: learned | rope` and `rope_theta`, keeping learned embeddings as the default.
- Add `attention_impl: manual | sdpa`, preserving the manual path for educational clarity.
- Add KV-cache generation so autoregressive decoding can reuse prior keys and values.
- Add `n_kv_head` support:
  - MHA when `n_kv_head == n_head`
  - GQA when `1 < n_kv_head < n_head`
  - MQA when `n_kv_head == 1`
- Scale residual projection initialization for attention and MoE output projections.

## Training And Data Tasks

- Add central `seed` handling for Python, NumPy, PyTorch, CUDA, DataLoader shuffle, and generation.
- Group optimizer parameters so weight decay applies only to matrix weights, not biases or LayerNorm-style vectors.
- Make gradient accumulation at epoch boundaries intentional and logged.
- Track `tokens_seen`, tokens/sec, learning rate, gradient norm, training loss, validation loss, and perplexity in JSONL.
- Improve pretraining validation split by splitting documents before tokenization and packing.
- Avoid training on padded EOS tokens in incomplete pretraining blocks by dropping the final short block or adding loss masks.

## Evaluation, Docs, And Release Tasks

- Rename the current prompt-generation evaluation flow so it is clear that it generates samples rather than metrics.
- Add real evaluation metrics: validation cross-entropy, perplexity, and at least one small manual evaluation task.
- Add README sections for architecture, design decisions, results, qualitative samples, and benchmark tables.
- Replace public GCP project IDs, bucket names, and job paths with placeholders.
- Add a Hugging Face release workflow that publishes:
  - public checkpoint
  - model config
  - tokenizer reference
  - model card
  - training command
  - dataset and tokenizer provenance
  - metrics
  - limitations
  - sample generations
- Add README instructions for loading the public Hugging Face checkpoint locally with `src.generate`.
- Add instructions for cutting a `v1.0` Git tag only after tests pass and the Hugging Face checkpoint is public.

## Testing And CI Tasks

- Add tests for attention causality.
- Add manual-vs-SDPA equivalence tests.
- Add cached-vs-uncached decoding equivalence tests.
- Add GQA tensor shape tests.
- Add RoPE norm preservation tests.
- Add loss shifting and tied-embedding tests.
- Add optimizer parameter grouping tests.
- Add deterministic generation tests.
- Add MoE auxiliary loss tests.
- Add a tiny-batch overfit test.
- Add GitHub Actions for pytest, lint/type checks if configured, CPU smoke training, checkpoint save/load, and Docker build.

## Release Checklist

- All tests pass locally.
- CI passes on the default branch.
- README documents reproducible training and generation commands.
- README links to the public Hugging Face model repository.
- Hugging Face model repository includes a model card and public checkpoint.
- The public checkpoint can be downloaded and used by the documented `src.generate` command.
- A `v1.0` Git tag is created only after the public checkpoint is verified.
