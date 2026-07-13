import torch

from src.model import GPT, GPTConfig
from src.train_utils import make_optimizer, save_checkpoint, train_loop


def test_model_forward_backward():
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    y = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(x, y)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    loss.backward()


def test_generate_extends_sequence():
    config = GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    y = model.generate(x, max_new_tokens=3, top_k=10)
    assert y.shape == (1, 7)


def test_generate_stops_at_eos():
    config = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)

    def fake_forward(idx, targets=None):
        logits = torch.full((idx.size(0), idx.size(1), config.vocab_size), -1000.0)
        logits[:, -1, 3] = 1000.0
        return logits, None

    model.forward = fake_forward
    x = torch.tensor([[1, 2]])
    y = model.generate(x, max_new_tokens=5, eos_token_id=3)
    assert y.tolist() == [[1, 2, 3]]


def test_moe_routes_to_multiple_experts():
    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        n_expert=4,
        n_expert_active=2,
    )
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    expert_grads = [
        expert.net[0].weight.grad is not None and expert.net[0].weight.grad.abs().sum().item() > 0
        for expert in model.transformer["h"][0].moe.experts
    ]
    assert sum(expert_grads) >= 2


def test_train_loop_can_initialize_from_checkpoint_with_reset_step(tmp_path):
    config = {
        "model": {
            "vocab_size": 64,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "n_expert": 2,
            "n_expert_active": 1,
            "expert_hidden_mult": 2,
            "moe_aux_loss_coef": 0.01,
        },
        "train": {
            "batch_size": 2,
            "grad_accum_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "max_steps": 1,
            "eval_interval": 1,
            "save_interval": 1,
            "eval_iters": 1,
            "val_fraction": 0.25,
            "mixed_precision": False,
            "grad_clip": 1.0,
        },
    }
    model = GPT(GPTConfig(**config["model"]))
    optimizer = make_optimizer(model, 0.001, 0.0)
    source_ckpt = tmp_path / "source.pt"
    save_checkpoint(model, optimizer, step=10, config=config, path=str(source_ckpt))
    blocks = torch.randint(0, 64, (4, 9), dtype=torch.long)
    dataset = [(row[:-1], row[1:]) for row in blocks]

    last = train_loop(
        config=config,
        dataset=dataset,
        output_dir=str(tmp_path / "out"),
        model_factory=lambda: GPT(GPTConfig(**config["model"])),
        resume_checkpoint=str(source_ckpt),
        reset_step=True,
        reset_optimizer=True,
    )

    saved = torch.load(last, map_location="cpu")
    assert saved["step"] == 1
