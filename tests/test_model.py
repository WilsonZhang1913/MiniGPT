import torch

from src.model import GPT, GPTConfig


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
