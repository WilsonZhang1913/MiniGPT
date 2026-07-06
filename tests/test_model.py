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

