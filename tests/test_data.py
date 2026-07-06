from src.data import SFTDataset, normalize_text, tokenize_blocks


class TinyTokenizer:
    eos_token = ""
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text):
        return [min(ord(ch), 255) for ch in text]


def test_normalize_text():
    assert normalize_text(" a\n\n b\t c ") == "a b c"


def test_tokenize_blocks(monkeypatch):
    monkeypatch.setattr("src.data.load_tokenizer", lambda _: TinyTokenizer())
    blocks = tokenize_blocks(["abcd", "efgh"], "tiny", 4)
    assert blocks == [[97, 98, 99, 100], [0, 101, 102, 103], [104, 0, 0, 0]]


def test_sft_dataset_masks_prompt(monkeypatch):
    monkeypatch.setattr("src.data.load_tokenizer", lambda _: TinyTokenizer())
    ds = SFTDataset([{"instruction": "Say hi", "response": "Hi"}], "tiny", 32)
    x, y = ds[0]
    assert x.shape == y.shape
    assert (y == -100).sum().item() > 0
