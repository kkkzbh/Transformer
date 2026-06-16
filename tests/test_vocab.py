from __future__ import annotations

from transformer.data.vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, Vocab


def test_digit_vocab_special_ids_are_stable() -> None:
    vocab = Vocab.digits(10)

    assert vocab.tokens[:4] == (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3


def test_digit_vocab_encode_decode() -> None:
    vocab = Vocab.digits(10)

    ids = vocab.encode(["3", "7", "1", EOS_TOKEN])

    assert ids == [7, 11, 5, vocab.eos_id]
    assert vocab.decode(ids) == ["3", "7", "1"]


def test_decode_stop_at_eos_truncates_following_tokens() -> None:
    vocab = Vocab.digits(10)

    ids = vocab.encode(["3", EOS_TOKEN, "7"])

    assert vocab.decode(ids, stop_at_eos=True) == ["3"]
    assert vocab.decode(ids, stop_at_eos=False) == ["3", "7"]


def test_unknown_token_encodes_to_unk() -> None:
    vocab = Vocab.digits(10)

    assert vocab.encode(["missing"]) == [vocab.unk_id]
