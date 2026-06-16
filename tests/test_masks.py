from __future__ import annotations

import torch

from transformer.modeling.attention import make_causal_mask


def test_causal_mask_hides_future_positions() -> None:
    mask = make_causal_mask(4)

    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_padding_mask_uses_pad_token_positions() -> None:
    pad_id = 0
    batch = torch.tensor([[4, 5, 0], [6, 0, 0]])

    assert torch.equal(
        batch.eq(pad_id),
        torch.tensor([[False, False, True], [False, True, True]]),
    )
