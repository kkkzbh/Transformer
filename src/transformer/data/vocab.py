from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


@dataclass(frozen=True, slots=True)
class Vocab:
    tokens: tuple[str, ...]                                      # 有序 token 表。
    token_to_id: dict[str, int] = field(init=False, repr=False)  # 词元到 id 的索引。

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary tokens must be unique.")
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(f"Vocabulary must start with {SPECIAL_TOKENS}.")
        object.__setattr__(
            self,
            "token_to_id",
            {token: token_id for token_id, token in enumerate(self.tokens)},
        )

    @classmethod
    def digits(cls, count: int = 10) -> Vocab:
        if count <= 0:
            raise ValueError("digit count must be positive.")
        return cls(tokens=SPECIAL_TOKENS + tuple(str(index) for index in range(count)))

    @classmethod
    def from_json(cls, path: str | Path) -> Vocab:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        tokens = raw["tokens"]
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
            raise ValueError("vocab json must contain a string list at key 'tokens'.")
        return cls(tokens=tuple(tokens))

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int], *, stop_at_eos: bool = True) -> list[str]:
        decoded: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.tokens):
                token = UNK_TOKEN
            else:
                token = self.tokens[token_id]
            if stop_at_eos and token == EOS_TOKEN:
                break
            if token not in SPECIAL_TOKENS:
                decoded.append(token)
        return decoded

    def require_known(self, tokens: Sequence[str]) -> None:
        missing = [token for token in tokens if token not in self.token_to_id]
        if missing:
            raise ValueError(f"Unknown token(s): {', '.join(missing)}")

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tokens": list(self.tokens),
                    "special_tokens": {
                        "pad": PAD_TOKEN,
                        "bos": BOS_TOKEN,
                        "eos": EOS_TOKEN,
                        "unk": UNK_TOKEN,
                    },
                },
                handle,
                ensure_ascii=True,
                indent=2,
            )
            handle.write("\n")
