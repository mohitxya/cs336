import json
from collections.abc import Iterable, Iterator

import regex as re  


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2's reversible byte <-> unicode-string mapping. Used so that arbitrary
    bytes can be round-tripped through a JSON string when saving/loading vocab
    and merges to/from disk (mirrors the format used by GPT-2's encoder.json /
    vocab.bpe). If your own train_bpe code saved vocab/merges differently
    (e.g. as pickled dict[int, bytes]), adjust `from_files` accordingly.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # rank of each merge pair -> lower rank = applied first (earlier merge)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        # reverse lookup: token bytes -> id
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Make sure every special token actually has an id. If it's missing
        # from vocab (shouldn't normally happen), add it so decode/encode work.
        next_id = max(vocab.keys(), default=-1) + 1
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.byte_to_id:
                self.vocab[next_id] = tok_bytes
                self.byte_to_id[tok_bytes] = next_id
                next_id += 1

        # compile pattern used to split off special tokens before pretokenizing.
        # Sort longest-first so overlapping special tokens (e.g. "<|x|>" vs
        # "<|x|><|x|>") are matched greedily/correctly.
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            self._special_re = re.compile(pattern)
        else:
            self._special_re = None

        self._pat_re = re.compile(PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

        def decode_token_str(s: str) -> bytes:
            return bytes(byte_decoder[c] for c in s)

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        vocab: dict[int, bytes] = {
            idx: decode_token_str(tok) for tok, idx in raw_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(" ")
                merges.append((decode_token_str(a), decode_token_str(b)))

        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, token_bytes: bytes) -> list[bytes]:
        """Apply learned merges (in rank order) to a single pretoken's bytes."""
        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        if len(parts) <= 1:
            return parts

        while True:
            best_pair = None
            best_rank = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
            if len(parts) == 1:
                break
        return parts

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a chunk of text that contains no special tokens."""
        ids: list[int] = []
        for match in self._pat_re.finditer(text):
            pretoken = match.group().encode("utf-8")
            for piece in self._bpe_merge(pretoken):
                ids.append(self.byte_to_id[piece])
        return ids

    def encode(self, text: str) -> list[int]:
        if not self._special_re:
            return self._encode_chunk(text)

        ids: list[int] = []
        # re.split with a capturing group keeps the special tokens in the result
        for part in self._special_re.split(text):
            if part == "":
                continue
            if part in self.special_tokens:
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")