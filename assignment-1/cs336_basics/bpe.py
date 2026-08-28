import os
import regex as re
from collections import Counter

def merge_pair(sequence, pair, new_id):
    result = []
    i = 0

    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1

    return result

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
):
    with open(input_path, "r", encoding="utf-8") as file: 
        text = file.read()
        parts = re.split(
            "|".join(map(re.escape, special_tokens)),
            text
        )

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counts = Counter()

        for part in parts: 
            for match in re.finditer(PAT, part):
                token = match.group()
                counts[token] +=1

        # now we'll apply the bpe algorithm
        # counts is essentially: token: count, token: count and so on . 

        num_merges = vocab_size - 256 - len(special_tokens)

        token_sequences = {
            key: list(key.encode("utf-8"))
            for key in counts
        }

        vocab = {
            i : bytes([i])
            for i in range(256)
        }

        merges = []

        while len(merges) < num_merges: 

            pair_counts = Counter()

            for key, sequence in token_sequences.items(): 
                freq = counts[key]

                for i in range(len(sequence)-1):
                    pair = (sequence[i], sequence[i+1])
                    pair_counts[pair] += freq

            best_pair = max(
                pair_counts.items(),
                key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]])
            )[0]

            new_id = len(vocab)
            vocab[new_id] = (
                vocab[best_pair[0]] + vocab[best_pair[1]]
            )

            merges.append(
                (
                    vocab[best_pair[0]],
                    vocab[best_pair[1]]
                )
            )

            for key in token_sequences: 
                token_sequences[key] = merge_pair(
                    token_sequences[key], 
                    best_pair, 
                    new_id
                )
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        return vocab, merges