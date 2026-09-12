import os
import regex as re
from collections import Counter, defaultdict


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


def get_pairs(sequence):
    """Unweighted pair counts within a single sequence."""
    local = Counter()
    for i in range(len(sequence) - 1):
        local[(sequence[i], sequence[i + 1])] += 1
    return local


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
):
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    parts = re.split("|".join(map(re.escape, special_tokens)), text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = Counter()

    for part in parts:
        for match in re.finditer(PAT, part):
            counts[match.group()] += 1

    num_merges = vocab_size - 256 - len(special_tokens)

    token_sequences = {key: list(key.encode("utf-8")) for key in counts}
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # --- ONE-TIME setup: build pair_counts + reverse index ---
    pair_counts = Counter()
    pair_to_keys = defaultdict(set)

    for key, sequence in token_sequences.items():
        freq = counts[key]
        for pair, n in get_pairs(sequence).items():
            pair_counts[pair] += n * freq
            pair_to_keys[pair].add(key)

    while len(merges) < num_merges:
        if not pair_counts:
            break  # no pairs left to merge (e.g. tiny/degenerate corpus)

        best_pair = max(
            pair_counts.items(),
            key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]])
        )[0]

        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # only touch keys that actually contain best_pair
        affected_keys = pair_to_keys.pop(best_pair, set())

        for key in affected_keys:
            freq = counts[key]
            old_seq = token_sequences[key]

            old_local = get_pairs(old_seq)
            new_seq = merge_pair(old_seq, best_pair, new_id)
            new_local = get_pairs(new_seq)

            token_sequences[key] = new_seq

            # remove this key from any pair it no longer contains
            for pair in old_local:
                if pair not in new_local:
                    pair_to_keys[pair].discard(key)
                    if not pair_to_keys[pair]:
                        del pair_to_keys[pair]

            # add this key to any pair it newly contains
            for pair in new_local:
                pair_to_keys[pair].add(key)

            # apply the count delta for just this key
            for pair, n in old_local.items():
                pair_counts[pair] -= n * freq
            for pair, n in new_local.items():
                pair_counts[pair] += n * freq

        pair_counts = +pair_counts  # drop zero/negative entries (Counter's unary + does this)

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges