"""
Data preparation and tokenizer training script for CS336 Assignment 1.

Capabilities:
1. Train a BPE tokenizer on a raw text corpus (or a representative sample).
2. Save tokenizer vocabulary (`vocab.json`) and merges (`merges.txt`) in GPT-2 format.
3. Tokenize raw text files in chunks and serialize them directly to memory-mappable binary files (.bin) of uint16 integers.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from typing import Optional

import numpy as np

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer, _bytes_to_unicode


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Save vocabulary and merges in standard GPT-2 format so they can be loaded by

    Tokenizer.from_files.
    """
    byte_encoder = _bytes_to_unicode()

    def encode_token_bytes(b: bytes) -> str:
        return "".join(byte_encoder[byte_val] for byte_val in b)

    raw_vocab = {encode_token_bytes(tok_bytes): idx for idx, tok_bytes in vocab.items()}

    os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(merges_path)), exist_ok=True)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(raw_vocab, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for a_bytes, b_bytes in merges:
            f.write(f"{encode_token_bytes(a_bytes)} {encode_token_bytes(b_bytes)}\n")

    print(f"Saved vocabulary to: {vocab_path} ({len(vocab)} tokens)")
    print(f"Saved merges to: {merges_path} ({len(merges)} merges)")


def train_tokenizer(
    corpus_path: str | os.PathLike,
    vocab_size: int = 10000,
    special_tokens: Optional[list[str]] = None,
    sample_mb: Optional[float] = 20.0,
    vocab_out: str | os.PathLike = "data/vocab.json",
    merges_out: str | os.PathLike = "data/merges.txt",
) -> Tokenizer:
    """Train BPE on corpus (or a sample of it) and save vocab/merges to disk."""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    corpus_path = str(corpus_path)
    file_size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
    print(f"Corpus size: {file_size_mb:.2f} MB")

    bpe_input_path = corpus_path
    temp_file = None

    if sample_mb is not None and sample_mb > 0 and file_size_mb > sample_mb:
        sample_bytes = int(sample_mb * 1024 * 1024)
        print(f"Sampling first {sample_mb:.1f} MB ({sample_bytes:,} bytes) for BPE training...")
        temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
        with open(corpus_path, "rb") as in_f:
            chunk = in_f.read(sample_bytes)
            # Find the last newline or story boundary to avoid split tokens at the end
            last_newline = chunk.rfind(b"\n")
            if last_newline != -1:
                chunk = chunk[:last_newline]
            temp_file.write(chunk)
        temp_file.close()
        bpe_input_path = temp_file.name

    print(f"Training BPE tokenizer (vocab_size={vocab_size}, special_tokens={special_tokens})...")
    start_time = time.time()
    try:
        vocab, merges = train_bpe(
            input_path=bpe_input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
    finally:
        if temp_file is not None and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    elapsed = time.time() - start_time
    print(f"BPE training completed in {elapsed:.2f}s.")

    save_vocab_and_merges(vocab, merges, vocab_out, merges_out)
    return Tokenizer.from_files(str(vocab_out), str(merges_out), special_tokens=special_tokens)


def _encode_lines_worker(args: tuple) -> list[int]:
    """Worker function for multiprocessing: encode a batch of lines."""
    lines, vocab_path, merges_path, special_tokens = args
    # Each worker loads its own tokenizer to avoid pickle issues with compiled regex
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    result: list[int] = []
    for line in lines:
        if line:
            result.extend(tok.encode(line))
    return result


def tokenize_text_to_bin(
    input_text_path: str | os.PathLike,
    output_bin_path: str | os.PathLike,
    tokenizer: Tokenizer,
    chunk_lines: int = 10000,
    dtype: str = "uint16",
    num_workers: int = 0,
    vocab_path: str | os.PathLike | None = None,
    merges_path: str | os.PathLike | None = None,
    special_tokens: list[str] | None = None,
) -> int:
    """Tokenize a text file and write directly into a compact binary file.

    When num_workers > 0 (or left at 0 for auto-detect), uses multiprocessing
    to parallelize encoding across CPU cores for a significant speedup.
    Requires vocab_path and merges_path so workers can load their own tokenizer.

    Falls back to single-process mode if tokenizer file paths are not provided.
    """
    import multiprocessing as mp

    input_text_path = str(input_text_path)
    output_bin_path = str(output_bin_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    np_dtype = np.dtype(dtype)

    # Decide whether we can use multiprocessing
    can_multiprocess = vocab_path is not None and merges_path is not None
    if num_workers == 0:
        num_workers = max(1, mp.cpu_count() - 1) if can_multiprocess else 1
    if not can_multiprocess:
        num_workers = 1

    print(
        f"Tokenizing {input_text_path} -> {output_bin_path} "
        f"(dtype={dtype}, workers={num_workers})..."
    )
    start_time = time.time()

    if special_tokens is None:
        special_tokens = list(tokenizer.special_tokens) if tokenizer.special_tokens else []

    if num_workers > 1:
        # ---- Multiprocessing path ----
        total_tokens = 0
        lines_read = 0

        with open(input_text_path, "r", encoding="utf-8") as in_f, \
             open(output_bin_path, "wb") as out_f, \
             mp.Pool(num_workers) as pool:

            batch: list[str] = []
            pending_futures = []
            # How many lines each worker gets per task
            worker_chunk = chunk_lines

            for line in in_f:
                batch.append(line)
                lines_read += 1

                if len(batch) >= worker_chunk * num_workers:
                    # Split batch into sub-chunks, one per worker
                    chunks = [
                        batch[i:i + worker_chunk]
                        for i in range(0, len(batch), worker_chunk)
                    ]
                    tasks = [
                        (chunk, str(vocab_path), str(merges_path), special_tokens)
                        for chunk in chunks
                    ]
                    results = pool.map(_encode_lines_worker, tasks)
                    for token_ids in results:
                        total_tokens += len(token_ids)
                        arr = np.array(token_ids, dtype=np_dtype)
                        out_f.write(arr.tobytes())
                    batch = []

                    elapsed = time.time() - start_time
                    tok_per_sec = total_tokens / max(elapsed, 1e-6)
                    print(
                        f"  Processed {lines_read:,} lines | "
                        f"{total_tokens:,} tokens ({tok_per_sec:,.0f} tok/s)"
                    )

            # Process remaining lines
            if batch:
                chunks = [
                    batch[i:i + worker_chunk]
                    for i in range(0, len(batch), worker_chunk)
                ]
                tasks = [
                    (chunk, str(vocab_path), str(merges_path), special_tokens)
                    for chunk in chunks
                ]
                results = pool.map(_encode_lines_worker, tasks)
                for token_ids in results:
                    total_tokens += len(token_ids)
                    arr = np.array(token_ids, dtype=np_dtype)
                    out_f.write(arr.tobytes())
    else:
        # ---- Single-process fallback (original path) ----
        total_tokens = 0
        buffer: list[int] = []
        lines_read = 0

        with open(input_text_path, "r", encoding="utf-8") as in_f, \
             open(output_bin_path, "wb") as out_f:
            for line_num, line in enumerate(in_f, start=1):
                lines_read = line_num
                if line:
                    tokens = tokenizer.encode(line)
                    buffer.extend(tokens)
                    total_tokens += len(tokens)

                if len(buffer) >= 200_000:
                    arr = np.array(buffer, dtype=np_dtype)
                    out_f.write(arr.tobytes())
                    buffer = []

                if line_num % 100_000 == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = total_tokens / max(elapsed, 1e-6)
                    print(
                        f"  Processed {line_num:,} lines | "
                        f"{total_tokens:,} tokens ({tok_per_sec:,.0f} tok/s)"
                    )

            if buffer:
                arr = np.array(buffer, dtype=np_dtype)
                out_f.write(arr.tobytes())

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    print(
        f"Completed {output_bin_path}: {total_tokens:,} tokens, "
        f"{file_size_mb:.2f} MB in {elapsed:.1f}s "
        f"({total_tokens / max(elapsed, 1e-6):,.0f} tok/s)."
    )
    return total_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer and pre-tokenize raw text into binary memmap datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_bpe", action="store_true", help="Train a new BPE tokenizer on raw text")
    parser.add_argument("--corpus", type=str, default="data/TinyStoriesV2-GPT4-train.txt", help="Path to corpus for BPE training")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--sample_mb", type=float, default=20.0, help="Megabytes of text to sample for BPE training (0 for full corpus)")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.json", help="Path to vocab.json")
    parser.add_argument("--merges_file", type=str, default="data/merges.txt", help="Path to merges.txt")

    parser.add_argument("--tokenize", action="store_true", help="Tokenize text files to binary files")
    parser.add_argument("--input_text", type=str, default=None, help="Input text file to tokenize")
    parser.add_argument("--output_bin", type=str, default=None, help="Output binary file path")
    parser.add_argument("--dataset_dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32", "int64"], help="Binary integer data type")

    parser.add_argument(
        "--prepare_tinystories",
        action="store_true",
        help="All-in-one shortcut: train BPE on TinyStories sample and tokenize train and valid datasets",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for tokenization (0 = auto-detect)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    # All-in-one pipeline for TinyStories
    if args.prepare_tinystories:
        train_text = "data/TinyStoriesV2-GPT4-train.txt"
        valid_text = "data/TinyStoriesV2-GPT4-valid.txt"
        vocab_path = args.vocab_file
        merges_path = args.merges_file

        if not os.path.exists(train_text):
            raise FileNotFoundError(f"Training file not found: {train_text}")

        # 1. Train or load tokenizer
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            print(f"Training BPE tokenizer on {args.sample_mb} MB sample of {train_text}...")
            tokenizer = train_tokenizer(
                corpus_path=train_text,
                vocab_size=args.vocab_size,
                special_tokens=special_tokens,
                sample_mb=args.sample_mb,
                vocab_out=vocab_path,
                merges_out=merges_path,
            )
        else:
            print(f"Using existing tokenizer from {vocab_path} and {merges_path}...")
            tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

        # 2. Tokenize validation dataset
        if os.path.exists(valid_text):
            tokenize_text_to_bin(
                input_text_path=valid_text,
                output_bin_path="data/tinystories_valid.bin",
                tokenizer=tokenizer,
                dtype=args.dataset_dtype,
                vocab_path=vocab_path,
                merges_path=merges_path,
                special_tokens=special_tokens,
                num_workers=args.num_workers,
            )

        # 3. Tokenize training dataset
        tokenize_text_to_bin(
            input_text_path=train_text,
            output_bin_path="data/tinystories_train.bin",
            tokenizer=tokenizer,
            dtype=args.dataset_dtype,
            vocab_path=vocab_path,
            merges_path=merges_path,
            special_tokens=special_tokens,
            num_workers=args.num_workers,
        )

        print("\nAll datasets prepared successfully!")
        print("You can now train your model using:")
        print("  uv run python train.py --train_data data/tinystories_train.bin --val_data data/tinystories_valid.bin")
        return

    # Individual steps
    tokenizer = None
    if args.train_bpe:
        tokenizer = train_tokenizer(
            corpus_path=args.corpus,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            sample_mb=args.sample_mb,
            vocab_out=args.vocab_file,
            merges_out=args.merges_file,
        )

    if args.tokenize:
        if args.input_text is None or args.output_bin is None:
            raise ValueError("--tokenize requires both --input_text and --output_bin.")

        if tokenizer is None:
            if not (os.path.exists(args.vocab_file) and os.path.exists(args.merges_file)):
                raise FileNotFoundError(
                    f"Tokenizer files not found ({args.vocab_file}, {args.merges_file}). Run with --train_bpe first."
                )
            tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file, special_tokens=special_tokens)

        tokenize_text_to_bin(
            input_text_path=args.input_text,
            output_bin_path=args.output_bin,
            tokenizer=tokenizer,
            dtype=args.dataset_dtype,
            vocab_path=args.vocab_file,
            merges_path=args.merges_file,
            special_tokens=special_tokens,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
