"""
Training script for CS336 Assignment 1: Language Modeling.

Supports:
- Configuring model and optimizer hyperparameters via CLI or function call.
- Memory-efficient loading of large datasets via np.memmap.
- Periodic validation evaluation.
- Periodic and final checkpoint saving / resuming via save_checkpoint and load_checkpoint.
- Console and Weights & Biases (wandb) logging.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.module import (
    adamw,
    SGD,
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_scheduler,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.transformer_lm import TransformerLM


def init_weights(model: nn.Module, std: float = 0.02) -> None:
    """Initialize model parameters with truncated normal distributions for weights,

    ones for RMSNorm scales, and zeros for biases.
    """
    for name, param in model.named_parameters():
        if "scale" in name:
            nn.init.ones_(param)
        elif "weight" in name or "W" in name or "proj" in name:
            nn.init.trunc_normal_(param, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        elif "bias" in name:
            nn.init.zeros_(param)


def load_memmap_dataset(path: str | os.PathLike, dtype: str = "uint16") -> np.ndarray:
    """Load a dataset efficiently from disk using memory mapping.

    Supports both .npy files (via np.load with mmap_mode) and raw binary files
    (via np.memmap).
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Dataset file does not exist at: {path_str}")

    if path_str.endswith(".npy"):
        dataset = np.load(path_str, mmap_mode="r")
    else:
        np_dtype = np.dtype(dtype)
        dataset = np.memmap(path_str, dtype=np_dtype, mode="r")

    return dataset


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: data_loading,
    eval_iters: int,
    loss_fn: nn.Module,
    vocab_size: int,
) -> float:
    """Estimate average loss over eval_iters batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = loader()
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(
    # Dataset hyperparameters
    train_data: str | os.PathLike | np.ndarray,
    val_data: Optional[str | os.PathLike | np.ndarray] = None,
    dataset_dtype: str = "uint16",
    # Model hyperparameters
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    d_ff: Optional[int] = None,
    rope_theta: float = 10000.0,
    # Optimizer hyperparameters
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    clip_grad_norm: float = 1.0,
    # Scheduler hyperparameters
    warmup_iters: int = 100,
    cosine_cycle_iters: int = 1000,
    # Training control hyperparameters
    batch_size: int = 32,
    max_iters: int = 1000,
    eval_interval: int = 100,
    eval_iters: int = 20,
    log_interval: int = 10,
    device: str = "cpu",
    seed: int = 42,
    # Checkpoint hyperparameters
    checkpoint_dir: Optional[str | os.PathLike] = "checkpoints",
    checkpoint_interval: int = 200,
    resume_from: Optional[str | os.PathLike] = None,
    # Logging hyperparameters
    log_file: Optional[str | os.PathLike] = None,
    use_wandb: bool = False,
    wandb_project: str = "cs336-assignment1",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> nn.Module:
    """Run the training loop for TransformerLM."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device)
    print(f"Training on device: {target_device}")

    # 1. Load memory-mapped datasets
    if isinstance(train_data, (str, os.PathLike)):
        print(f"Loading train dataset via np.memmap from {train_data} (dtype={dataset_dtype})...")
        train_dataset = load_memmap_dataset(train_data, dtype=dataset_dtype)
    else:
        train_dataset = train_data

    if len(train_dataset) <= context_length + 1:
        raise ValueError(
            f"Train dataset length ({len(train_dataset)}) must be greater than context_length + 1 ({context_length + 1})."
        )
    print(f"Train dataset loaded: {len(train_dataset):,} tokens.")

    val_dataset = None
    if val_data is not None:
        if isinstance(val_data, (str, os.PathLike)):
            print(f"Loading val dataset via np.memmap from {val_data} (dtype={dataset_dtype})...")
            val_dataset = load_memmap_dataset(val_data, dtype=dataset_dtype)
        else:
            val_dataset = val_data
        print(f"Validation dataset loaded: {len(val_dataset):,} tokens.")

    # Create batch data loaders
    train_loader = data_loading(
        dataset=train_dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=str(target_device),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = data_loading(
            dataset=val_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=str(target_device),
        )

    # 2. Compute d_ff if not provided
    if d_ff is None or d_ff <= 0:
        hidden_dim = int((8.0 / 3.0) * d_model)
        d_ff = 64 * ((hidden_dim + 63) // 64)

    # 3. Instantiate model
    print(
        f"Initializing TransformerLM: vocab_size={vocab_size}, context_length={context_length}, "
        f"d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}, theta={rope_theta}"
    )
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
    )
    init_weights(model)
    model.to(target_device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # 4. Optimizer & Loss function
    if optimizer_name.lower() == "adamw":
        optimizer = adamw(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: ['adamw', 'sgd']")

    lr_scheduler = learning_rate_scheduler(
        alpha_max=learning_rate,
        alpha_min=min_learning_rate,
        T_warm=warmup_iters,
        T_cosine=cosine_cycle_iters,
    )
    clipper = gradient_clipping(max_norm=clip_grad_norm) if clip_grad_norm > 0 else None
    loss_fn = cross_entropy()

    # 5. Checkpoint directory and resuming
    start_iter = 0
    if resume_from is not None:
        print(f"Resuming training from checkpoint: {resume_from}")
        start_iter = load_checkpoint(resume_from, model=model, optimizer=optimizer)
        print(f"Successfully loaded checkpoint at iteration {start_iter}")

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Determine local log file path (defaults to <checkpoint_dir>/metrics.jsonl)
    log_file_path = log_file
    if log_file_path is None and checkpoint_dir is not None:
        log_file_path = os.path.join(checkpoint_dir, "metrics.jsonl")
    if log_file_path is not None:
        parent_dir = os.path.dirname(log_file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        print(f"Logging structured metrics to: {log_file_path}")

    # 6. Wandb setup
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_config = {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "rope_theta": rope_theta,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "min_learning_rate": min_learning_rate,
                "weight_decay": weight_decay,
                "beta1": beta1,
                "beta2": beta2,
                "clip_grad_norm": clip_grad_norm,
                "warmup_iters": warmup_iters,
                "cosine_cycle_iters": cosine_cycle_iters,
                "batch_size": batch_size,
                "max_iters": max_iters,
                "total_params": total_params,
            }
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config=wandb_config,
            )
            print(f"Weights & Biases logging initialized for run: {wandb.run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}. Continuing without wandb.")
            wandb_run = None

    # 7. Training Loop
    print(f"Starting training from step {start_iter} to {max_iters}...")
    model.train()
    start_time = time.time()
    step_start_time = time.time()

    for it in range(start_iter, max_iters):
        # Update learning rate via scheduler
        current_lr = lr_scheduler(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        current_time = time.time()
        wall_clock_time = current_time - start_time

        # Periodic Evaluation
        if (it % eval_interval == 0 or it == max_iters - 1) and it > start_iter:
            train_eval_loss = estimate_loss(
                model=model,
                loader=train_loader,
                eval_iters=eval_iters,
                loss_fn=loss_fn,
                vocab_size=vocab_size,
            )
            val_eval_loss = None
            val_ppl = None
            msg = f"[EVAL] Step {it:6d} | Time: {wall_clock_time:6.1f}s | Train Loss: {train_eval_loss:.4f}"
            if val_loader is not None:
                val_eval_loss = estimate_loss(
                    model=model,
                    loader=val_loader,
                    eval_iters=eval_iters,
                    loss_fn=loss_fn,
                    vocab_size=vocab_size,
                )
                val_ppl = math.exp(min(val_eval_loss, 20.0))
                msg += f" | Val Loss: {val_eval_loss:.4f} | Val PPL: {val_ppl:.2f}"
            print(msg)

            eval_metrics = {
                "step": it,
                "wall_clock_time": round(wall_clock_time, 3),
                "train/eval_loss": train_eval_loss,
                "train/eval_perplexity": math.exp(min(train_eval_loss, 20.0)),
            }
            if val_eval_loss is not None:
                eval_metrics["val/loss"] = val_eval_loss
                eval_metrics["val/perplexity"] = val_ppl

            if wandb_run is not None:
                import wandb
                wandb.log(eval_metrics, step=it)

            if log_file_path is not None:
                with open(log_file_path, "a") as f:
                    import json
                    f.write(json.dumps({"type": "eval", **eval_metrics}) + "\n")

        # Periodic Checkpointing
        if checkpoint_dir is not None and checkpoint_interval > 0 and (it % checkpoint_interval == 0 and it > start_iter):
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{it}.pt")
            save_checkpoint(model=model, optimizer=optimizer, iteration=it, out=ckpt_path)
            print(f"[CHECKPOINT] Saved checkpoint to {ckpt_path}")

        # Training Step
        optimizer.zero_grad()
        x, y = train_loader()
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        grad_norm = None
        if clipper is not None:
            grad_norm = clipper(model.parameters())

        optimizer.step()

        # Periodic Console & Wandb Logging
        if it % log_interval == 0 or it == max_iters - 1:
            step_time = time.time() - step_start_time
            step_start_time = time.time()
            tokens_per_sec = (batch_size * context_length * log_interval) / max(step_time, 1e-6)
            grad_norm_str = f"{grad_norm.item():.4f}" if grad_norm is not None else "N/A"
            current_wall_time = time.time() - start_time
            print(
                f"Step {it:6d}/{max_iters:6d} | Loss: {loss.item():.4f} | Time: {current_wall_time:6.1f}s | "
                f"LR: {current_lr:.6e} | Grad Norm: {grad_norm_str} | Speed: {tokens_per_sec:,.0f} tok/s"
            )
            step_metrics = {
                "step": it,
                "wall_clock_time": round(current_wall_time, 3),
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "train/grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                "train/tokens_per_sec": tokens_per_sec,
            }
            if wandb_run is not None:
                import wandb
                wandb.log(step_metrics, step=it)

            if log_file_path is not None:
                with open(log_file_path, "a") as f:
                    import json
                    f.write(json.dumps({"type": "train", **step_metrics}) + "\n")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Save final checkpoint
    if checkpoint_dir is not None:
        final_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(model=model, optimizer=optimizer, iteration=max_iters, out=final_ckpt_path)
        print(f"[CHECKPOINT] Saved final checkpoint to {final_ckpt_path}")

    if wandb_run is not None:
        import wandb
        wandb.finish()

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TransformerLM on memory-mapped datasets with CS336 components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset arguments
    parser.add_argument("--train_data", type=str, default=None, help="Path to binary training dataset file (.bin, .npy)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to binary validation dataset file (.bin, .npy)")
    parser.add_argument("--dataset_dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32", "int64"], help="Data type of binary dataset files")
    parser.add_argument("--synthetic_data", action="store_true", help="Generate synthetic token data for testing/smoke testing")
    parser.add_argument("--synthetic_tokens", type=int, default=50000, help="Number of synthetic tokens to create if --synthetic_data is enabled")

    # Model architecture hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feedforward dimension (defaults to SwiGLU 8/3*d_model rounded to multiple of 64)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Base theta for Rotary Positional Embeddings (RoPE)")

    # Optimizer hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer choice")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Peak learning rate (alpha_max)")
    parser.add_argument("--min_learning_rate", "--min_lr", type=float, default=1e-4, help="Minimum learning rate (alpha_min)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max gradient L2 norm for clipping (0 to disable)")

    # Scheduler hyperparameters
    parser.add_argument("--warmup_iters", type=int, default=100, help="Linear warmup iterations (T_warm)")
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000, help="Cosine decay iterations (T_cosine)")

    # Training control hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per training step")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=100, help="Iterations between validation evaluations")
    parser.add_argument("--eval_iters", type=int, default=20, help="Batches to average over during validation evaluation")
    parser.add_argument("--log_interval", type=int, default=10, help="Iterations between console logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training ('cuda', 'cpu', 'cuda:0', etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint hyperparameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where checkpoints will be saved")
    parser.add_argument("--checkpoint_interval", type=int, default=200, help="Save a checkpoint every N iterations (0 to disable)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file (.pt) to resume training from")

    # Logging hyperparameters
    parser.add_argument("--log_file", type=str, default=None, help="Path to local JSONL metrics log file (defaults to <checkpoint_dir>/metrics.jsonl)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights and Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team name")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Handle synthetic data option for easy testing / demonstration
    train_data = args.train_data
    val_data = args.val_data

    if args.synthetic_data or train_data is None:
        if train_data is None and not args.synthetic_data:
            print("No --train_data provided. Falling back to synthetic dataset for demonstration.")
        os.makedirs("data/synthetic", exist_ok=True)
        train_path = "data/synthetic/synth_train.bin"
        val_path = "data/synthetic/synth_val.bin"
        synth_np_dtype = np.dtype(args.dataset_dtype)

        print(f"Generating synthetic memmap dataset at {train_path} ({args.synthetic_tokens} tokens)...")
        synth_train = np.memmap(train_path, dtype=synth_np_dtype, mode="w+", shape=(args.synthetic_tokens,))
        synth_train[:] = np.random.randint(0, args.vocab_size, size=args.synthetic_tokens, dtype=synth_np_dtype)
        synth_train.flush()

        val_tokens = max(args.synthetic_tokens // 5, args.context_length * 10)
        synth_val = np.memmap(val_path, dtype=synth_np_dtype, mode="w+", shape=(val_tokens,))
        synth_val[:] = np.random.randint(0, args.vocab_size, size=val_tokens, dtype=synth_np_dtype)
        synth_val.flush()

        train_data = train_path
        val_data = val_path

    train(
        train_data=train_data,
        val_data=val_data,
        dataset_dtype=args.dataset_dtype,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        clip_grad_norm=args.clip_grad_norm,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        log_file=args.log_file,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
