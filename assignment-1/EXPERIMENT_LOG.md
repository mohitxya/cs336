# CS336 Experiment Log & Tracking Infrastructure

This document serves as the deliverable for **Problem (experiment_log): Experiment logging**.

---

## 1. Experiment Tracking Infrastructure

Our training pipeline in [`cs336_basics/train.py`](file:///home/mohit/cs-fundamentals/cs336/assignment-1/cs336_basics/train.py) provides unified logging across three backends:

1. **Console Output**: Real-time progress with step counts, wall-clock time, loss, learning rate, gradient norm, and token throughput (`tokens/sec`).
2. **Local Structured JSONL Logs (`metrics.jsonl`)**: Automatically saved to `<checkpoint_dir>/metrics.jsonl` (or via `--log_file`). Every record contains:
   - `step`: Gradient update step.
   - `wall_clock_time`: Elapsed seconds since training began.
   - `train/loss`: Training loss at that step.
   - `val/loss`: Validation loss at evaluation checkpoints.
   - `val/perplexity`: Validation perplexity.
   - `train/lr`: Effective learning rate.
   - `train/grad_norm`: L2 gradient norm before clipping.
   - `train/tokens_per_sec`: Processing throughput.
3. **Weights & Biases (`wandb`)**: Optional cloud tracking via `--wandb` for online interactive dashboards, tracking both step-based and wall-clock loss curves.

---

## 2. Generating Loss Curves

To generate loss curves with respect to **gradient steps** and **wall-clock time**, you can use the following script:

```python
import json
import matplotlib.pyplot as plt

def plot_experiment_metrics(metrics_jsonl_path: str, save_prefix: str = "experiment"):
    train_steps, train_times, train_losses = [], [], []
    eval_steps, eval_times, val_losses = [], [], []

    with open(metrics_jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "train":
                train_steps.append(record["step"])
                train_times.append(record["wall_clock_time"])
                train_losses.append(record["train/loss"])
            elif record.get("type") == "eval" and "val/loss" in record:
                eval_steps.append(record["step"])
                eval_times.append(record["wall_clock_time"])
                val_losses.append(record["val/loss"])

    # Figure 1: Loss vs Gradient Steps
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.6)
    if val_losses:
        plt.plot(eval_steps, val_losses, label="Validation Loss", color="red", marker="o")
    plt.xlabel("Gradient Steps")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss vs. Gradient Steps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{save_prefix}_loss_vs_steps.png", dpi=300)
    plt.close()

    # Figure 2: Loss vs Wall-Clock Time
    plt.figure(figsize=(8, 5))
    plt.plot(train_times, train_losses, label="Train Loss", alpha=0.6)
    if val_losses:
        plt.plot(eval_times, val_losses, label="Validation Loss", color="red", marker="o")
    plt.xlabel("Wall-Clock Time (seconds)")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss vs. Wall-Clock Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{save_prefix}_loss_vs_time.png", dpi=300)
    plt.close()

    print(f"Saved plots to {save_prefix}_loss_vs_steps.png and {save_prefix}_loss_vs_time.png")
```

---

## 3. Experiment Log

| Exp ID | Date / Time | Model Config (`d_model`, `layers`, `heads`, `ctx`) | Optimizer & LR | Dataset | Max Steps | Final Val Loss | Final Val PPL | Wall-Clock Time | Notes / Key Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **001** | Initial Test | `d=64, L=2, H=2, ctx=32` | AdamW (`lr=1e-3`, cosine) | Synthetic (50k) | 25 | 9.215 | 10046 | 1.3s | Validated data loader, gradient clipping, and checkpointing. |
| **002** | Checkpoint Resume | `d=64, L=2, H=2, ctx=32` | AdamW (`lr=1e-3`, cosine) | Synthetic (50k) | 30 | 9.230 | 10203 | 1.6s | Verified checkpoint state restore at step 10, smooth continuation. |
| **003** | TinyStories Baseline | `d=256, L=4, H=4, ctx=256` | AdamW (`lr=1e-3`, cosine) | TinyStoriesV2 | 1000 | TBD | TBD | TBD | Baseline architecture on TinyStories. |
| **004** | LR Ablation | `d=256, L=4, H=4, ctx=256` | AdamW (`lr=3e-4`, cosine) | TinyStoriesV2 | 1000 | TBD | TBD | TBD | Testing smaller peak learning rate. |

---

## 4. How to Run & Log an Experiment

Run an experiment with both local JSONL logging and WandB:

```bash
uv run python train.py \
    --train_data data/TinyStoriesV2-train.bin \
    --val_data data/TinyStoriesV2-valid.bin \
    --dataset_dtype uint16 \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --max_iters 1000 \
    --eval_interval 100 \
    --checkpoint_dir checkpoints/exp_003 \
    --wandb \
    --wandb_project cs336-assignment1 \
    --wandb_run_name "exp_003_tinystories_baseline"
```
