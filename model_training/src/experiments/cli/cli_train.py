"""
===============================================================================
 cli_train.py
===============================================================================
Unified training entry point for config-driven training.

Responsibilities:
  - Parse command-line arguments (config path, resume, device override)
  - Load and resolve experiment configuration with defaults
  - Create run output directory with reproducible structure
  - Build dataset, model, loss, optimizer, scheduler components
  - Execute training loop with logging and checkpointing
  - Save effective config, split indices, normalizer to run directory

Design principles:
  - Thin orchestration layer; logic delegated to factories and training loop
  - Single entry point replaces 4 model-specific train_*.py scripts
  - Config-driven; no hardcoded defaults or architecture branching
  - Minimal exception handling (fail fast, clear errors)
  - No side effects before training starts (no model moves to device early)

This script does NOT:
  - Define loss or model architectures (factory functions do)
  - Manage Optuna (see cli_optuna.py)
  - Handle checkpoint resumption logic (training loop does)
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.experiments.config.experiments_config_loader import (
    create_dataloaders_from_config,
    load_and_resolve_config,
    save_yaml,
)
from src.learning.losses.learning_losses_factory import (
    build_eval_losses,
    build_training_loss,
)
from src.learning.models.learning_models_factory import build_model
from src.learning.training.learning_training_loop import train_loop
from src.learning.training.learning_training_optim import (
    build_optimizer,
    build_scheduler,
)


def main() -> int:
    """
    Run the training entry point.

    Returns
    -------
    int
        Exit code (0 on success)

    """
    parser = argparse.ArgumentParser(description="Train a neural operator model from config")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from run directory (loads best_checkpoint.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override training output root directory",
    )

    args = parser.parse_args()

    # Load and resolve config
    config = load_and_resolve_config(args.config_path)

    # Override device if specified
    if args.device:
        config["run"]["device"] = args.device

    # Override output root if specified
    if args.output_root:
        config["paths"]["train_root"] = args.output_root

    # Create run directory
    train_root = Path(config["paths"]["train_root"])
    task = config["task"]
    run_name = config["run"]["name"]
    run_dir = train_root / task / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    # Save effective config
    config_path = run_dir / "config.yaml"
    save_yaml(config, config_path)
    print(f"Config saved: {config_path}")

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders_from_config(config)
    train_loader = dataloaders["train"]
    eval_loader = dataloaders["eval"]
    data_processor = dataloaders["data_processor"]
    split_indices = dataloaders["split_indices"]

    # Save normalizer and split indices for reproducibility
    normalizer_path = run_dir / "normalizer.pt"
    torch.save(data_processor.state_dict(), normalizer_path)

    split_indices_path = run_dir / "split_indices.pt"
    torch.save(split_indices, split_indices_path)

    print(f"Normalizer saved: {normalizer_path}")
    print(f"Split indices saved: {split_indices_path}")

    # Build model
    print(f"Building model: {config['model']['architecture']}")
    model = build_model(config)

    # Build loss functions
    print("Building loss functions...")
    train_loss = build_training_loss(config)
    set_normalizers = getattr(train_loss, "set_normalizers", None)
    if callable(set_normalizers):
        set_normalizers(
            in_normalizer=data_processor.in_normalizer,
            out_normalizer=data_processor.out_normalizer,
        )
    eval_losses = build_eval_losses(config, out_normalizer=data_processor.out_normalizer)

    # Build optimizer and scheduler
    print("Building optimizer and scheduler...")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Determine checkpoint path if resuming
    checkpoint_path = None
    if args.resume:
        resume_dir = Path(args.resume)
        checkpoint_path = resume_dir / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            msg = f"Checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)
        print(f"Resuming from: {checkpoint_path}")

    # Run training loop
    print("Starting training loop...")
    start_time = datetime.now(UTC)

    result = train_loop(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=train_loss,
        eval_losses=eval_losses,
        data_processor=data_processor,
        scheduler=scheduler,
        save_dir=run_dir,
        use_amp=config["training"].get("mixed_precision", False),
        resume_from=checkpoint_path,
    )

    end_time = datetime.now(UTC)
    elapsed_seconds = (end_time - start_time).total_seconds()

    # Save summary
    summary = {
        "task": task,
        "model_architecture": config["model"]["architecture"],
        "best_epoch": result["best_epoch"],
        "best_metric": result["best_metric"],
        "metric_name": config["training"].get("save_best_metric", "eval_overall_rmse"),
        "checkpoint_path": result["checkpoint_path"],
        "status": result["status"],
        "elapsed_seconds": elapsed_seconds,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved: {summary_path}")
    print("\nTraining complete!")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best metric: {result['best_metric']:.6f}")
    print(f"  Elapsed time: {elapsed_seconds:.1f}s")
    print(f"  Status: {result['status']}")

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
