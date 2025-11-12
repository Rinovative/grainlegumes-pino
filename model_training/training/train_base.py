import os
import json
import datetime
import inspect
import torch
import wandb
import random
import numpy as np
from neuralop import Trainer
from src import dataset


# ================================================================
# 🧭 0) Utility functions
# ================================================================
def set_seed(seed: int = 1) -> None:
    """Ensure reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_init_params(obj):
    """Extract non-callable init parameters for logging/debugging."""
    try:
        sig = inspect.signature(obj.__class__.__init__)
        args = {k: getattr(obj, k, None) for k in sig.parameters if k != "self"}
        return {k: v for k, v in args.items() if v is not None and not callable(v)}
    except Exception:
        return {}


def make_json_safe(obj):
    """Recursively convert non-serializable objects to strings for JSON logging."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def build_wandb_config(CONFIG):
    """Create a structured and human-readable config dict for W&B."""
    return {
        "general": {
            "run_name": CONFIG.get("model_name"),
            "seed": CONFIG.get("seed"),
            "device": CONFIG.get("device"),
            "torch_version": CONFIG.get("torch_version", torch.__version__),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "data": {
            "train_dataset_name": CONFIG.get("train_dataset_name"),
            "ood_dataset_name": CONFIG.get("ood_dataset_name"),
            "train_ratio": CONFIG.get("train_ratio"),
            "ood_fraction": CONFIG.get("ood_fraction"),
            "batch_size": CONFIG.get("batch_size"),
            "num_workers": CONFIG.get("num_workers"),
        },
        "training": {
            "n_epochs": CONFIG.get("n_epochs"),
            "eval_interval": CONFIG.get("eval_interval"),
            "mixed_precision": CONFIG.get("mixed_precision"),
            "early_stopping": CONFIG.get("early_stopping"),
            "base_patience": CONFIG.get("base_patience"),
            "growth_factor": CONFIG.get("growth_factor"),
            "min_delta": CONFIG.get("min_delta"),
        },
        "model": {
            "architecture": CONFIG.get("model_name"),
            "train_loss": CONFIG.get("train_loss"),
            "eval_losses": CONFIG.get("eval_losses"),
            "optimizer": CONFIG.get("optimizer"),
            "scheduler": CONFIG.get("scheduler"),
        },
    }


# ================================================================
# 🚀 1) Main training pipeline
# ================================================================
def train_base(CONFIG, model, optimizer, scheduler=None, train_loss=None, eval_losses=None):
    """
    Generic base training routine using neuralop.Trainer.
    Supports:
      - automatic checkpointing & resume
      - adaptive early stopping
      - arbitrary optimizer, scheduler, and loss setups
      - resume from "latest" or run short name only
    """

    # ------------------------------------------------------------
    # ⚙️ 1. Setup
    # ------------------------------------------------------------
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    DATA_ROOT = "model_training/data/raw"
    train_dataset = os.path.join(DATA_ROOT, CONFIG["train_dataset_name"], f"{CONFIG['train_dataset_name']}.pt")
    ood_dataset = os.path.join(DATA_ROOT, CONFIG["ood_dataset_name"], f"{CONFIG['ood_dataset_name']}.pt")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"{CONFIG['model_name']}_{CONFIG['train_dataset_name']}_{timestamp}"
    SAVE_DIR = os.path.join("model_training/data/processed/model", RUN_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 🔑 2. W&B setup
    # ------------------------------------------------------------
    os.environ["WANDB_API_KEY"] = "REMOVED_WANDB_KEY"
    os.environ["WANDB_PROJECT"] = "grainlegumes_pino"
    os.environ["WANDB_ENTITY"] = "Rinovative-Hub"
    os.environ["WANDB_DIR"] = "model_training/training/wandb"

    # ------------------------------------------------------------
    # 📦 3. Dataset creation
    # ------------------------------------------------------------
    dataloader_cfg = {
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["num_workers"],
        "pin_memory": CONFIG["pin_memory"],
        "persistent_workers": CONFIG["persistent_workers"],
    }

    train_loader, test_loaders, data_processor = dataset.dataset_base.create_dataloaders(
        dataset_cls=dataset.dataset_simulation.PermeabilityFlowDataset,
        path_train=train_dataset,
        path_test_ood=ood_dataset,
        train_ratio=CONFIG["train_ratio"],
        ood_fraction=CONFIG["ood_fraction"],
        **dataloader_cfg,
    )
    data_processor = data_processor.to(device)

    # ------------------------------------------------------------
    # 🧾 4. Configuration logging
    # ------------------------------------------------------------
    CONFIG.update({
        "run_name": RUN_NAME,
        "train_dataset": train_dataset,
        "ood_dataset": ood_dataset,
        "save_dir": SAVE_DIR,
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": extract_init_params(optimizer),
        "scheduler": scheduler.__class__.__name__ if scheduler else None,
        "scheduler_params": extract_init_params(scheduler) if scheduler else None,
        "train_loss": train_loss.__class__.__name__ if train_loss else None,
        "eval_losses": [v.__class__.__name__ for v in eval_losses.values()] if eval_losses else None,
        "model": model.__class__.__name__,
        "model_params": extract_init_params(model),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=RUN_NAME,
        dir=os.environ["WANDB_DIR"],
        config=build_wandb_config(CONFIG),
        reinit=True,
    )

    # 💾 Safe JSON dump
    config_path = os.path.join(SAVE_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(CONFIG), f, indent=2)
    wandb.save(config_path)

    # ------------------------------------------------------------
    # ♻️ 5. Resume Handling (short name / "latest")
    # ------------------------------------------------------------
    resume_from = CONFIG.get("resume_from_dir", None)
    if resume_from:
        base = "model_training/data/processed/model"

        if resume_from == "latest":
            runs = sorted([r for r in os.listdir(base) if os.path.isdir(os.path.join(base, r))])
            if not runs:
                raise FileNotFoundError("❌ No previous runs found for 'latest'.")
            resume_from = os.path.join(base, runs[-1])
        else:
            if os.path.isabs(resume_from) or resume_from.startswith("model_training"):
                raise ValueError("❌ Only short run names or 'latest' are allowed for resume_from_dir.")
            resume_from = os.path.join(base, resume_from)

        if not os.path.isdir(resume_from):
            raise FileNotFoundError(f"❌ Resume directory not found: {resume_from}")

        CONFIG["resume_from_dir"] = resume_from
        print(f"♻️  Resuming training from checkpoint: {resume_from}")

    # ------------------------------------------------------------
    # 🧠 6. Trainer setup
    # ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        n_epochs=CONFIG["n_epochs"],
        wandb_log=True,
        device=device,
        mixed_precision=CONFIG["mixed_precision"],
        eval_interval=CONFIG["eval_interval"],
        verbose=True,
    )

    # ------------------------------------------------------------
    # ⏸️ 7. Adaptive Early Stopping
    # ------------------------------------------------------------
    use_early_stopping = CONFIG.get("early_stopping", False)
    base_patience = CONFIG.get("base_patience", 30)
    growth_factor = CONFIG.get("growth_factor", 1.5)
    min_delta = CONFIG.get("min_delta", 1e-4)
    best_loss = float("inf")
    no_improve = 0

    # ------------------------------------------------------------
    # 🏋️‍♂️ 8. Training Loop
    # ------------------------------------------------------------
    try:
        for epoch in range(CONFIG["n_epochs"]):
            metrics = trainer.train(
                train_loader=train_loader,
                test_loaders=test_loaders,
                optimizer=optimizer,
                scheduler=scheduler,
                training_loss=train_loss,
                eval_losses=eval_losses,
                eval_modes=CONFIG.get("eval_modes", {}),
                save_dir=SAVE_DIR,
                save_best=CONFIG.get("save_best", "eval_l2"),
                save_every=CONFIG.get("save_every", None),
                resume_from_dir=CONFIG.get("resume_from_dir", None),
                max_autoregressive_steps=CONFIG.get("max_autoregressive_steps", None),
            )

            if not use_early_stopping:
                break  # Trainer handles full loop internally

            current_loss_val = metrics.get("eval_l2")
            if current_loss_val is None:
                continue
            current_loss = float(current_loss_val)

            # --- Adaptive patience & delta ---
            patience = int(base_patience * (1 + (epoch / CONFIG["n_epochs"]) * growth_factor))
            adaptive_delta = max(min_delta * (1 - epoch / CONFIG["n_epochs"]), 1e-6)

            if current_loss + adaptive_delta < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"⏹️ Adaptive early stopping triggered at epoch {epoch + 1}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted — saving current state...")

    finally:
        wandb.finish()
        print("🏁 Training complete. Best model and checkpoints handled by Trainer.")
