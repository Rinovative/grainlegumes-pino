import os
import json
import datetime
import inspect
import random
import numpy as np
import torch
import wandb
from neuralop import Trainer
from src import dataset


# ================================================================
# 🧭 Utilities
# ================================================================
def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_init_params(obj):
    try:
        sig = inspect.signature(obj.__class__.__init__)
        args = {k: getattr(obj, k, None) for k in sig.parameters if k != "self"}
        return {k: v for k, v in args.items() if not callable(v)}
    except Exception:
        return {}


def make_json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def build_wandb_config(CONFIG, model, optimizer, scheduler, train_loss, eval_losses):
    return {
        "general": {
            "run_name": CONFIG["model_name"],
            "seed": CONFIG["seed"],
            "device": CONFIG["device"],
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "data": {
            "train_dataset": CONFIG["train_dataset_name"],
            "ood_dataset": CONFIG["ood_dataset_name"],
            "train_ratio": CONFIG["train_ratio"],
            "ood_fraction": CONFIG["ood_fraction"],
            "batch_size": CONFIG["batch_size"],
            "num_workers": CONFIG["num_workers"],
        },
        "training": {
            "n_epochs": CONFIG["n_epochs"],
            "eval_interval": CONFIG["eval_interval"],
            "mixed_precision": CONFIG["mixed_precision"],
        },
        "model": {
            "architecture": type(model).__name__,
            "model_params": extract_init_params(model),
            "optimizer": type(optimizer).__name__,
            "optimizer_params": extract_init_params(optimizer),
            "scheduler": type(scheduler).__name__ if scheduler else None,
            "scheduler_params": extract_init_params(scheduler) if scheduler else None,
            "train_loss": type(train_loss).__name__ if train_loss else None,
            "eval_losses": {k: type(v).__name__ for k, v in eval_losses.items()},
        },
    }


# ================================================================
# 🚀 Main Training Pipeline
# ================================================================
def train_base(CONFIG, model, optimizer, scheduler=None, train_loss=None, eval_losses=None):

    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    DATA_ROOT = "model_training/data/raw"
    train_dataset = os.path.join(DATA_ROOT, CONFIG["train_dataset_name"], f"{CONFIG['train_dataset_name']}.pt")
    ood_dataset = os.path.join(DATA_ROOT, CONFIG["ood_dataset_name"], f"{CONFIG['ood_dataset_name']}.pt")

    # ------------------------------------------------------------
    # 🔁 Resume logic FIRST (important!)
    # ------------------------------------------------------------
    resume_from = CONFIG.get("resume_from_dir", None)
    base = "model_training/data/processed/model"

    if resume_from:
        if resume_from == "latest":
            runs = sorted([r for r in os.listdir(base) if os.path.isdir(os.path.join(base, r))])
            if not runs:
                raise FileNotFoundError("❌ No previous runs for ‘latest’.")
            resume_from = os.path.join(base, runs[-1])
        else:
            resume_from = os.path.join(base, resume_from)

        if not os.path.isdir(resume_from):
            raise FileNotFoundError(f"❌ resume_from_dir not found: {resume_from}")

        print(f"♻️  Resuming from checkpoint: {resume_from}")

        # reuse existing run folder
        RUN_NAME = os.path.basename(resume_from)
        SAVE_DIR = resume_from

    else:
        # create new run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_NAME = f"{CONFIG['model_name']}_{CONFIG['train_dataset_name']}_{timestamp}"
        SAVE_DIR = os.path.join(base, RUN_NAME)
        os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 🟦 W&B Setup
    # ------------------------------------------------------------
    os.environ["WANDB_API_KEY"] = "REMOVED_WANDB_KEY"
    os.environ["WANDB_PROJECT"] = "grainlegumes_pino"
    os.environ["WANDB_ENTITY"] = "Rinovative-Hub"
    os.environ["WANDB_DIR"] = "model_training/training/wandb"

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=RUN_NAME,
        dir=os.environ["WANDB_DIR"],
        config=build_wandb_config(CONFIG, model, optimizer, scheduler, train_loss, eval_losses),
        reinit=True,
    )

    # Save config
    config_path = os.path.join(SAVE_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(CONFIG), f, indent=2)
    wandb.save(config_path)

    # ------------------------------------------------------------
    # 📦 Dataset
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
    # Trainer Setup
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
    # 🚀 Start Training
    # ------------------------------------------------------------
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_dir=SAVE_DIR,
        save_best=CONFIG.get("save_best"),
        resume_from_dir=resume_from,
    )

    wandb.finish()
    print("🏁 Training complete.")
