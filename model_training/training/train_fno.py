import torch  # noqa: D100
from neuralop import H1Loss, LpLoss
from neuralop.models import FNO
from neuralop.training import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.train_base import train_base

# ================================================================
# ⚙️ 1) Base configuration
# ================================================================
CONFIG = {
    # --- General ---
    "model_name": "FNO",
    "seed": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # --- Dataset ---
    "train_dataset_name": "uniform_var10_plog100_seed1",
    "ood_dataset_name": "uniform_var20_plog100_seed1",
    "train_ratio": 0.8,  # fraction of dataset used for training
    "ood_fraction": 0.2,  # fraction of OOD data for evaluation
    # --- Dataloader ---
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    # --- Training ---
    "n_epochs": 1000,
    "eval_interval": 5,  # evaluate every N epochs
    "mixed_precision": False,  # enables AMP on modern GPUs
    # --- Checkpointing & Resume ---
    # "resume_from_dir": "FNO_samples_uniform_var10_N1000_20251112_153012",
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_l2",  # metric key to monitor for best checkpoint
    "save_every": None,  # optional periodic checkpoint saving
}


# ================================================================
# 🧠 2) Model, optimizer, scheduler, and losses
# ================================================================
# --- Model ---
model = FNO(
    n_modes=(32, 32),
    hidden_channels=64,
    in_channels=4,
    out_channels=4,
).to(CONFIG["device"])

# --- Optimizer ---
optimizer = AdamW(
    model.parameters(),
    lr=1e-2,
    weight_decay=1e-4,
)

# --- Scheduler ---
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",  # reduce when validation loss plateaus
    factor=0.5,  # halve learning rate
    patience=20,  # number of evals to wait before reducing
    min_lr=1e-5,  # do not go below this lr
)

# --- Losses ---
train_loss = H1Loss(d=2)
eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
train_base(CONFIG, model, optimizer, scheduler, train_loss, eval_losses)
