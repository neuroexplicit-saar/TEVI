from dataclasses import dataclass


@dataclass
class Config:
    n_op_shards: int = 1
    n_replicas: int = 1  # 8

    n_dirs: int = 32768
    bs: int = 131072
    d_model: int = 1024  # 768
    k: int = 32
    auxk: int = 256

    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: float | None = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: float | None = None

    wandb_entity: str | None = None
    wandb_project: str | None = "new_openai_sae"
    wandb_name: str | None = "test_run"

    epochs: int = 200
