"""HW2 settings. YAML files in configs/ override these defaults; the CLI overrides YAML.

Add fields freely (for example to ModelConfig when your architecture needs a new
knob); YAML files and ``key=value`` overrides can then set them. Use a sentinel such
as 0 for "off" instead of ``X | None`` types.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    image_size: int = 64  # paintings are image_size x image_size, RGB, values in [0, 1]
    batch_size: int = 128
    num_workers: int = 0  # the data sit in memory; 0 is fastest (and safest on Windows)
    styles: list[str] = field(default_factory=list)  # [] = all 27 styles, e.g. ["Impressionism"]


@dataclass
class ModelConfig:
    latent_dim: int = 100  # K: z ~ N(0, I_K)
    g_channels: int = 64  # generator width; how you use it is up to you
    d_channels: int = 64  # discriminator width
    init: str = "dcgan"  # "dcgan" (models/gan.py: init_weights) or "default" (PyTorch's)


@dataclass
class TrainConfig:
    epochs: int = 30
    loss: str = "nonsaturating"  # "minimax", "nonsaturating", or "hinge" (losses.py)
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5  # Adam betas; DCGAN uses (0.5, 0.999)
    beta2: float = 0.999
    hflip: bool = False  # mirror real paintings at random (training only)
    ema_decay: float = 0.0  # exponential moving average of the weights for sampling; 0 = off
    seed: int = 0
    device: str = "auto"  # "auto" (CUDA if available), "cuda", or "cpu"
    sample_every: int = 1  # save a sample grid of the fixed z every N epochs (and the last)
    fid_every: int = 5  # FID on the validation split every N epochs (picks best.pt); 0 = off
    fid_samples: int = 2000  # images per validation FID (at most the size of the val split)
    max_batches_per_epoch: int = 0  # 0 = all; small values make quick smoke runs
    # Stop a dead game: mean D(G(z)) of an epoch below 1e-6 (G gets no gradient) for this many
    # epochs in a row -> one last validation FID, then stop. 0 = off (always all epochs).
    stop_dead_epochs: int = 0


@dataclass
class EvalConfig:
    checkpoint: str = "best"  # "best" (lowest validation FID-2048) or "last"
    use_ema: bool = True  # sample from the EMA weights when the run has them
    reference_split: str = "test"  # real paintings that FID compares against
    num_samples: int = 10000  # images per FID/IS estimate (FID-10k)
    repeats: int = 3  # independent sets of generated images -> mean and std
    batch_size: int = 128
    real_baseline: bool = True  # also score real training paintings against the reference
    seed: int = 1234


@dataclass
class Config:
    name: str = "gan"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
