"""Experiment-level configuration and result contracts for multipath DLGN runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from layer_lenses.training import TrainConfig


@dataclass(frozen=True)
class SeedBundle:
    """All seeds used by one run across data, init, and training branches."""

    master_seed: int
    data_seed: int
    init_seed: int
    train_seed_phase1: int
    train_seed_phase2: int
    train_seed_one_phase: int


@dataclass(frozen=True)
class SeedOffsets:
    """Deterministic offsets to derive branch seeds from one master seed."""

    data: int = 0
    init: int = 1000
    phase1: int = 2000
    phase2: int = 2001
    one_phase: int = 2002

    def resolve(self, master_seed: int) -> SeedBundle:
        """Resolve a concrete SeedBundle from a master seed."""
        return SeedBundle(
            master_seed=master_seed,
            data_seed=master_seed + self.data,
            init_seed=master_seed + self.init,
            train_seed_phase1=master_seed + self.phase1,
            train_seed_phase2=master_seed + self.phase2,
            train_seed_one_phase=master_seed + self.one_phase,
        )


@dataclass(frozen=True)
class DataConfig:
    """Dataset generation and split settings."""

    dim: int = 100
    depth: int = 5
    n_train: int = 80_000
    threshold: float = 0.0

    @property
    def num_data(self) -> int:
        return 2 * self.n_train


@dataclass(frozen=True)
class ModelConfig:
    """DLGNSF architecture and initialization settings."""

    hidden_dims: tuple[int, ...] = (16, 16, 16, 16, 16)
    beta: float = 11.0
    bias: bool = False
    value_input_mode: str = "ones"
    gating_weight_scale: float = 1.0
    value_weight_scale: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level multipath experiment config for one or many master seeds."""

    experiment_name: str = "multipath_dlgn"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    seed_offsets: SeedOffsets = SeedOffsets()
    master_seed: int = 5178
    total_epochs: int = 4000
    phase1_epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 160
    snapshot_stride: int = 20
    device: str = "cpu"
    show_progress: bool = False
    lr_scheduler: str = "cosine"
    lr_scheduler_eta_min_ratio: float = 0.2
    weight_decay_gating_phase1: float = 2e-4
    weight_decay_value_phase1: float = 0.0
    weight_decay_gating_phase2: float = 0.0
    weight_decay_value_phase2: float = 0.0
    weight_decay_gating_one_phase: float = 0.0
    weight_decay_value_one_phase: float = 0.0
    gating_norm_zero_threshold: float = 2.5

    def with_master_seed(self, master_seed: int) -> ExperimentConfig:
        """Return a clone with a different master seed."""
        return replace(self, master_seed=master_seed)

    @property
    def phase2_epochs(self) -> int:
        return self.total_epochs - self.phase1_epochs

    @property
    def seeds(self) -> SeedBundle:
        return self.seed_offsets.resolve(self.master_seed)

    def train_config_phase1(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.phase1_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seeds.train_seed_phase1,
            device=self.device,
            snapshot_epochs=tuple(range(0, self.phase1_epochs + 1, self.snapshot_stride)),
            show_progress=self.show_progress,
            weight_decay_gating=self.weight_decay_gating_phase1,
            weight_decay_value=self.weight_decay_value_phase1,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_eta_min_ratio=self.lr_scheduler_eta_min_ratio,
        )

    def train_config_phase2_nozero(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.phase2_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seeds.train_seed_phase2,
            device=self.device,
            snapshot_epochs=tuple(range(0, self.phase2_epochs + 1, self.snapshot_stride)),
            show_progress=self.show_progress,
            weight_decay_gating=self.weight_decay_gating_phase2,
            weight_decay_value=self.weight_decay_value_phase2,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_eta_min_ratio=self.lr_scheduler_eta_min_ratio,
        )

    def train_config_phase2_zeroed(self) -> TrainConfig:
        return replace(self.train_config_phase2_nozero(), freeze_zero_gating_rows=True)

    def train_config_one_phase(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.total_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seeds.train_seed_one_phase,
            device=self.device,
            snapshot_epochs=tuple(range(0, self.total_epochs + 1, self.snapshot_stride)),
            show_progress=self.show_progress,
            weight_decay_gating=self.weight_decay_gating_one_phase,
            weight_decay_value=self.weight_decay_value_one_phase,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_eta_min_ratio=self.lr_scheduler_eta_min_ratio,
        )


@dataclass(frozen=True)
class EvalMetrics:
    """Stable eval metric schema for all regimes and splits."""

    log_loss: float
    zero_one_loss: float
    accuracy: float
    num_samples: int


@dataclass(frozen=True)
class PruningSummary:
    """Metadata for the zeroed/frozen phase-2 run."""

    threshold: float
    rows_zeroed_per_layer: tuple[int, ...]
    total_rows_zeroed: int
    rows_still_zero_after_training: bool


@dataclass(frozen=True)
class RegimeResult:
    """Result payload for one regime in one master-seed run."""

    regime: str
    seed_bundle: SeedBundle
    train_metrics: EvalMetrics
    test_metrics: EvalMetrics
    epoch_losses: tuple[float, ...]
    lr_history: tuple[float, ...]
    pruning_summary: PruningSummary | None = None


@dataclass(frozen=True)
class SeedRunResult:
    """Container for all regimes for a single master seed."""

    experiment_name: str
    seed_bundle: SeedBundle
    train_size: int
    eval_size: int
    regimes: dict[str, RegimeResult]
    data_meta: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
