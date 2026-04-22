"""Typed configuration for PortfolioTrainer.

Fails fast on typos/missing sections/out-of-range values instead of KeyError-ing
deep inside train(). Defined as a small tree of @dataclass objects:

    TrainingConfig
    ├── DataConfig
    ├── EnvironmentConfig
    ├── AgentConfig
    └── TrainingLoopConfig

Agent-specific SB3 kwargs (n_steps, buffer_size, clip_range, ...) are kept as
a passthrough dict in AgentConfig.extra, since the per-algorithm field set
differs between PPO/SAC/A2C.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from environment.constants import DEFAULT_INITIAL_BALANCE, DEFAULT_TRANSACTION_COST
from environment.rewards import RewardFunctionFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    """Market-data parameters."""

    tickers: List[str] = field(
        default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    )
    train_days: int = 730
    val_days: int = 180

    def __post_init__(self):
        if not self.tickers:
            raise ValueError("data.tickers must be non-empty")
        if self.train_days <= 0:
            raise ValueError(f"data.train_days must be > 0, got {self.train_days}")
        if self.val_days <= 0:
            raise ValueError(f"data.val_days must be > 0, got {self.val_days}")


@dataclass
class EnvironmentConfig:
    """PortfolioEnv construction parameters."""

    initial_balance: float = DEFAULT_INITIAL_BALANCE
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    window_size: int = 20
    reward_function: str = 'sharpe'
    reward_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.initial_balance <= 0:
            raise ValueError(
                f"environment.initial_balance must be > 0, got {self.initial_balance}"
            )
        if not 0 <= self.transaction_cost < 1:
            raise ValueError(
                f"environment.transaction_cost must be in [0, 1), "
                f"got {self.transaction_cost}"
            )
        if self.window_size <= 0:
            raise ValueError(
                f"environment.window_size must be > 0, got {self.window_size}"
            )
        available = RewardFunctionFactory.list_available()
        if self.reward_function not in available:
            raise ValueError(
                f"environment.reward_function must be one of {sorted(available)}, "
                f"got {self.reward_function!r}"
            )


@dataclass
class AgentConfig:
    """SB3 agent parameters.

    Only `algorithm` and `policy` are structurally validated; every other
    key in the YAML `agent:` section is forwarded verbatim to the SB3
    constructor via `extra`. This keeps the schema tolerant of PPO/SAC/A2C
    divergence while still catching typos in the top-level agent key.
    """

    algorithm: str = 'PPO'
    policy: str = 'MlpPolicy'
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AgentConfig':
        d = dict(d)  # copy so we don't mutate the caller's dict
        return cls(
            algorithm=d.pop('algorithm', 'PPO'),
            policy=d.pop('policy', 'MlpPolicy'),
            extra=d,
        )


@dataclass
class TrainingLoopConfig:
    """Training-loop parameters (timesteps, eval cadence, output paths)."""

    total_timesteps: int = 100_000
    eval_freq: int = 5_000
    save_freq: int = 10_000
    log_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / 'logs'))
    model_dir: str = field(
        default_factory=lambda: str(PROJECT_ROOT / 'training' / 'models')
    )

    def __post_init__(self):
        if self.total_timesteps <= 0:
            raise ValueError(
                f"training.total_timesteps must be > 0, got {self.total_timesteps}"
            )
        if self.eval_freq <= 0:
            raise ValueError(f"training.eval_freq must be > 0, got {self.eval_freq}")
        if self.save_freq <= 0:
            raise ValueError(f"training.save_freq must be > 0, got {self.save_freq}")


@dataclass
class TrainingConfig:
    """Top-level config for PortfolioTrainer."""

    data: DataConfig = field(default_factory=DataConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    _ALLOWED_SECTIONS = frozenset({'data', 'environment', 'agent', 'training'})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        unknown = set(d.keys()) - cls._ALLOWED_SECTIONS
        if unknown:
            raise ValueError(
                f"Unknown top-level config keys: {sorted(unknown)}. "
                f"Allowed: {sorted(cls._ALLOWED_SECTIONS)}"
            )
        return cls(
            data=DataConfig(**d.get('data', {})),
            environment=EnvironmentConfig(**d.get('environment', {})),
            agent=AgentConfig.from_dict(d.get('agent', {})),
            training=TrainingLoopConfig(**d.get('training', {})),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        """Dict form for logging as hyperparameters.

        Flattens AgentConfig.extra back up into the agent dict so the
        representation matches the YAML schema.
        """
        d = asdict(self)
        agent = d['agent']
        extra = agent.pop('extra', {})
        agent.update(extra)
        return d
