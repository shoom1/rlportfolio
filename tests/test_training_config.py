"""Tests for the typed TrainingConfig loader/validator."""

import pytest
import yaml

from training.config import (
    AgentConfig,
    DataConfig,
    EnvironmentConfig,
    TrainingConfig,
    TrainingLoopConfig,
    UniverseConfig,
)


class TestDefaults:
    def test_default_trainingconfig_is_valid(self):
        cfg = TrainingConfig()
        assert cfg.data.tickers  # non-empty default
        assert cfg.environment.reward_function == 'sharpe'
        assert cfg.agent.algorithm == 'PPO'
        assert cfg.training.total_timesteps > 0


class TestDataConfigValidation:
    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers"):
            DataConfig(tickers=[])

    def test_non_positive_train_days_raises(self):
        with pytest.raises(ValueError, match="train_days"):
            DataConfig(tickers=['A'], train_days=0)

    def test_non_positive_val_days_raises(self):
        with pytest.raises(ValueError, match="val_days"):
            DataConfig(tickers=['A'], val_days=-1)


class TestUniverseConfigValidation:
    def test_default_universe_is_static_current_with_known_bias(self):
        cfg = UniverseConfig()
        assert cfg.mode == 'static_current'
        assert cfg.survivorship_bias == 'known'

    def test_static_current_accepts_explicit_known_bias(self):
        cfg = UniverseConfig(mode='static_current', survivorship_bias='known')
        assert cfg.survivorship_bias == 'known'

    def test_rejects_point_in_time_index_until_implemented(self):
        with pytest.raises(ValueError, match="point_in_time_index"):
            UniverseConfig(mode='point_in_time_index')

    def test_rejects_membership_file_until_point_in_time_is_implemented(self):
        with pytest.raises(ValueError, match="membership_file"):
            UniverseConfig(membership_file='data/sp500_membership.csv')

    def test_rejects_static_current_with_controlled_bias_claim(self):
        with pytest.raises(ValueError, match="survivorship_bias"):
            UniverseConfig(mode='static_current', survivorship_bias='controlled')

    def test_metadata_labels_static_current_universe(self):
        metadata = UniverseConfig().metadata(['MSFT', 'AAPL'])
        assert metadata['universe_mode'] == 'static_current'
        assert metadata['survivorship_bias'] == 'known'
        assert metadata['n_assets'] == 2
        assert metadata['tickers'] == 'MSFT,AAPL'
        assert len(metadata['tickers_hash']) == 12


class TestEnvironmentConfigValidation:
    def test_non_positive_initial_balance_raises(self):
        with pytest.raises(ValueError, match="initial_balance"):
            EnvironmentConfig(initial_balance=0)

    def test_transaction_cost_out_of_range_raises(self):
        with pytest.raises(ValueError, match="transaction_cost"):
            EnvironmentConfig(transaction_cost=-0.01)
        with pytest.raises(ValueError, match="transaction_cost"):
            EnvironmentConfig(transaction_cost=1.0)

    def test_zero_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            EnvironmentConfig(window_size=0)

    def test_unknown_reward_function_raises(self):
        with pytest.raises(ValueError, match="reward_function"):
            EnvironmentConfig(reward_function='not_a_real_reward')

    def test_all_registered_rewards_accepted(self):
        from environment.rewards import RewardFunctionFactory

        for name in RewardFunctionFactory.list_available():
            cfg = EnvironmentConfig(reward_function=name)
            assert cfg.reward_function == name


class TestTrainingLoopConfigValidation:
    def test_non_positive_total_timesteps_raises(self):
        with pytest.raises(ValueError, match="total_timesteps"):
            TrainingLoopConfig(total_timesteps=0)

    def test_non_positive_eval_freq_raises(self):
        with pytest.raises(ValueError, match="eval_freq"):
            TrainingLoopConfig(eval_freq=0)

    def test_non_positive_save_freq_raises(self):
        with pytest.raises(ValueError, match="save_freq"):
            TrainingLoopConfig(save_freq=-1)


class TestAgentConfigFromDict:
    def test_unknown_kwargs_flow_into_extra(self):
        agent = AgentConfig.from_dict({
            'algorithm': 'SAC',
            'policy': 'MlpPolicy',
            'buffer_size': 100000,
            'tau': 0.005,
        })
        assert agent.algorithm == 'SAC'
        assert agent.extra == {'buffer_size': 100000, 'tau': 0.005}

    def test_empty_dict_uses_defaults(self):
        agent = AgentConfig.from_dict({})
        assert agent.algorithm == 'PPO'
        assert agent.policy == 'MlpPolicy'
        assert agent.extra == {}


class TestFromDict:
    def test_unknown_top_level_section_raises(self):
        with pytest.raises(ValueError, match="Unknown top-level"):
            TrainingConfig.from_dict({
                'data': {'tickers': ['A']},
                'typo_section': {},
            })

    def test_missing_sections_use_defaults(self):
        cfg = TrainingConfig.from_dict({
            'data': {'tickers': ['AAPL']},
        })
        assert cfg.environment.reward_function == 'sharpe'
        assert cfg.agent.algorithm == 'PPO'
        assert cfg.data.universe.mode == 'static_current'
        assert cfg.data.universe.survivorship_bias == 'known'

    def test_empty_dict_uses_all_defaults(self):
        """Empty dict is treated as all defaults (including default tickers)."""
        cfg = TrainingConfig.from_dict({})
        assert cfg.data.tickers  # default ticker list is non-empty

    def test_nested_universe_config_parses(self):
        cfg = TrainingConfig.from_dict({
            'data': {
                'tickers': ['AAPL', 'MSFT'],
                'universe': {
                    'mode': 'static_current',
                    'survivorship_bias': 'known',
                },
            },
        })
        assert cfg.data.universe.mode == 'static_current'
        assert cfg.data.universe.survivorship_bias == 'known'


class TestFromYaml:
    @pytest.mark.parametrize("yaml_path", [
        "configs/default_config.yaml",
        "configs/sac_config.yaml",
        "configs/a2c_config.yaml",
        "configs/tech_stocks_config.yaml",
    ])
    def test_shipped_configs_parse(self, yaml_path):
        """All four shipped configs must parse and validate cleanly."""
        cfg = TrainingConfig.from_yaml(yaml_path)
        assert cfg.data.tickers
        assert cfg.agent.algorithm in {'PPO', 'SAC', 'A2C'}

    def test_yaml_roundtrip(self, tmp_path):
        cfg = TrainingConfig()
        p = tmp_path / "rt.yaml"
        with open(p, 'w') as f:
            yaml.safe_dump(cfg.to_dict(), f)
        cfg2 = TrainingConfig.from_yaml(p)
        assert cfg2.data.tickers == cfg.data.tickers
        assert cfg2.data.universe.mode == cfg.data.universe.mode
        assert cfg2.data.universe.survivorship_bias == cfg.data.universe.survivorship_bias
        assert cfg2.environment.reward_function == cfg.environment.reward_function
        assert cfg2.agent.algorithm == cfg.agent.algorithm


class TestToDict:
    def test_agent_extra_flattens_back_into_agent_dict(self):
        cfg = TrainingConfig.from_dict({
            'agent': {'algorithm': 'SAC', 'buffer_size': 50000},
        })
        d = cfg.to_dict()
        assert d['agent']['algorithm'] == 'SAC'
        assert d['agent']['buffer_size'] == 50000
        assert 'extra' not in d['agent']

    def test_universe_config_roundtrips_to_dict(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        assert d['data']['universe']['mode'] == 'static_current'
        assert d['data']['universe']['survivorship_bias'] == 'known'
