"""Tests for evaluation.visualize_network utility helpers."""

from evaluation.visualize_network import analyze_network_from_config


def test_analyze_network_from_config_prints_layer_numbers(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  algorithm: PPO
  policy: MlpPolicy
  policy_kwargs:
    net_arch:
      - 32
      - 16
""".lstrip()
    )

    analyze_network_from_config(str(config_path))

    out = capsys.readouterr().out
    assert "Layer 1: ~" in out
    assert "Layer 2: ~" in out
