"""
Utilities for visualizing neural network architecture.
"""

import torch
import numpy as np
from stable_baselines3 import PPO, SAC, A2C
from pathlib import Path
from typing import Optional, Union


def print_network_structure(model_path: str):
    """
    Print detailed network structure from a trained model.

    Args:
        model_path: Path to saved model (.zip file)
    """
    print("\n" + "="*80)
    print("NEURAL NETWORK STRUCTURE")
    print("="*80)

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Try to load as different algorithms
    model = None
    for algo_class in [PPO, SAC, A2C]:
        try:
            model = algo_class.load(model_path)
            print(f"\nAlgorithm: {algo_class.__name__}")
            break
        except:
            continue

    if model is None:
        print("Error: Could not load model")
        return

    # Print policy network
    print("\n" + "-"*80)
    print("POLICY NETWORK (Actor)")
    print("-"*80)
    print(model.policy)

    # Get detailed layer information
    print("\n" + "-"*80)
    print("DETAILED LAYER BREAKDOWN")
    print("-"*80)

    # Get the policy network architecture
    if hasattr(model.policy, 'mlp_extractor'):
        print("\n[Shared Feature Extractor]")
        print(model.policy.mlp_extractor)

    if hasattr(model.policy, 'action_net'):
        print("\n[Actor Head (Action Network)]")
        print(model.policy.action_net)

    if hasattr(model.policy, 'value_net'):
        print("\n[Critic Head (Value Network)]")
        print(model.policy.value_net)

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    print("\n" + "-"*80)
    print("PARAMETER COUNT")
    print("-"*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print parameters per layer
    print("\n[Parameters by Layer]")
    for name, param in model.policy.named_parameters():
        print(f"{name:60s} {param.shape} ({param.numel():,} params)")

    print("\n" + "="*80)


def print_layer_dimensions(model_path: str):
    """
    Print dimensions of each layer in a simple format.

    Args:
        model_path: Path to saved model
    """
    print("\n" + "="*80)
    print("LAYER DIMENSIONS")
    print("="*80)

    # Load model
    for algo_class in [PPO, SAC, A2C]:
        try:
            model = algo_class.load(model_path)
            break
        except:
            continue

    print("\nArchitecture Flow:")
    print("-"*80)

    layer_num = 1
    for name, module in model.policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Layer {layer_num}: Linear({module.in_features} → {module.out_features})")
            layer_num += 1
        elif isinstance(module, torch.nn.ReLU):
            print(f"         ReLU()")
        elif isinstance(module, torch.nn.Tanh):
            print(f"         Tanh()")

    print("-"*80)


def visualize_network_graphviz(model_path: str, output_path: str = "network_graph.png"):
    """
    Create a visual graph of the network using graphviz/torchviz.

    Args:
        model_path: Path to saved model
        output_path: Where to save the visualization

    Note: Requires: pip install torchviz graphviz
    """
    try:
        from torchviz import make_dot
    except ImportError:
        print("Error: torchviz not installed. Install with: pip install torchviz graphviz")
        return

    print(f"\nGenerating network graph...")

    # Load model
    for algo_class in [PPO, SAC, A2C]:
        try:
            model = algo_class.load(model_path)
            break
        except:
            continue

    # Create dummy input
    obs_space = model.observation_space
    dummy_input = torch.randn(1, obs_space.shape[0])

    # Forward pass
    with torch.no_grad():
        if hasattr(model.policy, 'forward'):
            output = model.policy.forward(dummy_input)
        else:
            output = model.policy(dummy_input)

    # Create graph
    if isinstance(output, tuple):
        output = output[0]

    dot = make_dot(output, params=dict(model.policy.named_parameters()))
    dot.render(output_path.replace('.png', ''), format='png', cleanup=True)

    print(f"Network graph saved to: {output_path}")


def create_ascii_diagram(model_path: str):
    """
    Create an ASCII art diagram of the network.

    Args:
        model_path: Path to saved model
    """
    print("\n" + "="*80)
    print("NETWORK ARCHITECTURE DIAGRAM")
    print("="*80)

    # Load model
    for algo_class in [PPO, SAC, A2C]:
        try:
            model = algo_class.load(model_path)
            break
        except:
            continue

    # Get observation and action dimensions
    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]

    # Extract layer dimensions
    layers = []
    for name, module in model.policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append((module.in_features, module.out_features))

    # Draw ASCII diagram
    print(f"\nInput: {obs_dim} dimensions")
    print("   │")
    print("   ▼")

    # Shared layers
    for i, (in_feat, out_feat) in enumerate(layers[:-2] if len(layers) > 2 else layers):
        print(f"┌─────────────────────┐")
        print(f"│  Dense({in_feat:4d} → {out_feat:4d}) │")
        print(f"└─────────────────────┘")
        print("          │")
        print("          ▼")
        print("      [ ReLU ]")
        print("          │")
        print("          ▼")

    # Split for actor and critic
    if len(layers) >= 2:
        print("          │")
        print("    ┌─────┴─────┐")
        print("    ▼           ▼")
        print("┌─────────┐ ┌─────────┐")
        print(f"│  Actor  │ │  Critic │")
        print(f"│ (π net) │ │ (V net) │")
        print("└─────────┘ └─────────┘")
        print("     │           │")
        print("     ▼           ▼")
        print(f"  {act_dim} outputs  1 output")
        print(f"  (weights)   (value)")

    print("\n" + "="*80)


def analyze_network_from_config(config_path: str):
    """
    Analyze network architecture from a config file (before training).

    Args:
        config_path: Path to YAML config file
    """
    import yaml

    print("\n" + "="*80)
    print("NETWORK CONFIGURATION ANALYSIS")
    print("="*80)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract network info
    policy_kwargs = config.get('agent', {}).get('policy_kwargs', {})
    net_arch = policy_kwargs.get('net_arch', [64, 64])

    print(f"\nAlgorithm: {config.get('agent', {}).get('algorithm', 'N/A')}")
    print(f"Policy: {config.get('agent', {}).get('policy', 'N/A')}")
    print(f"\nNetwork Architecture: {net_arch}")

    # Calculate approximate parameter count
    # This is an estimate - actual count depends on input/output dims
    print("\n[Architecture Details]")
    for i, layer_size in enumerate(net_arch):
        print(f"Hidden Layer {i+1}: {layer_size} neurons")

    # Estimate parameters (rough calculation)
    # Assuming input dim ~1300, output dim ~5
    estimated_params = 0
    prev_size = 1300  # approximate input size

    for layer_size in net_arch:
        params = (prev_size + 1) * layer_size  # +1 for bias
        estimated_params += params
        print(f"  Layer {len(estimated_params):d}: ~{params:,} parameters")
        prev_size = layer_size

    # Output layer (actor)
    params = (prev_size + 1) * 5  # 5 assets
    estimated_params += params
    print(f"  Actor head: ~{params:,} parameters")

    # Output layer (critic)
    params = (prev_size + 1) * 1
    estimated_params += params
    print(f"  Critic head: ~{params:,} parameters")

    print(f"\nEstimated total parameters: ~{estimated_params:,}")
    print("\n(Note: Actual count depends on observation/action space dimensions)")

    print("\n" + "="*80)


def main():
    """Main function with examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize neural network architecture")
    parser.add_argument('--model', type=str, help='Path to trained model (.zip)')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml)')
    parser.add_argument('--output', type=str, default='network_graph.png',
                       help='Output path for graph visualization')
    parser.add_argument('--graph', action='store_true',
                       help='Generate graphviz visualization (requires torchviz)')

    args = parser.parse_args()

    if args.model:
        print(f"Analyzing model: {args.model}\n")

        # Print detailed structure
        print_network_structure(args.model)

        # Print layer dimensions
        print_layer_dimensions(args.model)

        # Create ASCII diagram
        create_ascii_diagram(args.model)

        # Generate graph if requested
        if args.graph:
            visualize_network_graphviz(args.model, args.output)

    elif args.config:
        print(f"Analyzing config: {args.config}\n")
        analyze_network_from_config(args.config)

    else:
        print("Please provide either --model or --config")
        print("\nExamples:")
        print("  python evaluation/visualize_network.py --model training/models/best/best_model.zip")
        print("  python evaluation/visualize_network.py --config configs/ppo_config.yaml")
        print("  python evaluation/visualize_network.py --model model.zip --graph")


if __name__ == '__main__':
    main()
