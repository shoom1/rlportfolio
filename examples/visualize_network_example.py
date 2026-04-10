"""
Example: How to visualize neural network architecture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.visualize_network import (
    print_network_structure,
    print_layer_dimensions,
    create_ascii_diagram,
    analyze_network_from_config
)


def example_1_analyze_config():
    """Example 1: Analyze architecture from config file (before training)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Analyze Network from Config File")
    print("="*80)

    config_path = '../configs/ppo_config.yaml'

    try:
        analyze_network_from_config(config_path)
    except FileNotFoundError:
        print(f"\nConfig file not found at: {config_path}")
        print("Create a config file first or adjust the path.")


def example_2_analyze_trained_model():
    """Example 2: Analyze a trained model."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Analyze Trained Model")
    print("="*80)

    # You need to train a model first
    model_path = '../training/models/best/best_model.zip'

    print(f"\nLooking for model at: {model_path}")
    print("\nTo use this example:")
    print("1. Train a model first:")
    print("   python training/train.py --config configs/ppo_config.yaml")
    print("2. Then run this script again")

    try:
        # Print detailed structure
        print_network_structure(model_path)

        # Print layer dimensions
        print_layer_dimensions(model_path)

        # Create ASCII diagram
        create_ascii_diagram(model_path)

    except FileNotFoundError:
        print(f"\nModel not found. Train a model first!")


def example_3_quick_visualization():
    """Example 3: Quick ASCII visualization."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Quick ASCII Architecture Diagram")
    print("="*80)

    print("""
This is what a typical PPO network looks like:

Input: 1,306 dimensions (20 steps × 5 assets × 13 features + 5 weights + 1 cash)
   │
   ▼
┌─────────────────────┐
│  Dense(1306 → 256)  │  ← First hidden layer
└─────────────────────┘
          │
          ▼
      [ ReLU ]           ← Activation function
          │
          ▼
┌─────────────────────┐
│  Dense(256 → 256)   │  ← Second hidden layer
└─────────────────────┘
          │
          ▼
      [ ReLU ]
          │
          ▼
┌─────────────────────┐
│  Dense(256 → 128)   │  ← Third hidden layer
└─────────────────────┘
          │
          ▼
      [ ReLU ]
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌─────────┐
│  Actor  │ │  Critic │  ← Split into two heads
│ (π net) │ │ (V net) │
└─────────┘ └─────────┘
     │           │
     ▼           ▼
  5 outputs   1 output   ← Actor outputs portfolio weights,
  (weights)   (value)       Critic outputs state value

Total Parameters: ~400,000 - 500,000 (depending on exact dimensions)
    """)


def example_4_parameter_count():
    """Example 4: Understanding parameter count."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Understanding Parameter Count")
    print("="*80)

    print("""
For a network with architecture [256, 256, 128]:

Layer 1: (1,306 × 256) + 256 = 334,592 parameters
         (input × neurons) + bias

Layer 2: (256 × 256) + 256 = 65,792 parameters

Layer 3: (256 × 128) + 128 = 32,896 parameters

Actor head: (128 × 5) + 5 = 645 parameters

Critic head: (128 × 1) + 1 = 129 parameters

TOTAL: ~434,000 parameters

This is relatively small compared to modern deep learning models!
For comparison:
- ResNet-50: ~25 million parameters
- GPT-3: 175 billion parameters

The network is intentionally kept simple because:
1. Portfolio optimization is a relatively simple task
2. Smaller networks train faster
3. Less prone to overfitting
4. Can run on CPU efficiently
    """)


def example_5_custom_architecture():
    """Example 5: How to create custom architectures."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Architecture Examples")
    print("="*80)

    print("""
You can customize the architecture in your config file:

1. DEEPER NETWORK (more layers):
   policy_kwargs:
     net_arch: [512, 512, 256, 256, 128]

2. WIDER NETWORK (more neurons per layer):
   policy_kwargs:
     net_arch: [1024, 1024, 512]

3. SEPARATE ACTOR AND CRITIC:
   policy_kwargs:
     net_arch:
       - pi: [256, 256]    # Actor has 2 layers of 256
         vf: [512, 512]    # Critic has 2 layers of 512

4. WITH DIFFERENT ACTIVATION:
   policy_kwargs:
     net_arch: [256, 256, 128]
     activation_fn: torch.nn.Tanh  # Instead of ReLU

5. SHARED THEN SEPARATE:
   policy_kwargs:
     net_arch:
       - 256                # Shared layer
       - 256                # Shared layer
       - pi: [128, 64]      # Actor-specific layers
         vf: [128, 64]      # Critic-specific layers

Recommendations:
- Start with default [256, 256, 128]
- If underfitting: Make network larger
- If overfitting: Make network smaller or add regularization
- If training is slow: Use smaller network
- Use W&B sweeps to find optimal architecture!
    """)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("NEURAL NETWORK VISUALIZATION EXAMPLES")
    print("="*80)

    # Run examples
    example_1_analyze_config()
    example_3_quick_visualization()
    example_4_parameter_count()
    example_5_custom_architecture()

    # Conditional examples (require trained model)
    # example_2_analyze_trained_model()

    print("\n" + "="*80)
    print("COMMAND-LINE USAGE")
    print("="*80)
    print("""
To visualize from command line:

# Analyze a config file
python evaluation/visualize_network.py --config configs/ppo_config.yaml

# Analyze a trained model
python evaluation/visualize_network.py --model training/models/best/best_model.zip

# Generate graphical visualization (requires: pip install torchviz graphviz)
python evaluation/visualize_network.py --model training/models/best/best_model.zip --graph

# The graph will be saved as network_graph.png
    """)

    print("\n" + "="*80)
    print("PROGRAMMATIC USAGE")
    print("="*80)
    print("""
In your Python code:

    from evaluation.visualize_network import (
        print_network_structure,
        create_ascii_diagram,
        analyze_network_from_config
    )

    # Analyze config
    analyze_network_from_config('configs/ppo_config.yaml')

    # Analyze trained model
    print_network_structure('training/models/best/best_model.zip')

    # Create ASCII diagram
    create_ascii_diagram('training/models/best/best_model.zip')
    """)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
