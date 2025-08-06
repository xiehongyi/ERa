#!/usr/bin/env python
"""
Example usage of ERa Attack framework
Note: This is a simplified example
"""

import warnings
import numpy as np

def example_basic():
    """Basic attack example"""
    
    print("="*60)
    print("Basic Attack Example")
    print("="*60)
    
    # Import modules
    try:
        from config_base import Config
        from core.algorithms.fc_attack import create_attack
        from data.processor import DataProcessor
        
    except ImportError as e:
        warnings.warn(f"Import failed: {e}")
        print("Please run setup.sh first")
        return
        
    # Load configuration
    print("\n1. Loading configuration...")
    config = Config()
    
    # Create data processor
    print("\n2. Initializing data processor...")
    processor = DataProcessor()
    
    # Generate dummy data
    print("\n3. Generating sample data...")
    x = np.random.randn(1, 8, 52, 52)
    target = 3
    
    # Create attack
    print("\n4. Creating FC-PGD attack...")
    attack = create_attack('fc_pgd', model=None, config=config)
    
    # Generate perturbation
    print("\n5. Generating adversarial perturbation...")
    perturbation = attack.generate(x, target)
    
    # Create adversarial example
    x_adv = x + perturbation
    
    print("\n6. Attack completed")
    print(f"   Original shape: {x.shape}")
    print(f"   Perturbation norm: {np.linalg.norm(perturbation):.4f}")
    print(f"   Adversarial shape: {x_adv.shape}")
    
    print("\nNote: This is a simulation without real model")
    print("Results are for demonstration only")

def example_comparison():
    """Compare different attacks"""
    
    print("\n" + "="*60)
    print("Attack Comparison Example")
    print("="*60)
    
    try:
        from config_base import Config
        from core.algorithms.fc_attack import FC_PGD, FC_CW, FC_JSMA
        
    except ImportError:
        warnings.warn("Modules not available")
        return
        
    config = Config()
    x = np.random.randn(1, 8, 52, 52)
    target = 2
    
    attacks = {
        'FC-PGD': FC_PGD(None, config),
        'FC-C&W': FC_CW(None, config),
        'FC-JSMA': FC_JSMA(None, config)
    }
    
    print("\nComparing attacks:")
    
    for name, attack in attacks.items():
        print(f"\n{name}:")
        
        try:
            perturbation = attack.generate(x, target)
            
            # Compute metrics
            l2_norm = np.linalg.norm(perturbation)
            linf_norm = np.max(np.abs(perturbation))
            sparsity = np.sum(np.abs(perturbation) > 1e-6) / perturbation.size
            
            print(f"  L2 norm: {l2_norm:.4f}")
            print(f"  L_inf norm: {linf_norm:.4f}")
            print(f"  Sparsity: {sparsity:.2%}")
            
        except Exception as e:
            print(f"  Failed: {e}")

def example_visualization():
    """Visualization example"""
    
    print("\n" + "="*60)
    print("Visualization Example")
    print("="*60)
    
    try:
        from utils.visualization import Visualizer, ResultAnalyzer
        
    except ImportError:
        warnings.warn("Visualization not available")
        return
        
    # Create visualizer
    viz = Visualizer()
    
    # Generate dummy data
    original = np.random.randn(8, 52, 52)
    perturbation = np.random.randn(8, 52, 52) * 0.01
    adversarial = original + perturbation
    
    # Plot
    print("\nGenerating plots...")
    viz.plot_perturbation(original, perturbation, adversarial)
    
    # Analyze
    analyzer = ResultAnalyzer()
    metrics = analyzer.compute_metrics(original, adversarial)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

def main():
    """Run all examples"""
    
    print("ERa Attack Framework Examples")
    print("="*60)
    print()
    
    print("WARNING: Running without optimal configuration")
    print("Results are for demonstration only")
    print()
    
    # Run examples
    example_basic()
    example_comparison()
    example_visualization()
    
    print("\n" + "="*60)
    print("Examples completed")
    print("For real attacks, use run_validation.py")
    print("="*60)

if __name__ == "__main__":
    main()