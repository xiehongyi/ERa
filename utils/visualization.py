"""
Visualization utilities (simplified)
Full visualization requires matplotlib configuration
"""

import warnings
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    warnings.warn("Matplotlib/Seaborn not available - plotting disabled")
    PLOTTING_AVAILABLE = False

class Visualizer:
    """Simple visualization tools"""
    
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available")
            
    def plot_perturbation(self, original, perturbation, adversarial):
        """
        Plot perturbation analysis
        
        Args:
            original: Original signal
            perturbation: Perturbation
            adversarial: Adversarial example
        """
        
        if not PLOTTING_AVAILABLE:
            warnings.warn("Cannot plot - matplotlib not available")
            return
            
        try:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Plot original
            axes[0].imshow(np.mean(original, axis=0), cmap='viridis')
            axes[0].set_title('Original')
            
            # Plot perturbation
            axes[1].imshow(np.mean(perturbation, axis=0), cmap='coolwarm')
            axes[1].set_title('Perturbation')
            
            # Plot adversarial
            axes[2].imshow(np.mean(adversarial, axis=0), cmap='viridis')
            axes[2].set_title('Adversarial')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/perturbation_analysis.png')
            plt.close()
            
            print(f"Plot saved to {self.save_dir}/perturbation_analysis.png")
            
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")
            
    def plot_success_rate(self, attack_types, success_rates):
        """
        Plot attack success rates
        
        Args:
            attack_types: List of attack names
            success_rates: List of success rates
        """
        
        if not PLOTTING_AVAILABLE:
            return
            
        try:
            plt.figure(figsize=(8, 6))
            plt.bar(attack_types, success_rates)
            plt.xlabel('Attack Type')
            plt.ylabel('Success Rate (%)')
            plt.title('Attack Performance Comparison')
            plt.ylim(0, 100)
            
            for i, v in enumerate(success_rates):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center')
                
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/success_rates.png')
            plt.close()
            
            print(f"Plot saved to {self.save_dir}/success_rates.png")
            
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")

class ResultAnalyzer:
    """Analyze attack results"""
    
    @staticmethod
    def compute_metrics(original, adversarial):
        """
        Compute attack metrics
        
        Args:
            original: Original samples
            adversarial: Adversarial samples
            
        Returns:
            Dictionary of metrics
        """
        
        metrics = {}
        
        try:
            # L2 norm
            l2_norm = np.linalg.norm(adversarial - original, ord=2)
            metrics['l2_norm'] = float(l2_norm)
            
            # L_inf norm
            linf_norm = np.max(np.abs(adversarial - original))
            metrics['linf_norm'] = float(linf_norm)
            
            # Mean perturbation
            mean_pert = np.mean(np.abs(adversarial - original))
            metrics['mean_perturbation'] = float(mean_pert)
            
            # SNR
            signal_power = np.mean(original ** 2)
            noise_power = np.mean((adversarial - original) ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                metrics['snr_db'] = float(snr)
            else:
                metrics['snr_db'] = float('inf')
                
        except Exception as e:
            warnings.warn(f"Metric computation failed: {e}")
            
        return metrics
    
    @staticmethod
    def generate_report(results, save_path='./results/report.txt'):
        """
        Generate analysis report
        
        Args:
            results: Attack results
            save_path: Where to save report
        """
        
        try:
            with open(save_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("Attack Analysis Report\n")
                f.write("="*60 + "\n\n")
                
                f.write("WARNING: Results generated without optimal configuration\n")
                f.write("See paper for expected performance\n\n")
                
                if 'metrics' in results:
                    f.write("Metrics:\n")
                    for key, value in results['metrics'].items():
                        f.write(f"  {key}: {value}\n")
                        
                f.write("\n" + "="*60 + "\n")
                
            print(f"Report saved to {save_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to save report: {e}")