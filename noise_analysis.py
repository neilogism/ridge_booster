from boosting_comparison import BoostingComparison
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def process_results(results, noise_values, lambda_values, learning_rates):
    """
    Process the results dictionary to extract best and average case performance for each noise level
    """
    processed = {
        'ridge_exact': {'noise': [], 'train_mse_avg': [], 'val_mse_avg': [], 'overfitting_avg': [],  'train_mse_best': [], 'val_mse_best': [], 'overfitting_best': [], 'overfitting_at_best_val': [], 'overfitting': [], 'val_mse': []},
        'fixed_lr': {'noise': [], 'train_mse_avg': [], 'val_mse_avg': [], 'overfitting_avg': [],  'train_mse_best': [], 'val_mse_best': [], 'overfitting_best': [], 'overfitting_at_best_val': [], 'overfitting': [], 'val_mse': []}
    }
    
    # Group results by noise level for ridge_exact
    ridge_by_noise = defaultdict(list)
    for result in results['ridge_exact']:
        ridge_by_noise[result['noise']].append(result)
    
    # Find best lambda for each noise level (minimize validation MSE)
    for noise in noise_values:
        if noise in ridge_by_noise:
            best_result = min(ridge_by_noise[noise], key=lambda x: x['val_mse'])
            val_mse = [x['val_mse'] for x in ridge_by_noise[noise]]
            avg_val_mse=sum(val_mse)/len(val_mse)
            avg_train_mse = sum([x['train_mse'] for x in ridge_by_noise[noise]])/len([x['train_mse'] for x in ridge_by_noise[noise]])
            overfitting = [x['val_mse'] - x['train_mse'] for x in ridge_by_noise[noise]]
            avg_overfitting = sum(overfitting)/len(overfitting)            
            processed['ridge_exact']['noise'].append(noise)
            processed['ridge_exact']['train_mse_best'].append(best_result['train_mse'])
            processed['ridge_exact']['val_mse_best'].append(best_result['val_mse'])
            processed['ridge_exact']['overfitting_at_best_val'].append(
                best_result['val_mse'] - best_result['train_mse']
            )
            best_overfitting = min(overfitting)
            processed['ridge_exact']['overfitting_best'].append(best_overfitting)
            processed['ridge_exact']['train_mse_avg'].append(avg_train_mse)
            processed['ridge_exact']['val_mse_avg'].append(avg_val_mse)
            processed['ridge_exact']['overfitting_avg'].append(
                avg_overfitting
            )
            processed['ridge_exact']['overfitting'].append(overfitting)
            processed['ridge_exact']['val_mse'].append(val_mse)


    # Group results by noise level for fixed_lr
    lr_by_noise = defaultdict(list)
    for result in results['fixed_lr']:
        lr_by_noise[result['noise']].append(result)
    
    # Find best learning rate for each noise level
    for noise in noise_values:
        if noise in lr_by_noise:
            best_result = min(lr_by_noise[noise], key=lambda x: x['val_mse'])
            processed['fixed_lr']['noise'].append(noise)
            processed['fixed_lr']['train_mse_best'].append(best_result['train_mse'])
            processed['fixed_lr']['val_mse_best'].append(best_result['val_mse'])
            best_overfitting = min([x['val_mse'] - x['train_mse'] for x in ridge_by_noise[noise]])
            processed['fixed_lr']['overfitting_best'].append(
                best_overfitting
            )
            val_mse = [x['val_mse'] for x in lr_by_noise[noise]]
            avg_val_mse=sum(val_mse)/len(val_mse)
            avg_train_mse = sum([x['train_mse'] for x in lr_by_noise[noise]])/len([x['train_mse'] for x in lr_by_noise[noise]])
            overfitting = [x['val_mse'] - x['train_mse'] for x in lr_by_noise[noise]]
            avg_overfitting = sum(overfitting)/len(overfitting)
            processed['fixed_lr']['overfitting_at_best_val'].append(
                best_result['val_mse'] - best_result['train_mse']
            )
            processed['fixed_lr']['val_mse_avg'].append(avg_val_mse)
            processed['fixed_lr']['train_mse_avg'].append(avg_train_mse)
            processed['fixed_lr']['overfitting_avg'].append(
                avg_overfitting
            )
            processed['fixed_lr']['overfitting'].append(overfitting)
            processed['fixed_lr']['val_mse'].append(val_mse)

    return processed

def create_noise_analysis_plots(results, noise_values, lambda_values, learning_rates):
    """
    Create three plots analyzing the effect of noise on model performance
    """
    # Process results to get best performance at each noise level
    processed = process_results(results, noise_values, lambda_values, learning_rates)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 5))
    
    # Top row: Best Case Performance
    # Plot 1: Validation MSE vs Noise (Best Case)
    axes[0, 0].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['val_mse_best'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[0, 0].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['val_mse_best'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Noise Level')
    axes[0, 0].set_ylabel('Validation MSE')
    axes[0, 0].set_title('Validation MSE vs Noise Level (Best Case)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training MSE vs Noise (Best Case)
    axes[0, 1].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['train_mse_best'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[0, 1].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['train_mse_best'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('Training MSE')
    axes[0, 1].set_title('Training MSE vs Noise Level (Best Case)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overfitting vs Noise (Best Case)
    axes[0, 2].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['overfitting_best'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[0, 2].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['overfitting_best'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Noise Level')
    axes[0, 2].set_ylabel('Overfitting (Val MSE - Train MSE)')
    axes[0, 2].set_title('Overfitting vs Noise Level (Best Case)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    
    # Bottom row: Average Case Performance
    # Plot 4: Validation MSE vs Noise (Average Case)
    axes[1, 0].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['val_mse_avg'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[1, 0].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['val_mse_avg'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Noise Level')
    axes[1, 0].set_ylabel('Validation MSE')
    axes[1, 0].set_title('Validation MSE vs Noise Level (Avg. Case)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Training MSE vs Noise (Average Case)
    axes[1, 1].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['train_mse_avg'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[1, 1].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['train_mse_avg'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Noise Level')
    axes[1, 1].set_ylabel('Training MSE')
    axes[1, 1].set_title('Training MSE vs Noise Level (Avg. Case)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Overfitting vs Noise (Average Case)
    axes[1, 2].loglog(processed['ridge_exact']['noise'], processed['ridge_exact']['overfitting_avg'], 
                      'o-', label='Ridge Exact', linewidth=2, markersize=6)
    axes[1, 2].loglog(processed['fixed_lr']['noise'], processed['fixed_lr']['overfitting_avg'], 
                      's-', label='Fixed LR', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Noise Level')
    axes[1, 2].set_ylabel('Overfitting (Val MSE - Train MSE)')
    axes[1, 2].set_title('Overfitting vs Noise Level (Avg. Case)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
    
# Additional analysis: Statistical comparison
def compare_methods(results, noise_values):
    """Compare ridge vs fixed LR across noise levels"""
    processed = process_results(results, noise_values, lambda_values, learning_rates)
    
    ridge_data = processed['ridge_exact']
    lr_data = processed['fixed_lr']
    
    # Ensure we have the same noise values for comparison
    common_noise = set(ridge_data['noise']) & set(lr_data['noise'])
    
    ridge_better_best_case_count = 0
    ridge_better_avg_case_count = 0

    total_comparisons = 0
    
    for noise in common_noise:
        ridge_idx = ridge_data['noise'].index(noise)
        lr_idx = lr_data['noise'].index(noise)
        
        ridge_val_best = ridge_data['val_mse_best'][ridge_idx]
        lr_val_best = lr_data['val_mse_best'][lr_idx]
        
        ridge_val_avg = ridge_data['val_mse_avg'][ridge_idx]
        lr_val_avg = lr_data['val_mse_avg'][lr_idx]
        
        if ridge_val_best < lr_val_best:
            ridge_better_best_case_count += 1
        
        if ridge_val_avg < lr_val_avg:
            ridge_better_avg_case_count += 1
            
        total_comparisons += 1
    
    print(f"\nMethod Comparison:")
    print(f"Best-case ridge exact performs better in {ridge_better_best_case_count}/{total_comparisons} "
        f"({100*ridge_better_best_case_count/total_comparisons:.1f}%) of noise levels")
    
    print(f"Average-case ridge exact performs better in {ridge_better_avg_case_count}/{total_comparisons} "
        f"({100*ridge_better_avg_case_count/total_comparisons:.1f}%) of noise levels")


if __name__ == "__main__":
    features = [100]
    noise = np.logspace(1, 4, 20)
    samples = [1000]
    lambda_values = np.logspace(2, 6, 6)
    learning_rates = [i/1e6 for i in lambda_values]
    results = {
        "ridge_exact": [],
        'fixed_lr': [],
    }
    for i in features:
        for j in samples:
            for k in noise:
                print(f"features: {i}, samples: {j}, noise: {k}")
                comparison = BoostingComparison(
                    n_estimators=30, 
                    max_depth=3, 
                    random_state=42,
                    n_features=int(i),
                    n_samples=int(j),
                    noise=float(k)
                )

                X_train, X_val, y_train, y_val = comparison.load_data('synthetic')

                for lambda_reg in lambda_values:
                    result = comparison.ridge_boosting(X_train, y_train, X_val, y_val, lambda_reg, exact=True)
                    result['noise'] = k
                    results['ridge_exact'].append(result)
                    print(f"  λ={lambda_reg:.3f}: Train MSE={result['train_mse']:.4f}, "
                        f"Val MSE={result['val_mse']:.4f}, "
                        f"Cond. Num={result['final_condition_number']:.2e}, "
                        f"Time={result['training_time']:.2f}s")
                            # Fixed Learning Rate Boosting

                print("\nRunning Fixed Learning Rate Boosting...")
                for lr in learning_rates:
                    result = comparison.fixed_lr_boosting(X_train, y_train, X_val, y_val, lr)
                    result['noise'] = k
                    results['fixed_lr'].append(result)
                    print(f"  lr={lr:.3f}: Train MSE={result['train_mse']:.4f}, "
                        f"Val MSE={result['val_mse']:.4f}, "
                        f"Time={result['training_time']:.2f}s")

    # Create the plots
    fig = create_noise_analysis_plots(results, noise, lambda_values, learning_rates)

    # Save the figure
    plt.savefig('data/images/noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    process_results(results, noise, lambda_values, learning_rates)
    compare_methods(results, noise)