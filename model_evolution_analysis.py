import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ridge_booster import RidgeBooster

"""
Run this to program to view how the ridge booster's weights and MSE evolve over the course of its training process.
This is useful for assessing the stability of the weight updates and other properties of the training process.
"""

# create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# parameters
n_estimators = 30
max_depth = 3

# compare weight behavior over a range of lambda hyperparameter choices
lambda_values = np.logspace(2, 6, 5)

# storage for results
results = {
    'lambda_values': lambda_values,
    'final_metrics': {
        'cond_numbers': [],
        'train_mses': [],
        'val_mses': []
    },
    'weight_evolution': {},
    'mse_evolution': {},
    'tree_contributions': {},
    'weight_stability': {}
}

for lambda_reg in lambda_values:
    print(f"\nProcessing λ = {lambda_reg:.1e}")
    ridge_booster = RidgeBooster(X_val=X_val, X_train=X_train, y_train=y_train, y_val=y_val, n_estimators=n_estimators, max_depth=max_depth, lambda_reg=lambda_reg)
    ridge_booster.fit_model()
    
    results['weight_evolution'][lambda_reg] = np.array(ridge_booster.iteration_weights)
    results['mse_evolution'][lambda_reg] = {
        'train': ridge_booster.iteration_train_mse,
        'val': ridge_booster.iteration_val_mse
    }
    results['tree_contributions'][lambda_reg] = ridge_booster.final_weights * ridge_booster.tree_strengths
    if len(ridge_booster.iteration_weights) > 1:
        weight_matrix = np.array(ridge_booster.iteration_weights)
        results['weight_stability'][lambda_reg] = np.var(weight_matrix, axis=0)
    else:
        results['weight_stability'][lambda_reg] = np.zeros(n_estimators)
    
    results['final_metrics']['cond_numbers'].append(ridge_booster.cond_number)
    results['final_metrics']['train_mses'].append(ridge_booster.final_train_mse)
    results['final_metrics']['val_mses'].append(ridge_booster.final_val_mse)
        

def plot_basic_results():
    """plot various lambda vs performance curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # condition number plot
    axes[0].semilogx(lambda_values, results['final_metrics']['cond_numbers'], 'o-')
    axes[0].set_title('Condition Number vs λ')
    axes[0].set_xlabel('λ')
    axes[0].set_ylabel('Condition Number')
    axes[0].grid(True)
     
    # MSE comparison plot
    axes[1].semilogx(lambda_values, results['final_metrics']['train_mses'], 'o-', label='Train MSE')
    axes[1].semilogx(lambda_values, results['final_metrics']['val_mses'], 's-', label='Val MSE')
    axes[1].set_title('MSE vs λ')
    axes[1].set_xlabel('λ')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)
    
    # Overfitting plot
    overfitting = np.array(results['final_metrics']['val_mses']) - np.array(results['final_metrics']['train_mses'])
    axes[2].semilogx(lambda_values, overfitting, 'd-', color='purple')
    axes[2].set_title('Overfitting vs λ')
    axes[2].set_xlabel('λ')
    axes[2].set_ylabel('Val MSE - Train MSE')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('data/images/lambda_performance_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_evolution_analysis():
    """Plot weight and MSE evolution over iterations"""
    n_lambdas = len(lambda_values)
    fig, axes = plt.subplots(2, n_lambdas, figsize=(4*n_lambdas, 8))
    
    if n_lambdas == 1:
        axes = axes.reshape(2, 1)
    
    for i, lam in enumerate(lambda_values):
        # Weight evolution heatmap
        weights = results['weight_evolution'][lam]
        im = axes[0, i].imshow(weights.T, aspect='auto', cmap='RdBu_r', 
                              interpolation='nearest', origin='lower')
        axes[0, i].set_title(f'Weight Evolution\nλ={lam:.1e}')
        axes[0, i].set_xlabel('Boosting Iteration')
        axes[0, i].set_ylabel('Tree Index')
        plt.colorbar(im, ax=axes[0, i])
        
        # MSE evolution
        train_mse = results['mse_evolution'][lam]['train']
        val_mse = results['mse_evolution'][lam]['val']
        iterations = range(len(train_mse))        
        axes[1, i].plot(iterations, train_mse, 'o-', label='Train MSE', linewidth=2)
        axes[1, i].plot(iterations, val_mse, 's-', label='Val MSE', linewidth=2)
        axes[1, i].set_title(f'MSE Evolution\nλ={lam:.1e}')
        axes[1, i].set_xlabel('Boosting Iteration')
        axes[1, i].set_ylabel('MSE')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
        # highlight iteration with biggest training MSE improvement
        if len(train_mse) > 1:
            improvements = np.diff(train_mse) * -1  # Negative diff = improvement
            if len(improvements) > 0:
                best_iter = np.argmax(improvements) + 1
                axes[1, i].axvline(x=best_iter, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data/images/weight_and_mse_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_analysis():
    """Plot additional analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Final weight distributions
    for lam in lambda_values:
        final_weights = results['weight_evolution'][lam][-1]
        axes[0, 0].plot(range(len(final_weights)), final_weights, 'o-', 
                       label=f'λ={lam:.1e}', alpha=0.7)
    axes[0, 0].set_title('Final Weight Distribution')
    axes[0, 0].set_xlabel('Tree Index')
    axes[0, 0].set_ylabel('Weight Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Weight stability
    stability_means = [np.mean(results['weight_stability'][lam]) for lam in lambda_values]
    axes[0, 1].semilogx(lambda_values, stability_means, 'o-', color='green')
    axes[0, 1].set_title('Weight Stability vs λ')
    axes[0, 1].set_xlabel('λ')
    axes[0, 1].set_ylabel('Mean Weight Variance')
    axes[0, 1].grid(True)
    
    # MSE improvement per iteration (first lambda only for clarity)
    first_lambda = lambda_values[0]
    train_mse = results['mse_evolution'][first_lambda]['train']
    if len(train_mse) > 1:
        improvements = np.diff(train_mse) * -1  # Convert to improvements
        axes[1, 0].bar(range(1, len(improvements) + 1), improvements, alpha=0.7)
        axes[1, 0].set_title(f'MSE Improvement per Iteration\n(λ={first_lambda:.1e})')
        axes[1, 0].set_xlabel('Boosting Iteration')
        axes[1, 0].set_ylabel('MSE Reduction')
        axes[1, 0].grid(True, alpha=0.3)
    
    # tree contributions
    for lam in lambda_values:
        contributions = results['tree_contributions'][lam]
        axes[1, 1].plot(range(len(contributions)), contributions, 'o-', 
                       label=f'λ={lam:.1e}', alpha=0.7)
    axes[1, 1].set_title('Tree Contributions (Weight × Strength)')
    axes[1, 1].set_xlabel('Tree Index')
    axes[1, 1].set_ylabel('Contribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('data/images/post_training_details.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary():
    best_lambda_idx = np.argmin(results['final_metrics']['val_mses'])
    best_lambda = lambda_values[best_lambda_idx]
    best_weights = results['weight_evolution'][best_lambda][-1]
    
    print("\n" + "="*60)
    print("RIDGE BOOSTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"Best λ (min val MSE): {best_lambda:.2e}")
    print(f"Best validation MSE: {results['final_metrics']['val_mses'][best_lambda_idx]:.4f}")
    print(f"Corresponding train MSE: {results['final_metrics']['train_mses'][best_lambda_idx]:.4f}")
    print(f"Overfitting: {results['final_metrics']['val_mses'][best_lambda_idx] - results['final_metrics']['train_mses'][best_lambda_idx]:.4f}")
    
    print(f"\nWeight Statistics (best λ):")
    print(f"  Mean weight: {np.mean(best_weights):.4f}")
    print(f"  Weight std: {np.std(best_weights):.4f}")
    print(f"  Weight range: [{np.min(best_weights):.4f}, {np.max(best_weights):.4f}]")
    
    # Weight concentration analysis
    weight_abs_sum = np.sum(np.abs(best_weights))
    if weight_abs_sum > 0:
        weight_contribution_pct = np.abs(best_weights) / weight_abs_sum * 100
        dominant_trees = np.sum(weight_contribution_pct > 10)
        print(f"  Trees contributing >10%: {dominant_trees}/{len(best_weights)}")
        
        # Top contributing trees
        top_indices = np.argsort(weight_contribution_pct)[-3:][::-1]
        print(f"  Top 3 tree contributions: {[f'Tree {i}: {weight_contribution_pct[i]:.1f}%' for i in top_indices]}")

print("Running Ridge Booster Analysis...")
plot_basic_results()
plot_evolution_analysis()
plot_detailed_analysis()
print_summary()

print(f"\nAnalysis complete. Processed {len(lambda_values)} lambda values with {n_estimators} trees each.")