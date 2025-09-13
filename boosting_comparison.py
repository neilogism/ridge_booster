import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from typing import Tuple, List, Dict
import time


class BoostingComparison:
    """
    A comprehensive comparison framework for different boosting methods.
    """
    
    def __init__(self, n_estimators: int = 30, max_depth: int = 3, random_state: int = 42, n_samples: int = 100, n_features: int = 10, noise: float = 10000.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.results = {}
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        
    def load_data(self, dataset_type: str = 'synthetic') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split dataset.
        
        Args:
            dataset_type: 'synthetic' or 'california'
        """
        if dataset_type == 'synthetic':
            X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features, noise=self.noise, 
                                 random_state=self.random_state)
        elif dataset_type == 'california':
            california = fetch_california_housing()
            X, y = california.data, california.target
        else:
            raise ValueError("dataset_type must be 'synthetic' or 'california'")
            
        return train_test_split(X, y, test_size=0.3, random_state=self.random_state)
    
    def ridge_boosting(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray, 
                         lambda_reg: float, exact: bool = False) -> Dict:
        """
        Ridge-regularized boosting using conjugate gradient solver.
        """
        start_time = time.time()
        
        F_train = []  # Store tree predictions on training set
        F_val = []    # Store tree predictions on validation set
        train_mse_history = []
        val_mse_history = []
        condition_numbers = []
        
        for i in range(self.n_estimators):
            # Calculate residuals
            if i == 0:
                residuals = y_train.copy()
            else:
                # Form the matrix F (n_trees × n_samples)
                F_mat = np.vstack(F_train)  # shape = (i, n_train)
                # Build A = F · F^T + λ I
                A = F_mat @ F_mat.T + lambda_reg * np.eye(i)
                # Build b = F · y_train
                b = F_mat @ y_train
                
                if not exact:
                    # Solve A·w = b via Conjugate Gradient
                    w, info = cg(A, b, maxiter=1000)
                else:
                    w = np.linalg.solve(A, b)
                
                # Compute partial prediction and residuals
                train_pred_partial = w @ F_mat
                residuals = y_train - train_pred_partial
                
                # Track condition number
                cond_num = np.linalg.cond(A)
                condition_numbers.append(cond_num)
            
            # Fit new tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                       random_state=self.random_state)
            tree.fit(X_train, residuals)
            
            # Store predictions
            f_train = tree.predict(X_train)
            f_val = tree.predict(X_val)
            F_train.append(f_train)
            F_val.append(f_val)
            
            # Calculate current performance (for intermediate tracking)
            if i > 0:  # Skip first iteration since we don't have weights yet
                F_mat_current = np.vstack(F_train)
                A_current = F_mat_current @ F_mat_current.T + lambda_reg * np.eye(len(F_train))
                b_current = F_mat_current @ y_train
                if exact:
                    w_current = np.linalg.solve(A_current, b_current)
                else:
                    w_current, _ = cg(A_current, b_current, maxiter=1000)
                
                train_pred = w_current @ F_mat_current
                val_pred = w_current @ np.vstack(F_val)
                
                train_mse_history.append(mean_squared_error(y_train, train_pred))
                val_mse_history.append(mean_squared_error(y_val, val_pred))
        
        # Final solve
        F_mat_final = np.vstack(F_train)
        A_final = F_mat_final @ F_mat_final.T + lambda_reg * np.eye(self.n_estimators)
        b_final = F_mat_final @ y_train
        if exact:
            final_w = np.linalg.solve(A_final, b_final)
        else:
            final_w, _ = cg(A_final, b_final, maxiter=1000)
        
        # Final predictions
        train_preds = final_w @ F_mat_final
        val_preds = final_w @ np.vstack(F_val)
        
        # Final condition number
        final_cond_num = np.linalg.cond(A_final)
        
        training_time = time.time() - start_time
        
        return {
            'method': 'Ridge CG Boosting',
            'lambda': lambda_reg,
            'train_mse': mean_squared_error(y_train, train_preds),
            'val_mse': mean_squared_error(y_val, val_preds),
            'train_mse_history': train_mse_history,
            'val_mse_history': val_mse_history,
            'condition_numbers': condition_numbers,
            'final_condition_number': final_cond_num,
            'weights': final_w,
            'training_time': training_time
        }
    
    def fixed_lr_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         learning_rate: float) -> Dict:
        """
        Traditional gradient boosting with fixed learning rate.
        """
        start_time = time.time()
        
        train_preds = np.zeros_like(y_train, dtype=float)
        val_preds = np.zeros_like(y_val, dtype=float)
        train_mse_history = []
        val_mse_history = []
        
        residuals = y_train.copy()
        
        for i in range(self.n_estimators):
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                       random_state=self.random_state)
            tree.fit(X_train, residuals)
            
            # Update predictions
            train_preds += learning_rate * tree.predict(X_train)
            val_preds += learning_rate * tree.predict(X_val)
            
            # Update residuals
            residuals = y_train - train_preds
            
            # Track performance
            train_mse_history.append(mean_squared_error(y_train, train_preds))
            val_mse_history.append(mean_squared_error(y_val, val_preds))
        
        training_time = time.time() - start_time
        
        return {
            'method': 'Fixed Learning Rate Boosting',
            'learning_rate': learning_rate,
            'train_mse': mean_squared_error(y_train, train_preds),
            'val_mse': mean_squared_error(y_val, val_preds),
            'train_mse_history': train_mse_history,
            'val_mse_history': val_mse_history,
            'training_time': training_time
        }
    
    def greedy_reweighting_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  lambda_reg: float, alpha_penalty: float = 1.0) -> Dict:
        """
        Greedy reweighting boosting approach.
        """
        start_time = time.time()
        
        # First, collect all tree predictions
        F_train, F_val = [], []
        residuals = y_train.copy()
        
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                       random_state=self.random_state)
            tree.fit(X_train, residuals)
            
            f_train = tree.predict(X_train)
            f_val = tree.predict(X_val)
            F_train.append(f_train)
            F_val.append(f_val)
            
            # Update residuals based on current sum
            residuals = y_train - np.sum(F_train, axis=0)
        
        # Greedy reweighting
        train_preds = np.zeros_like(y_train)
        val_preds = np.zeros_like(y_val)
        weights = []
        train_mse_history = []
        val_mse_history = []
        
        for i in range(self.n_estimators):
            f_train = F_train[i]
            f_val = F_val[i]
            
            def greedy_loss(w):
                return (mean_squared_error(y_train, train_preds + w * f_train) + 
                       lambda_reg * w**2 + alpha_penalty)
            
            res = minimize(greedy_loss, x0=[0.0])
            w = res.x[0]
            weights.append(w)
            
            train_preds += w * f_train
            val_preds += w * f_val
            
            train_mse_history.append(mean_squared_error(y_train, train_preds))
            val_mse_history.append(mean_squared_error(y_val, val_preds))
        
        training_time = time.time() - start_time
        
        return {
            'method': 'Greedy Reweighting Boosting',
            'lambda': lambda_reg,
            'alpha_penalty': alpha_penalty,
            'train_mse': mean_squared_error(y_train, train_preds),
            'val_mse': mean_squared_error(y_val, val_preds),
            'train_mse_history': train_mse_history,
            'val_mse_history': val_mse_history,
            'weights': weights,
            'training_time': training_time
        }
    
    def run_comparison(self, dataset_type: str = 'synthetic',
                      lambda_values: List[float] = [0.1, 1.0, 10.0],
                      learning_rates: List[float] = [0.01, 0.1, 1.0]) -> Dict:
        """
        Run comprehensive comparison across all methods and hyperparameters.
        """
        X_train, X_val, y_train, y_val = self.load_data(dataset_type)
        
        print(f"Dataset: {dataset_type}")
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print("-" * 60)
        
        results = {
            'ridge_cg': [],
            "ridge_exact": [],
            'fixed_lr': [],
            'greedy_reweight': []
        }
        
        # Ridge CG Boosting
        print("Running Ridge CG Boosting...")
        for lambda_reg in lambda_values:
            result = self.ridge_boosting(X_train, y_train, X_val, y_val, lambda_reg)
            results['ridge_cg'].append(result)
            print(f"  λ={lambda_reg:.3f}: Train MSE={result['train_mse']:.4f}, "
                  f"Val MSE={result['val_mse']:.4f}, "
                  f"Cond. Num={result['final_condition_number']:.2e}, "
                  f"Time={result['training_time']:.2f}s")
        
        for lambda_reg in lambda_values:
            result = self.ridge_boosting(X_train, y_train, X_val, y_val, lambda_reg, exact=True)
            results['ridge_exact'].append(result)
            print(f"  λ={lambda_reg:.3f}: Train MSE={result['train_mse']:.4f}, "
                  f"Val MSE={result['val_mse']:.4f}, "
                  f"Cond. Num={result['final_condition_number']:.2e}, "
                  f"Time={result['training_time']:.2f}s")
        
        # Fixed Learning Rate Boosting
        print("\nRunning Fixed Learning Rate Boosting...")
        for lr in learning_rates:
            result = self.fixed_lr_boosting(X_train, y_train, X_val, y_val, lr)
            results['fixed_lr'].append(result)
            print(f"  lr={lr:.3f}: Train MSE={result['train_mse']:.4f}, "
                  f"Val MSE={result['val_mse']:.4f}, "
                  f"Time={result['training_time']:.2f}s")
        
        # Greedy Reweighting Boosting
        print("\nRunning Greedy Reweighting Boosting...")
        for lambda_reg in lambda_values:
            result = self.greedy_reweighting_boosting(X_train, y_train, X_val, y_val, lambda_reg)
            results['greedy_reweight'].append(result)
            print(f"  λ={lambda_reg:.3f}: Train MSE={result['train_mse']:.4f}, "
                  f"Val MSE={result['val_mse']:.4f}, "
                  f"Time={result['training_time']:.2f}s")
        
        self.results[dataset_type] = results
        return results
    
    def plot_comparison(self, dataset_type: str = 'synthetic', figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive comparison plots.
        """
        if dataset_type not in self.results:
            raise ValueError(f"No results found for dataset '{dataset_type}'. Run comparison first.")
        
        results = self.results[dataset_type]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Boosting Methods Comparison - {dataset_type.title()} Dataset', fontsize=16)
        
        # Plot 1: Validation MSE vs Hyperparameter
        ax1 = axes[0, 0]
        
        # Ridge exact results
        lambda_vals = [r['lambda'] for r in results['ridge_exact']]
        ridge_exact_val_mse = [r['val_mse'] for r in results['ridge_exact']]
        ax1.semilogx(lambda_vals, ridge_exact_val_mse, 'o-', label='Ridge', color='purple')
        
        # Ridge CG results
        lambda_vals = [r['lambda'] for r in results['ridge_cg']]
        ridge_val_mse = [r['val_mse'] for r in results['ridge_cg']]
        ax1.semilogx(lambda_vals, ridge_val_mse, 'o-', label='Ridge CG', color='darkorange')
        
        # Fixed LR results  
        lr_vals = [r['learning_rate'] for r in results['fixed_lr']]
        fixed_val_mse = [r['val_mse'] for r in results['fixed_lr']]
        ax1.semilogx(lr_vals, fixed_val_mse, 's-', label='Fixed LR', color='royalblue')
        
        # Greedy results
        greedy_val_mse = [r['val_mse'] for r in results['greedy_reweight']]
        ax1.semilogx(lambda_vals, greedy_val_mse, '^-', label='Greedy Reweight', color='firebrick')
        
        ax1.set_xlabel('Hyperparameter Value')
        ax1.set_ylabel('Validation MSE')
        ax1.set_title('Validation MSE vs Hyperparameter')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Training vs Validation MSE
        ax2 = axes[0, 1]
        
        ridge_exact_train_mse = [r['train_mse'] for r in results['ridge_exact']]
        ridge_train_mse = [r['train_mse'] for r in results['ridge_cg']]
        fixed_train_mse = [r['train_mse'] for r in results['fixed_lr']]
        greedy_train_mse = [r['train_mse'] for r in results['greedy_reweight']]
        
        print(ridge_exact_val_mse, ridge_exact_train_mse)
        
        
        ax2.scatter(ridge_exact_train_mse, ridge_exact_val_mse, c='purple', marker='o', 
            s=60, label='Ridge', alpha=0.7)
        
        ax2.scatter(ridge_train_mse, ridge_val_mse, c='darkorange', marker='o', 
                   s=60, label='Ridge CG', alpha=0.7)
        ax2.scatter(fixed_train_mse, fixed_val_mse, c='royalblue', marker='s', 
                   s=60, label='Fixed LR', alpha=0.7)
        ax2.scatter(greedy_train_mse, greedy_val_mse, c='firebrick', marker='^', 
                   s=60, label='Greedy Reweight', alpha=0.7)
        
        # Add diagonal line for reference
        min_mse = min(min(ridge_train_mse + fixed_train_mse + greedy_train_mse + ridge_exact_train_mse),
                     min(ridge_val_mse + fixed_val_mse + greedy_val_mse + ridge_exact_val_mse))
        max_mse = max(max(ridge_train_mse + fixed_train_mse + greedy_train_mse + ridge_exact_train_mse),
                     max(ridge_val_mse + fixed_val_mse + greedy_val_mse + ridge_exact_val_mse))
        ax2.plot([min_mse, max_mse], [min_mse, max_mse], 'k--', alpha=0.5, label='Perfect Fit')
        
        ax2.set_xlabel('Training MSE')
        ax2.set_ylabel('Validation MSE')
        ax2.set_title('Training vs Validation MSE')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Condition Numbers (Ridge only)
        ax3 = axes[0, 2]
        condition_nums = [r['final_condition_number'] for r in results['ridge_exact']]
        ax3.semilogy(lambda_vals, condition_nums, 'o-', color='purple')
        ax3.set_xlabel('λ')
        ax3.set_ylabel('Condition Number')
        ax3.set_title('Matrix Condition Number (Ridge)')
        ax3.grid(True)
        
        # Plot 4: Training Time Comparison
        ax4 = axes[1, 0]
        
        methods = ['Ridge CG', 'Ridge Exact', 'Fixed LR', 'Greedy Reweight']
        avg_times = [
            np.mean([r['training_time'] for r in results['ridge_cg']]),
            np.mean([r['training_time'] for r in results['ridge_exact']]),
            np.mean([r['training_time'] for r in results['fixed_lr']]),
            np.mean([r['training_time'] for r in results['greedy_reweight']])
        ]
        colors = ['darkorange', 'purple', 'royalblue', 'firebrick']
        
        bars = ax4.bar(methods, avg_times, color=colors, alpha=0.7)
        ax4.set_ylabel('Average Training Time (s)')
        ax4.set_title('Training Time Comparison')
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # Plot 5: Learning Curves (best performing models)
        ax5 = axes[1, 1]
        
        # Find best performing model for each method
        best_ridge_cg = min(results['ridge_cg'], key=lambda x: x['val_mse'])
        best_ridge = min(results['ridge_exact'], key=lambda x: x['val_mse'])
        best_fixed = min(results['fixed_lr'], key=lambda x: x['val_mse'])
        best_greedy = min(results['greedy_reweight'], key=lambda x: x['val_mse'])
        
        if best_ridge['val_mse_history']:
            ax5.plot(best_ridge['val_mse_history'], label=f"Ridge CG (λ={best_ridge['lambda']})", 
                    color='purple')
        if best_ridge['val_mse_history']:
            ax5.plot(best_ridge_cg['val_mse_history'], label=f"Ridge CG (λ={best_ridge['lambda']})", 
                    color='darkorange')
            
        ax5.plot(best_fixed['val_mse_history'], label=f"Fixed LR (lr={best_fixed['learning_rate']})", 
                color='royalblue')
        ax5.plot(best_greedy['val_mse_history'], label=f"Greedy (λ={best_greedy['lambda']})", 
                color='firebrick')
        
        ax5.set_xlabel('Boosting Iteration')
        ax5.set_ylabel('Validation MSE')
        ax5.set_title('Learning Curves (Best Models)')
        ax5.legend()
        ax5.grid(True)
        
        # Plot 6: Summary Table
        ax6 = axes[1, 2]
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        table_data = []
        table_data.append(['Method', 'Best Val MSE', 'Best Hyperparam.', 'Avg Time (s)'])
        
        best_ridge_mse = min(r['val_mse'] for r in results['ridge_exact'])
        best_ridge_param = next(r['lambda'] for r in results['ridge_exact'] if r['val_mse'] == best_ridge_mse)
        ridge_avg_time = np.mean([r['training_time'] for r in results['ridge_exact']])
        table_data.append(['Ridge', f'{best_ridge_mse:.4f}', f'λ={best_ridge_param:.4f}', f'{ridge_avg_time:.2f}'])
        
        best_ridge_cg_mse = min(r['val_mse'] for r in results['ridge_cg'])
        best_ridge_cg_param = next(r['lambda'] for r in results['ridge_cg'] if r['val_mse'] == best_ridge_cg_mse)
        ridge_cg_avg_time = np.mean([r['training_time'] for r in results['ridge_cg']])
        table_data.append(['Ridge CG', f'{best_ridge_cg_mse:.4f}', f'λ={best_ridge_cg_param:.4f}', f'{ridge_cg_avg_time:.2f}'])
        
        best_fixed_mse = min(r['val_mse'] for r in results['fixed_lr'])
        best_fixed_param = next(r['learning_rate'] for r in results['fixed_lr'] if r['val_mse'] == best_fixed_mse)
        fixed_avg_time = np.mean([r['training_time'] for r in results['fixed_lr']])
        table_data.append(['Fixed LR', f'{best_fixed_mse:.4f}', f'lr={best_fixed_param:.4f}', f'{fixed_avg_time:.2f}'])
        
        best_greedy_mse = min(r['val_mse'] for r in results['greedy_reweight'])
        best_greedy_param = next(r['lambda'] for r in results['greedy_reweight'] if r['val_mse'] == best_greedy_mse)
        greedy_avg_time = np.mean([r['training_time'] for r in results['greedy_reweight']])
        table_data.append(['Greedy', f'{best_greedy_mse:.4f}', f'λ={best_greedy_param:.4f}', f'{greedy_avg_time:.2f}'])
        
        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 3)
        ax6.set_title('Performance Summary')
        
        plt.tight_layout(pad=3.0)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize comparison
    comparison = BoostingComparison(n_estimators=30, max_depth=3, random_state=42)
    
    # Define hyperparameter ranges
    lambda_values = np.logspace(2, 6, 20)  # [1, 10, 100, 1e3, 1e4, 1e5, 1e6]

    # learning_rates = [0.01, 0.1, 1.0]
    learning_rates = [i/1e6 for i in lambda_values]
    
    # Run comparison on synthetic dataset
    print("=" * 60)
    print("SYNTHETIC DATASET COMPARISON")
    print("=" * 60)
    results_synthetic = comparison.run_comparison(
        dataset_type='synthetic',
        lambda_values=lambda_values,
        learning_rates=learning_rates
    )
    
    # Plot results
    comparison.plot_comparison('synthetic')
    
    # Run comparison on California housing dataset
    print("\n" + "=" * 60)
    print("CALIFORNIA HOUSING DATASET COMPARISON")
    print("=" * 60)
    results_california = comparison.run_comparison(
        dataset_type='california',
        lambda_values=lambda_values,
        learning_rates=learning_rates
    )
    
    # Plot results
    comparison.plot_comparison('california')