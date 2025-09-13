import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class RidgeBooster:
    def __init__(self, X_train, X_val, y_train, y_val, lambda_reg, n_estimators, max_depth):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.lambda_reg = lambda_reg
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    

    def fit_model(self):
        self.cond_number = None     # matrix condition number of (F·F^T + λ I) (max eigenvalue / min eigenvalue)
        self.train_mse   = None     # final training MSE
        self.val_mse     = None     # final validation MSE

        F_train = []  # list of shape (num_trees, training_data_sample_size)
        F_val   = []  # list of shape (num_trees, validation_data_sample_size)
            
        # store per-iteration weights
        self.iteration_weights = []
        
        # use mean model to get a baseline MSE for comparison with model MSEs        
        mean_pred = np.mean(self.y_train)
        baseline_train_mse = mean_squared_error(self.y_train, np.full_like(self.y_train, mean_pred))
        baseline_val_mse = mean_squared_error(self.y_val, np.full_like(self.y_val, mean_pred))
        print(f"Baseline MSE - Train: {baseline_train_mse:.2f}, Val: {baseline_val_mse:.2f}")
        self.iteration_train_mse = [baseline_train_mse]
        self.iteration_val_mse = [baseline_val_mse]

        for i in range(self.n_estimators):
            if i == 0:
                # The first iteration trains the first tree to y_train.
                # Subsequent iterations will train to residuals of the ensemble's predictions from the previous iteration.
                residuals = self.y_train.copy()
            else:
                # Form the matrix F (T_trees × N_samples)
                F_mat = np.vstack(F_train)
                # Build A = F · F^T + λ I
                A = F_mat @ F_mat.T + self.lambda_reg * np.eye(i)
                # Build b = F · y_train
                b = F_mat @ self.y_train
                # Solve A·w = b
                w = np.linalg.solve(A, b)
                # compute training and validation predictions on the training set
                train_pred = F_mat.T @ w
                val_pred = np.vstack(F_val[:i]).T @ w
                # compute and store history of MSEs for post-training analysis
                current_train_mse = mean_squared_error(self.y_train, train_pred)
                current_val_mse = mean_squared_error(self.y_val, val_pred)
                self.iteration_train_mse.append(current_train_mse)
                self.iteration_val_mse.append(current_val_mse)
                
                # New residual for this iteration
                residuals = self.y_train - train_pred
                
                # track weight evolution for post-training analysis
                full_weights = np.zeros(self.n_estimators)
                full_weights[:len(w)] = w
                self.iteration_weights.append(full_weights.copy())

            # fit a new tree to residuals of latest ensemble prediction
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(self.X_train, residuals)

            # obtain latest tree's predictions on both training and validation data
            f_train = tree.predict(self.X_train)
            f_val   = tree.predict(self.X_val)

            # save latest residual prediction series to tree prediction matrices
            F_train.append(f_train)
            F_val.append(f_val)
            
        F_matrix_final = np.vstack(F_train)
        A_final = F_matrix_final @ F_matrix_final.T + self.lambda_reg * np.eye(self.n_estimators)
        b_final = F_matrix_final @ self.y_train
        
        # solve Aw = b for the final weights w
        self.final_weights = np.linalg.solve(A_final, b_final)
        
        eigvals = np.linalg.eigvalsh(A_final)
        self.cond_number = eigvals.max() / eigvals.min()
        
        # final predictions and MSE
        final_train_pred = F_matrix_final.T @ self.final_weights
        final_val_pred = np.vstack(F_val).T @ self.final_weights
        self.final_train_mse = mean_squared_error(self.y_train, final_train_pred)
        self.final_val_mse = mean_squared_error(self.y_val, final_val_pred)
        
        # store final weights and MSE for post-training analysis
        final_weights_padded = np.zeros(self.n_estimators)
        final_weights_padded[:len(self.final_weights)] = self.final_weights
        self.iteration_weights.append(final_weights_padded)
        self.iteration_train_mse.append(self.final_train_mse)
        self.iteration_val_mse.append(self.final_val_mse)
        self.tree_strengths = np.array([np.std(tree_pred) for tree_pred in F_train])

    

if __name__ == "__main__":
    # create a synthetic regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # # load a real regression dataset
    # california = fetch_california_housing()
    # X, y = california.data, california.target
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # common parameters
    n_estimators = 30
    max_depth    = 3

    # choose a set of lambda values to explore (from 1 up to 10^6)
    lambda_values = np.logspace(0, 6, 20)

    cond_numbers = []     # condition number of (F·F^T + λ I)
    train_mses   = []     # final training MSEs
    val_mses     = []     # final validation MSEs
    for l in lambda_values:
        print(f"\nProcessing λ = {l:.1e}")
        ridge_booster = RidgeBooster(X_val=X_val, X_train=X_train, y_train=y_train, y_val=y_val, n_estimators=n_estimators, max_depth=max_depth, lambda_reg=l)
        ridge_booster.fit_model()
        # Store results
        cond_numbers.append(ridge_booster.cond_number)
        train_mses.append(ridge_booster.train_mse)
        val_mses.append(ridge_booster.val_mse)
    
    # ---- plotting of key results ----
    plt.figure(figsize=(12, 4))

    # condition number vs λ
    plt.subplot(1, 2, 1)
    plt.semilogx(lambda_values, cond_numbers, marker='o', color='darkorange')
    plt.title("Condition Number vs. λ (CG Ridge)")
    plt.xlabel("λ")
    plt.ylabel("Condition Number of (F·Fᵀ + λI)")
    plt.grid(True)

    # train/val MSE vs λ
    plt.subplot(1, 2, 2)
    plt.semilogx(lambda_values, train_mses, marker='s', label="Train MSE", color='royalblue')
    plt.semilogx(lambda_values, val_mses,   marker='^', label="Val MSE",   color='firebrick')
    plt.title("Train/Validation MSE vs. λ")
    plt.xlabel("λ")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
