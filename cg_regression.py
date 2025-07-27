import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

# 1) Create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2) load a real regression dataset
# california = fetch_california_housing()
# X, y = california.data, california.target
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# 2) Common parameters
n_estimators = 30
max_depth    = 3

# 3) Choose a set of lambda values to explore (from 10^0 up to 10^6)
lambda_values = np.logspace(2, 6, 20)  # [1, 10, 100, 1e3, 1e4, 1e5, 1e6]

# Containers for results
cond_numbers = []     # Condition number of (F·F^T + λ I)
train_mses   = []     # Final training MSE
val_mses     = []     # Final validation MSE

for lambda_reg in lambda_values:
    # ---- Build the ensemble, stagewise ----
    F_train = []  # list of shape (n_trees, n_samples_train)
    F_val   = []  # list of shape (n_trees, n_samples_val)

    for i in range(n_estimators):
        if i == 0:
            # First iteration: no trees yet, so residual = y_train
            residuals = y_train.copy()
        else:
            # Form the matrix F (n_trees × n_samples)
            F_mat = np.vstack(F_train)  # shape = (i, n_train)
            # Build A = F · F^T + λ I
            A = F_mat @ F_mat.T + lambda_reg * np.eye(i)
            # Build b = F · y_train
            b = F_mat @ y_train

            # Solve A·w = b via Conjugate Gradient
            w, info = cg(A, b, maxiter=1000)
            # Compute partial prediction on the training set
            train_pred_partial = w @ F_mat  # shape = (n_train,)
            # New residual for this iteration
            residuals = y_train - train_pred_partial

        # Fit a new tree to (X_train, residuals)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X_train, residuals)

        # Save its predictions (f_i) on both train and val
        f_train = tree.predict(X_train)  # shape = (n_train,)
        f_val   = tree.predict(X_val)    # shape = (n_val,)

        F_train.append(f_train)
        F_val.append(f_val)

    # ---- Once all trees are built, do the final CG solve ----
    F_mat   = np.vstack(F_train)  # shape = (n_trees, n_train)
    A_final = F_mat @ F_mat.T + lambda_reg * np.eye(n_estimators)
    b_final = F_mat @ y_train

    # Solve (A_final) w = b_final
    final_w, info_final = cg(A_final, b_final, maxiter=1000)

    # Compute condition number of A_final
    eigvals = np.linalg.eigvalsh(A_final)
    cond_num = eigvals.max() / eigvals.min()
    cond_num = np.linalg.cond(A_final)


    # Compute final predictions & MSE on train/val
    train_preds = final_w @ F_mat                       # shape = (n_train,)
    val_preds   = final_w @ np.vstack(F_val)            # shape = (n_val,)

    train_mse = mean_squared_error(y_train, train_preds)
    val_mse   = mean_squared_error(y_val, val_preds)

    # Store results
    cond_numbers.append(cond_num)
    train_mses.append(train_mse)
    val_mses.append(val_mse)

# ---- Plotting ----
plt.figure(figsize=(12, 4))

# (a) Condition number vs λ
plt.subplot(1, 2, 1)
plt.semilogx(lambda_values, cond_numbers, marker='o', color='darkorange')
plt.title("Condition Number vs. λ (CG Ridge)")
plt.xlabel("λ")
plt.ylabel("Condition Number of (F·Fᵀ + λI)")
plt.grid(True)

# (b) Train/Val MSE vs λ
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
