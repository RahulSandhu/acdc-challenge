from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder


class CoefficientThresholdLasso(BaseEstimator, TransformerMixin):
    """
    Multi-level feature selection algorithm using Lasso with coefficient
    thresholding.

    This class applies a multi-step process:
    1. Variance filtering
    2. ANOVA F-test for multiclass relevance
    3. Lasso for sparse feature selection
    4. Coefficient thresholding to remove weakly contributing features

    Attributes:
        - lambda_path (np.ndarray): Values of lambda to compute the lasso path.
        - lambda_grid (np.ndarray): Grid of lambda values for cross-validation.
        - cv_folds (int): Number of folds for cross-validation.
        - lambda_opt (float): Optimal lambda found by CV.
        - coe_thropt (float): Optimal coefficient threshold minimizing MSE.
        - selected_features_ (np.ndarray): Indices of final selected features.
        - coefs_lasso_path (np.ndarray): Lasso coefficients across lambda_path.
        - lambda_grid_mse (list): Cross-validated MSE for each lambda in
          lambda_grid.
        - coe_thr_mse (list): MSEs for coefficient thresholds.
        - lambda_values (np.ndarray): Lambda values used in lasso_path.
        - coe_thr_values (list): Coefficient threshold values explored.
    """

    def __init__(
        self,
        lambda_path: np.ndarray = np.linspace(1, 0.05, 100),
        lambda_grid: np.ndarray = np.arange(0.01, 1, 0.01),
        cv_folds: int = 10,
    ):
        # Initialization of search parameters and result holders
        self.lambda_path = lambda_path
        self.lambda_grid = lambda_grid
        self.cv_folds = cv_folds

        self.lambda_opt = None
        self.coe_thropt = None

        self.selected_features_ = None
        self.coefs_lasso_path = None
        self.lambda_grid_mse = None
        self.coe_thr_mse = None
        self.lambda_values = None
        self.coe_thr_values = None

    def fit(self, X, y):
        """
        Fit the feature selection model to the data.

        Inputs:
            - X (np.ndarray or pd.DataFrame): Input feature matrix.
            - y (array-like): Class labels (multiclass).
        """

        # Step 1: Variance filtering
        var_filter = VarianceThreshold(threshold=0.0)
        X_var = var_filter.fit_transform(X)
        var_support_idx = np.array(var_filter.get_support(indices=True))

        # Step 2: ANOVA F-test (for multiclass feature selection)
        groups = [X_var[np.array(y) == label] for label in np.unique(y)]
        _, p_values = f_oneway(*groups)
        anova_support_idx = np.where(p_values < 0.05)[0]
        X_filtered = X_var[:, anova_support_idx]
        selected_indices = var_support_idx[anova_support_idx]

        # Step 3: Lasso path for visualization
        y_encoded = np.array(LabelEncoder().fit_transform(y))
        self.lambda_values, self.coefs_lasso_path, *_ = lasso_path(
            X_filtered, y_encoded, alphas=self.lambda_path
        )

        # Step 4: Tune λ using cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        lambda_grid_mse = []

        # Loop over each lambda in the grid to perform cross-validation
        for lam in self.lambda_grid:
            # List to store mean squared errors for each fold at the current lambda
            mse_folds = []

            # For each fold in K-Fold cross-validation
            for train_idx, val_idx in kf.split(X_filtered):
                # Split training and validation sets for this fold
                X_train, X_val = X_filtered[train_idx], X_filtered[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

                # Train a Lasso model on the training fold with current lambda
                model = Lasso(alpha=float(lam), max_iter=50000, tol=1e-5)
                model.fit(X_train, y_train)

                # Predict on the validation fold
                y_pred = model.predict(X_val)

                # Compute and store the MSE for this fold
                mse_folds.append(((y_val - y_pred) ** 2).mean())

            # After all folds, store the average MSE for this lambda
            lambda_grid_mse.append(np.mean(mse_folds))

        self.lambda_grid_mse = lambda_grid_mse
        self.lambda_opt = self.lambda_grid[np.argmin(lambda_grid_mse)]

        # Step 5: Final Lasso with optimal λ
        final_lasso = Lasso(alpha=self.lambda_opt, max_iter=50000, tol=1e-5)
        final_lasso.fit(X_filtered, y_encoded)
        h = final_lasso.coef_
        abs_h = np.abs(h)

        # Step 6: Grid search over coefficient thresholds
        hmax = abs_h.max()
        hmin = abs_h[abs_h > 0].min() if np.any(abs_h > 0) else 0
        coe_thr_values = []
        coe_thr_mse = {}

        # Split data into train and test sets (50/50) for evaluating MSE at different coefficient thresholds
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.5, random_state=42
        )

        # Fix for pyright
        X_train = cast(pd.DataFrame, X_train)
        X_test = cast(pd.DataFrame, X_test)

        # Start from the smallest non-zero coefficient (hmin)
        coe_thr = hmin

        # Iterate through thresholds from hmin to hmax to perform grid search
        while coe_thr <= hmax:
            # Create a mask of features whose coefficients are greater than current threshold
            coe_thr_values.append(coe_thr)
            mask = abs_h > coe_thr

            # Train Lasso using features above the threshold and compute test MSE
            if mask.sum() == 0:
                coe_thr_mse[coe_thr] = float("inf")
            else:
                model = Lasso(alpha=self.lambda_opt, max_iter=50000, tol=1e-5)
                model.fit(X_train[:, mask], y_train)
                y_pred = model.predict(X_test[:, mask])
                coe_thr_mse[coe_thr] = ((y_test - y_pred) ** 2).mean()

            # Increment threshold with step size 0.001
            coe_thr = round(coe_thr + 0.001, 3)

        self.coe_thr_values = coe_thr_values
        self.coe_thr_mse = [coe_thr_mse[thr] for thr in coe_thr_values]
        self.coe_thropt = min(coe_thr_mse.items(), key=lambda x: x[1])[0]

        # Step 7: Final selected features
        final_mask = abs_h > self.coe_thropt
        self.selected_features_ = selected_indices[final_mask]

        return self

    def transform(self, X):
        """
        Transform the input data by selecting the pre-determined features.

        Inputs:
            - X (array-like): The input data with shape (n_samples,
              n_features).

        Outputs:
            - ndarray: The transformed data with only the selected features,
              shape (n_samples, n_selected_features).
        """
        # Ensure X is a NumPy array
        X = np.asarray(X)

        # Select only the columns corresponding to the selected feature indices
        X_transformed = X[:, self.selected_features_]

        return X_transformed

    def get_selected_features(self, feature_names):
        """
        Retrieve the names of the selected features after fitting the model.

        Inputs:
            - feature_names (array-like or pd.Index): The complete list of
              feature names from the original dataset before any filtering.

        Outputs:
            - list: A list of feature names that were selected by the full
              pipeline (variance filtering, ANOVA, Lasso, and coefficient
              thresholding).
        """
        # Map the final selected indices to the original feature names
        selected_names = feature_names[self.selected_features_].tolist()

        return selected_names


if __name__ == "__main__":
    # Custom style
    plt.style.use("../../misc/custom_style.mplstyle")

    # Load dataset
    df = pd.read_csv("../../data/processed/norm_acdc_radiomics.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Ensure output directory exists
    img_dir = Path("../../images/feature_selection/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Split before feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fix for Pyright
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(pd.Series, y_train)
    y_test = cast(pd.Series, y_test)

    # Fit feature selector
    ctl = CoefficientThresholdLasso()
    ctl.fit(X_train, y_train)

    # Apply selection to both train and test sets
    X_train_selected = X_train.iloc[:, ctl.selected_features_]
    X_test_selected = X_test.iloc[:, ctl.selected_features_]

    # Fix for Pyright
    assert ctl.lambda_values is not None
    assert ctl.coefs_lasso_path is not None
    assert ctl.lambda_grid_mse is not None
    assert ctl.lambda_opt is not None
    assert ctl.lambda_grid is not None
    assert ctl.coe_thr_values is not None
    assert ctl.coe_thr_mse is not None
    assert ctl.coe_thropt is not None

    # Plot coefficient paths
    plt.figure(figsize=(10, 6))
    for i in range(ctl.coefs_lasso_path.shape[0]):
        plt.plot(ctl.lambda_values, ctl.coefs_lasso_path[i], linewidth=1)
    plt.axvline(
        x=ctl.lambda_opt,
        color='red',
        linestyle='--',
        label=f'λ opt = {ctl.lambda_opt:.3f}',
    )
    plt.title("Lasso Coefficient Paths")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient Value")
    plt.xlim(min(ctl.lambda_values), max(ctl.lambda_values))
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "coeff_vs_lambda.png")
    plt.show()

    # Find index and MSE of optimal lambda
    opt_idx = list(ctl.lambda_grid).index(ctl.lambda_opt)
    opt_mse = ctl.lambda_grid_mse[opt_idx]

    # Plot MSE vs Lambda
    plt.figure(figsize=(8, 5))
    plt.plot(ctl.lambda_grid, ctl.lambda_grid_mse, marker='o', label="MSE")
    plt.plot(
        ctl.lambda_opt,
        opt_mse,
        'ro',
        label=f'Min MSE: {opt_mse:.4f} at λ = {ctl.lambda_opt:.3f}',
    )
    plt.title("Loss (MSE) vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "mse_vs_lambda.png")
    plt.show()

    # Find index of optimal threshold and corresponding MSE
    opt_idx = ctl.coe_thr_values.index(ctl.coe_thropt)
    opt_mse = ctl.coe_thr_mse[opt_idx]

    # Plot MSE vs Coefficient Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(ctl.coe_thr_values, ctl.coe_thr_mse, marker='o', label="MSE")
    plt.plot(
        ctl.coe_thropt,
        opt_mse,
        'ro',
        label=f'Min MSE: {opt_mse:.4f} at θ = {ctl.coe_thropt:.3f}',
    )
    plt.title("Loss (MSE) vs Coefficient Threshold")
    plt.xlabel("Coefficient Threshold")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "mse_vs_threshold.png")
    plt.show()

    # Print selected features
    selected_features = ctl.get_selected_features(X.columns)
    print("Final selected features:", selected_features)
