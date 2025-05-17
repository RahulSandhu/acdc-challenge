import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from utils.CoefficientThresholdLasso import CoefficientThresholdLasso


class SimpleANN(nn.Module):
    """
    Fully-connected Artificial Neural Network (ANN) for classification tasks.

    This class builds a feedforward neural network with:
    1. An input layer
    2. A user-defined number of hidden layers
    3. An output layer
    4. User-specified activation functions between layers

    Attributes:
        - input_dim (int): Number of input features.
        - hidden_size (int): Number of neurons in each hidden layer.
        - hidden_layers (int): Number of hidden layers.
        - output_dim (int): Number of output classes.
        - activation_fn (nn.Module): Activation function to use (e.g.,
          nn.ReLU).
    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        hidden_layers,
        output_dim,
        activation_fn,
    ):
        """
        Initialize the SimpleANN model structure.

        Inputs:
            - input_dim (int): Number of input features.
            - hidden_size (int): Number of neurons in each hidden layer.
            - hidden_layers (int): Number of hidden layers.
            - output_dim (int): Number of output classes.
            - activation_fn (nn.Module): Activation function class (e.g.,
              nn.ReLU).
        """
        # PyTorch setup
        super(SimpleANN, self).__init__()

        # List to hold the layers sequentially
        layers = []

        # First layer: input dimension to hidden layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation_fn())

        # Hidden layers: hidden_size → hidden_size
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())

        # Output layer: hidden_size → output_dim
        layers.append(nn.Linear(hidden_size, output_dim))

        # Combine all layers into a single model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Inputs:
            - x (torch.Tensor): Input tensor with shape (batch_size,
              input_dim).

        Outputs:
            - torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        return self.network(x)


def ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_temp: pd.DataFrame,
    y_temp: pd.Series,
    hidden_layers: Sequence[int],
    hidden_size: Sequence[int],
    learning_rate: Sequence[float],
    activation_fn: Sequence[Any],
    optimizer: Sequence[Any],
    max_epochs: Sequence[int],
    pen: float = 0.5,
) -> Tuple[Dict[str, Optional[Pipeline]], pd.DataFrame, pd.DataFrame, str]:
    """
    Train and evaluate ANN models using two selection strategies: simple
    train/validation split and K-fold cross-validation, both incorporating CTL
    (Coefficient Threshold Lasso) and LDA (Linear Discriminant Analysis) as
    preprocessing steps.

    The function performs:
    1. A manual grid search on a train/validation split using CTL → LDA → ANN,
    selecting the model based on a penalized validation accuracy that accounts
    for overfitting (train-validation gap).
    2. A GridSearchCV over a combined training+validation set using the same
    CTL → LDA → ANN pipeline and penalized scoring.

    Inputs:
        - X_train (pd.DataFrame): Training features (already scaled).
        - y_train (pd.Series): Labels for the training set.
        - X_val (pd.DataFrame): Validation features (already scaled).
        - y_val (pd.Series): Labels for the validation set.
        - X_temp (pd.DataFrame): Combined training+validation features.
        - y_temp (pd.Series): Labels corresponding to X_temp.
        - hidden_layers (Sequence[int]): Number of hidden layers to try.
        - hidden_size (Sequence[int]): Number of neurons in each hidden layer
          to try.
        - learning_rate (Sequence[float]): Learning rates to try.
        - activation_fn (Sequence[Any]): Activation functions to try (e.g.,
          nn.ReLU, nn.Tanh).
        - optimizer (Sequence[Any]): Optimizers to try (e.g.,
          torch.optim.Adam).
        - max_epochs (Sequence[int]): Max epochs to try.
        - pen (float, optional): Penalty weight applied to the train-validation
          gap when scoring. Default is 0.5.

    Outputs:
        - Tuple containing:
            - Dict[str, Optional[Pipeline]]: Trained models under keys 'simple'
              and 'kfold'.
            - pd.DataFrame: DataFrame summarizing the manual grid search
              results.
            - pd.DataFrame: DataFrame summarizing the cross-validation search
              results.
            - str: Summary string reporting the selected hyperparameters and
              model performance.
    """
    # Initialize result container
    models: Dict[str, Optional[Pipeline]] = {}

    # Grid search pipeline: CTL → LDA → ANN
    grid_pipeline_simple = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    grid_parameters_simple = [
        {
            "hidden_layers": hl,
            "hidden_size": hs,
            "learning_rate": lr,
            "activation_fn": afn,
            "optimizer": opt,
            "max_epochs": me,
        }
        for hl in hidden_layers
        for hs in hidden_size
        for lr in learning_rate
        for afn in activation_fn
        for opt in optimizer
        for me in max_epochs
    ]

    # Fit on training data and transform both train and val
    X_train_lda = grid_pipeline_simple.fit_transform(X_train.values, y_train)
    X_val_lda = grid_pipeline_simple.transform(X_val.values)

    # Manual grid search with scoring collected in a dataframe
    df_simple = []

    # Manual grid search (simple train-val split)
    for params in grid_parameters_simple:
        # Fit ANN with current parameters
        ann_mdl = NeuralNetClassifier(
            module=SimpleANN,
            module__input_dim=X_train_lda.shape[1],
            module__output_dim=len(set(y_train)),
            module__hidden_size=params["hidden_size"],
            module__hidden_layers=params["hidden_layers"],
            module__activation_fn=params["activation_fn"],
            max_epochs=params["max_epochs"],
            lr=params["learning_rate"],
            optimizer=params["optimizer"],
            criterion=cast(Any, nn.CrossEntropyLoss),
            verbose=0,
        )
        ann_mdl.fit(X_train_lda.astype("float32"), y_train.values.astype("long"))

        # Evaluate and penalize overfitting
        train_score = accuracy_score(
            y_train, ann_mdl.predict(X_train_lda.astype("float32"))
        )
        val_score = accuracy_score(y_val, ann_mdl.predict(X_val_lda.astype("float32")))
        gap = abs(train_score - val_score)
        penalized = val_score - pen * gap

        # Append metrics
        df_simple.append(
            {
                "params": params,
                "train_score": train_score,
                "val_score": val_score,
                "gap": gap,
                "score": penalized,
            }
        )
    df_simple = pd.DataFrame(df_simple)

    # Pick best model from penalized score
    best_idx_simple = df_simple["score"].idxmax()
    best_parameters_simple = df_simple.loc[best_idx_simple, "params"]
    performance_simple = {
        "train_acc": df_simple.loc[best_idx_simple, "train_score"],
        "val_acc": df_simple.loc[best_idx_simple, "val_score"],
        "gap": df_simple.loc[best_idx_simple, "gap"],
        "score": df_simple.loc[best_idx_simple, "score"],
    }

    # Refit best model on the full training set
    model_simple = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            (
                "ann",
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=X_train_lda.shape[1],
                    module__output_dim=len(set(y_train)),
                    module__hidden_size=best_parameters_simple["hidden_size"],
                    module__hidden_layers=best_parameters_simple["hidden_layers"],
                    module__activation_fn=best_parameters_simple["activation_fn"],
                    max_epochs=best_parameters_simple["max_epochs"],
                    lr=best_parameters_simple["learning_rate"],
                    optimizer=best_parameters_simple["optimizer"],
                    criterion=cast(Any, nn.CrossEntropyLoss),
                    verbose=0,
                ),
            ),
        ]
    )
    model_simple.fit(X_train.values.astype("float32"), y_train.values.astype("long"))

    # Update models
    models["simple"] = model_simple

    # Report results
    summary = "=" * 50 + "\n"
    summary += "  Best Parameters (SIMPLE + CTL/LDA + ANN)\n"
    summary += "=" * 50 + "\n"
    summary += f"  hidden_layers : {best_parameters_simple['hidden_layers']}\n"
    summary += f"  hidden_size   : {best_parameters_simple['hidden_size']}\n"
    summary += f"  learning_rate : {best_parameters_simple['learning_rate']}\n"
    summary += f"  activation_fn : {best_parameters_simple['activation_fn'].__name__}\n"
    summary += f"  optimizer     : {best_parameters_simple['optimizer'].__name__}\n"
    summary += f"  max_epochs    : {best_parameters_simple['max_epochs']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_simple['train_acc']:.4f}, "
        f"Val Acc: {performance_simple['val_acc']:.4f}, "
        f"Score: {performance_simple['score']:.4f}"
    )

    # Grid search pipeline: CTL → LDA → ANN
    grid_pipeline_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            (
                "ann",
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=len(np.unique(y_temp)) - 1,
                    module__output_dim=len(set(y_temp)),
                    module__hidden_size=32,
                    module__hidden_layers=1,
                    module__activation_fn=nn.ReLU,
                    max_epochs=100,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    criterion=cast(Any, nn.CrossEntropyLoss),
                    verbose=0,
                ),
            ),
        ]
    )

    # Define hyperparameter grid
    grid_parameters_kfold = {
        "ann__module__hidden_layers": hidden_layers,
        "ann__module__hidden_size": hidden_size,
        "ann__module__activation_fn": activation_fn,
        "ann__lr": learning_rate,
        "ann__optimizer": optimizer,
        "ann__max_epochs": max_epochs,
    }

    # Define stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Run stratified K-fold grid search
    grid_search_kfold = GridSearchCV(
        estimator=grid_pipeline_kfold,
        param_grid=grid_parameters_kfold,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid_search_kfold.fit(X_temp.values.astype("float32"), y_temp.values.astype("long"))

    # Penalize overfit models
    df_kfold = pd.DataFrame(grid_search_kfold.cv_results_)
    df_kfold["gap"] = abs(df_kfold["mean_train_score"] - df_kfold["mean_test_score"])
    df_kfold["score"] = df_kfold["mean_test_score"] - pen * df_kfold["gap"]

    # Pick best model from penalized score
    best_idx_kfold = df_kfold["score"].idxmax()
    best_parameters_kfold = df_kfold.loc[best_idx_kfold, "params"]
    performance_kfold = {
        "train_acc": df_kfold.loc[best_idx_kfold, "mean_train_score"],
        "val_acc": df_kfold.loc[best_idx_kfold, "mean_test_score"],
        "gap": df_kfold.loc[best_idx_kfold, "gap"],
        "score": df_kfold.loc[best_idx_kfold, "score"],
    }

    # Refit best model on the full training+val set
    best_parameters_kfold = {
        k.replace("ann__", ""): v for k, v in best_parameters_kfold.items()
    }
    model_kfold = Pipeline(
        [
            ("ctl", CoefficientThresholdLasso()),
            ("lda", LinearDiscriminantAnalysis()),
            (
                "ann",
                NeuralNetClassifier(
                    module=SimpleANN,
                    module__input_dim=len(np.unique(y_temp)) - 1,
                    module__hidden_size=best_parameters_kfold["module__hidden_size"],
                    module__hidden_layers=best_parameters_kfold[
                        "module__hidden_layers"
                    ],
                    module__output_dim=len(set(y_temp)),
                    module__activation_fn=best_parameters_kfold[
                        "module__activation_fn"
                    ],
                    max_epochs=best_parameters_kfold["max_epochs"],
                    lr=best_parameters_kfold["lr"],
                    optimizer=best_parameters_kfold["optimizer"],
                    criterion=cast(Any, nn.CrossEntropyLoss),
                    verbose=0,
                ),
            ),
        ]
    )
    model_kfold.fit(X_temp.values.astype("float32"), y_temp.values.astype("long"))

    # Update models
    models["kfold"] = model_kfold

    # Report results
    summary += "\n\n" + "=" * 50 + "\n"
    summary += "  Best Parameters (KFOLD + CTL/LDA + ANN)\n"
    summary += "=" * 50 + "\n"
    summary += f"  hidden_layers : {best_parameters_kfold['module__hidden_layers']}\n"
    summary += f"  hidden_size   : {best_parameters_kfold['module__hidden_size']}\n"
    summary += f"  learning_rate : {best_parameters_kfold['lr']}\n"
    summary += (
        f"  activation_fn : {best_parameters_kfold['module__activation_fn'].__name__}\n"
    )
    summary += f"  optimizer     : {best_parameters_kfold['optimizer'].__name__}\n"
    summary += f"  max_epochs    : {best_parameters_kfold['max_epochs']}\n"
    summary += "-" * 50 + "\n"
    summary += (
        f"Train Acc: {performance_kfold['train_acc']:.4f}, "
        f"Val Acc: {performance_kfold['val_acc']:.4f}, "
        f"Score: {performance_kfold['score']:.4f}\n"
    )

    return models, df_simple, df_kfold, summary


if __name__ == "__main__":
    # Load dataset
    X_temp = pd.read_csv("../../data/kfold/X_temp_norm.csv").squeeze()
    X_train = pd.read_csv("../../data/simple/X_train_norm.csv").squeeze()
    X_val = pd.read_csv("../../data/simple/X_val_norm.csv").squeeze()
    y_temp = pd.read_csv("../../data/kfold/y_temp_norm.csv").squeeze()
    y_train = pd.read_csv("../../data/simple/y_train_norm.csv").squeeze()
    y_val = pd.read_csv("../../data/simple/y_val_norm.csv").squeeze()

    # Fix for Pyright
    X_temp = cast(pd.DataFrame, X_temp)
    X_train = cast(pd.DataFrame, X_train)
    X_val = cast(pd.DataFrame, X_val)
    y_temp = cast(pd.Series, y_temp)
    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)

    # ANN Hyperparameters
    hidden_layers = [1, 2, 3]
    hidden_size = [32, 100, 128]
    learning_rate = [0.01, 0.001, 0.0001]
    activation_fn = [nn.ReLU, nn.Tanh, nn.Sigmoid]
    optimizer = [torch.optim.Adam, torch.optim.SGD]
    max_epochs = [50, 100, 200]

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run ANN
    models, df_simple, df_kfold, summary = ann(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_temp=X_temp,
        y_temp=y_temp,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
    )

    # Ensure output directory exists
    Path("../../results/models/ann").mkdir(parents=True, exist_ok=True)
    img_dir = Path("../../images/models/")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    Path("../../results/models/ann/ann_summary.txt").write_text(summary)

    # Expand parameters for simple strategy
    df_simple_expanded = df_simple.copy()
    df_simple_expanded["hidden_layers"] = df_simple_expanded["params"].apply(
        lambda d: d["hidden_layers"]
    )
    df_simple_expanded["hidden_size"] = df_simple_expanded["params"].apply(
        lambda d: d["hidden_size"]
    )
    df_simple_expanded["learning_rate"] = df_simple_expanded["params"].apply(
        lambda d: d["learning_rate"]
    )
    df_simple_expanded["activation_fn"] = df_simple_expanded["params"].apply(
        lambda d: d["activation_fn"].__name__
    )
    df_simple_expanded["optimizer"] = df_simple_expanded["params"].apply(
        lambda d: d["optimizer"].__name__
    )
    df_simple_expanded["max_epochs"] = df_simple_expanded["params"].apply(
        lambda d: d["max_epochs"]
    )

    # Expand parameters for kfold strategy
    df_kfold_expanded = df_kfold.copy()
    df_kfold_expanded["hidden_layers"] = df_kfold_expanded[
        "param_ann__module__hidden_layers"
    ]
    df_kfold_expanded["hidden_size"] = df_kfold_expanded[
        "param_ann__module__hidden_size"
    ]
    df_kfold_expanded["learning_rate"] = df_kfold_expanded["param_ann__lr"]
    df_kfold_expanded["activation_fn"] = df_kfold_expanded[
        "param_ann__module__activation_fn"
    ].apply(lambda fn: fn.__name__)
    df_kfold_expanded["optimizer"] = df_kfold_expanded["param_ann__optimizer"].apply(
        lambda opt: opt.__name__
    )
    df_kfold_expanded["max_epochs"] = df_kfold_expanded["param_ann__max_epochs"]

    # Custom style
    plt.style.use("../../config/custom_style.mplstyle")

    # Hyperparameters evolution plot
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Score vs hidden_layers
    sns.lineplot(
        data=df_simple_expanded,
        x="hidden_layers",
        y="score",
        ax=axs[0],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="hidden_layers",
        y="score",
        ax=axs[0],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[0].set_title("hidden_layers")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Mean Score")
    axs[0].legend(loc="upper right")

    # Score vs hidden_size
    sns.lineplot(
        data=df_simple_expanded,
        x="hidden_size",
        y="score",
        ax=axs[1],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="hidden_size",
        y="score",
        ax=axs[1],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[1].set_title("hidden_size")
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Mean Score")
    axs[1].legend(loc="upper right")

    # Score vs learning_rate
    sns.lineplot(
        data=df_simple_expanded,
        x="learning_rate",
        y="score",
        ax=axs[2],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="learning_rate",
        y="score",
        ax=axs[2],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[2].set_title("learning_rate")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Mean Score")
    axs[2].legend(loc="upper right")

    # Score vs activation_fn
    sns.lineplot(
        data=df_simple_expanded,
        x="activation_fn",
        y="score",
        ax=axs[3],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="activation_fn",
        y="score",
        ax=axs[3],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[3].tick_params(axis="x", rotation=30)
    axs[3].set_title("activation_fn")
    axs[3].set_xlabel("")
    axs[3].set_ylabel("Mean Score")
    axs[3].legend(loc="upper right")

    # Score vs optimizer
    sns.lineplot(
        data=df_simple_expanded,
        x="optimizer",
        y="score",
        ax=axs[4],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="optimizer",
        y="score",
        ax=axs[4],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[4].tick_params(axis="x", rotation=30)
    axs[4].set_title("optimizer")
    axs[4].set_xlabel("")
    axs[4].legend(loc="upper right")

    # Score vs max_epochs
    sns.lineplot(
        data=df_simple_expanded,
        x="max_epochs",
        y="score",
        ax=axs[5],
        marker="o",
        label="Simple",
    )
    sns.lineplot(
        data=df_kfold_expanded,
        x="max_epochs",
        y="score",
        ax=axs[5],
        marker="X",
        linestyle="--",
        label="K-Fold",
    )
    axs[5].set_title("max_epochs")
    axs[5].set_xlabel("")
    axs[5].set_ylabel("Mean Score")
    axs[5].legend(loc="upper right")

    # Save figure
    plt.tight_layout()
    plt.savefig(img_dir / "ann_hyperparameters_evolution.png")
    plt.show()
