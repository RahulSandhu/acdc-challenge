from typing import Any, Dict, Optional, Tuple

import torch.nn as nn


def parse_best_params(
    file_path: str,
    simple: bool = True,
    simple_lines: Optional[Tuple[int, int]] = (4, 7),
    kfold_lines: Optional[Tuple[int, int]] = (13, 16),
) -> Dict[str, Any]:
    """
    Parse best model parameters from a summary text file.

    Inputs:
        - file_path (str): Path to the summary text file.
        - simple (bool): Whether to extract parameters from the 'simple' model
          block or from the 'kfold' model block. Defaults to True.
        - simple_lines (tuple): Line range (start, end) for simple block. Defaults to (4, 7).
        - kfold_lines (tuple): Line range (start, end) for kfold block. Defaults to (13, 16).

    Outputs:
        - Dict[str, Any]: A dictionary with parameter names as keys and
          parameter values as values.
    """
    params = {}

    # Map string to actual nn.Module activation classes
    act_fn_mapping = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Choose correct block
        start, end = simple_lines if simple else kfold_lines
        param_lines = lines[start:end]

        for line in param_lines:
            parts = line.strip().split(':')

            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()

                # Special case: activation_fn must map to nn.Module
                if key == 'activation_fn':
                    value_converted = act_fn_mapping.get(
                        value, value
                    )  # fallback to original if unknown
                else:
                    if value.lower() == 'none':
                        value_converted = None
                    else:
                        try:
                            value_converted = int(value)
                        except ValueError:
                            try:
                                value_converted = float(value)
                            except ValueError:
                                value_converted = value

                params[key] = value_converted

    return params


if __name__ == "__main__":
    # Example usage with defaults
    best_params_knn_kfold = parse_best_params(
        "../../results/models/knn/knn_summary.txt", simple=False
    )
    print("\nKFold Best Params:")
    print(best_params_knn_kfold)

    # Example usage for ANN
    best_params_ann_simple = parse_best_params(
        "../../results/models/ann/ann_summary.txt",
        simple=True,
        simple_lines=(3, 7),
        kfold_lines=(13, 17),
    )
    print("\nANN Simple Best Params:")
    print(best_params_ann_simple)
