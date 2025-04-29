from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.optim as optim


def parse_best_params(
    file_path: str,
    line_range: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Parse best model parameters from a summary text file within a specified
    line range.

    Inputs:
        - file_path (str): Path to the summary text file.
        - line_range (tuple): Line range (start, end) to extract parameters.

    Outputs:
        - Dict[str, Any]: A dictionary with parameter names as keys and
          parameter values as values.
    """
    # Initialize dictionary to hold parameters
    params = {}

    # Mapping activation function names to nn classes
    act_fn_mapping = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
    }

    # Mapping optimizer names to torch.optim classes
    optimizer_mapping = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
    }

    # Open the summary text file and read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Fix for Pyright
        assert line_range is not None

        # Grab the selected block of lines
        start, end = line_range
        param_lines = lines[start : end + 1]

        # Parse each line in the selected block
        for line in param_lines:
            parts = line.strip().split(':')

            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()

                # Activation function mapping
                if key == 'activation_fn':
                    value_converted = act_fn_mapping.get(value, value)
                # Optimizer mapping
                elif key == 'optimizer':
                    value_converted = optimizer_mapping.get(value, value)
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

                # Save the parsed key-value pair
                params[key] = value_converted

    return params


if __name__ == "__main__":
    # Example usage for KNN KFold model
    best_params_knn_kfold = parse_best_params(
        "../../results/models/knn/knn_summary.txt",
        line_range=(12, 15),
    )
    print("\nKNN KFold Best Params:")
    print(best_params_knn_kfold)
