from typing import List, Sequence

import pandas as pd


def generate_short_labels(
    features: Sequence[str], max_len: int = 15
) -> List[str]:
    """
    Generate short feature labels by removing the 'original_' prefix and
    truncating the label to a maximum length.

    Inputs:
        - features (Sequence[str]): A list or sequence of feature names.
        - max_len (int): Maximum length for each output label. Defaults to 15.

    Outputs:
        - List[str]: A list of shortened labels with 'original_' removed and
          long names truncated.
    """
    # Initialize list for short labels
    short_labels = []

    # Iterate over features
    for f in features:
        # Remove 'original_' prefix and shorten to 'max_len'
        clean = f.replace("original_", "")
        label = clean if len(clean) <= max_len else clean[:max_len] + "..."
        short_labels.append(label)

    return short_labels


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/datasets/raw_acdc_radiomics.csv")

    # Separate features and classes
    X = df.drop(columns=["class"])
    y = df["class"]

    # Generate short labels
    short_labels = generate_short_labels(list(X.columns))
    print(short_labels[1:6])
