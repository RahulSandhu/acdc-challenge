<div align="justify">

# ACDC Radiomics Challenge

This repository contains all the code, data, and results for a radiomics-based
classification challenge using the ACDC dataset. We developed and evaluated
several machine learning models (ANN, KNN, RF, SVM) for multi-class
classification of cardiac pathologies using radiomic features extracted from
cardiac MRI images.

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/RahulSandhu/acdc-challenge
   cd acdc-challenge
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Code

- `code/eda/`: Exploratory Data Analysis scripts
- `code/models/`: Machine learning model definitions (ANN, KNN, RF, SVM)
- `code/metrics/`: Custom metrics and evaluation functions
- `code/utils/`: Utility functions (e.g., Lasso feature selection, parsing best
parameters)

## 📁 Data

- `data/datasets/`
  - `norm_acdc_radiomics.csv`: Normalized dataset used for model training and
  validation
  - `raw_acdc_radiomics.csv`: Raw extracted radiomics features
- `data/simple/`
  - `X_train_norm.csv`, `X_train_raw.csv`, ..., `y_val_norm.csv`,
  `y_val_raw.csv`
- `data/kfold/`
  - `X_temp_norm.csv`, `X_test_norm.csv`, `y_temp_norm.csv`, `y_test_norm.csv`

## 📊 Results

- `results/models/`: Trained models saved as `.pkl` files
- `results/metrics/`: Classification reports (CSV and TXT) for each model under
simple and k-fold settings

## 📈 Figures

All generated figures for EDA, feature selection, model training, and
evaluations are stored under the `images/` directory:

- EDA visualizations (BMI, multicollinearity, outliers, etc.)
- Feature selection curves (MSE vs lambda/threshold)
- Model performance metrics (accuracy, precision, recall, F1-score)
- Hyperparameter tuning visualizations
- Overall workflow schemas

## 📄 Report

A detailed report is available in `report/main.pdf`. The LaTeX sources used to
generate the report are also provided under the `report/` folder.

## 🎓 Acknowledgements

- ACDC Challenge Dataset
- Project developed as part of the Health Data Science Master's program at
Universitat Rovira i Virgili (URV)

</div>
