# ACDC Radiomics Classification Challenge

This repository contains all the code, data, and results for a radiomics-based
classification challenge using the ACDC dataset. We developed and evaluated
several machine learning models (ANN, KNN, RF, SVM) for multi-class
classification of cardiac pathologies using radiomic features extracted from
cardiac MRI images.

## 📂 Project Structure

```
├── code/               # Source code (EDA, models, metrics, utilities)
├── data/               # Normalized and raw datasets, train/val/test sets
├── images/             # Figures for EDA, model metrics, and workflow schemas
├── misc/               # Additional files (guides, custom Matplotlib style, references)
├── report/             # LaTeX files for project report
├── results/            # Saved models, metrics, and evaluation results
├── LICENSE             # License file
├── pyproject.toml      # Project configuration
├── README.md           # Project overview 
├── requirements.txt    # Python dependencies
```

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
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

## 🕹️ Code Modules

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
- `data/testing/`
  - `X_test.csv`, `y_test.csv`: Test datasets for final model evaluation

## 📊 Results

- `results/models/`: Trained models saved as `.pkl` files
- `results/metrics/`: Classification reports (CSV and TXT) for each model under
simple and k-fold settings

## 🌐 Figures

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

## 📚 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## 🖊️ Acknowledgements

- ACDC Challenge Dataset
- Project developed as part of the Health Data Science Master's program at
Universitat Rovira i Virgili (URV)
