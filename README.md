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

## 📁 Dataset

The project uses the ACDC (Automated Cardiac Diagnosis Challenge) dataset,
which contains cardiac MRI images with extracted radiomic features for
multi-class classification of cardiac pathologies. Key features include:

- **Shape features**: Geometric characteristics of cardiac structures (volume,
  surface area, compactness)
- **First-order statistics**: Intensity histogram-based features (mean,
  variance, skewness, kurtosis)
- **Texture features**: Gray-level co-occurrence matrix (GLCM), gray-level
  run-length matrix (GLRLM), gray-level size zone matrix (GLSZM)
- **Wavelet features**: Multi-scale decomposition features capturing different
  frequency components
- **Clinical labels**: Multi-class cardiac pathology classifications including
  normal, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal
  right ventricle

The analysis employs multiple machine learning approaches including **Artificial
Neural Networks (ANN)**, **K-Nearest Neighbors (KNN)**, **Random Forest (RF)**,
and **Support Vector Machines (SVM)** with Lasso-based feature selection to
classify cardiac conditions from radiomic features.

## 📊 Results

- Multiple models evaluated under simple train-validation and k-fold
  cross-validation settings
- Comprehensive performance metrics including accuracy, precision, recall, and
  F1-score for each model
- Feature selection via Lasso regression improved model interpretability and
  generalization
- Best-performing models saved and documented with detailed classification
  reports
- Normalized features demonstrated superior performance compared to raw radiomic
  features

## 🎓 Acknowledgements

- [Kaggle ACDC Dataset](https://www.kaggle.com/datasets/anhoangvo/acdc-dataset)
  – Automated Cardiac Diagnosis Challenge dataset
- Developed as part of the Machine Learning course in the Master in Health Data
  Science program at Universitat Rovira i Virgili (URV)

</div>
