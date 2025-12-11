# Random Forest Income Classification – Tutorial

A machine learning tutorial using the **Random Forest** algorithm to predict whether a person earns **> $50K** per year using the **Adult Census Income** dataset.  
Designed for MSc-level coursework (Machine Learning and Neural Networks – University of Hertfordshire).

---

## Project Overview

This tutorial demonstrates a complete supervised learning pipeline:

Random Forest fundamentals → data exploration → preprocessing mixed features → baseline model → hyperparameter tuning → evaluation → feature importance → fairness considerations.

The focus is on a reproducible scikit-learn pipeline with clear metrics and visualisations.

---

## Tutorial Flow (Summary)

### 1. Introduction  
Defines the income classification problem and introduces decision trees and Random Forests.

### 2. Understanding Random Forests  
Explains bagging, random feature sampling, majority voting, and key hyperparameters.

### 3. Visual Overview  
Lists all tutorial figures: class distribution, confusion matrices, ROC curves, metric comparison, feature importance.

### 4. Exploratory Data Analysis  
Covers class imbalance, missing values, and initial insights into demographic and employment variables.

### 5. Data Preprocessing  
Includes:
- Handling “?” values  
- Numerical scaling  
- Categorical one-hot encoding  
- ColumnTransformer  
- Stratified train–test split  
All implemented in a scikit-learn Pipeline.

### 6. Baseline Model  
Trains an initial Random Forest and reports accuracy, precision, recall, F1-macro, and confusion matrix.

### 7. Hyperparameter Tuning  
Uses RandomizedSearchCV to tune depth, estimators, feature sampling, and split criteria.  
Evaluates improvements using F1-macro and ROC-AUC.

### 8. Model Visualisations  
Generates confusion matrices, ROC curves, metric comparison plots, and permutation feature importance.

### 9. Interpretation & Fairness  
Explains why key features (education, marital status, capital gain, etc.) influence income and discusses ethical risks of demographic features.

### 10. Discussion & Conclusion  
Highlights main findings, model improvements after tuning, and the need for fairness checks.

---

## Project Structure

```

RandomForest-Income-Classification/
│
├── Random_Forest_Income_Classification_Tutorial.ipynb
├── adult.csv
├── README.md
├── Random Forest Classification Using the Adult Census Income Dataset.docx
├── requirements.txt
└── figures/
├── class_distribution.png
├── confusion_matrix_baseline.png
├── confusion_matrix_tuned.png
├── roc_curves_baseline_vs_tuned.png
├── metrics_comparison_baseline_vs_tuned.png
└── permutation_importance_top_features.png

```
