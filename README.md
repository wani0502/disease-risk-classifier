# Disease Risk Classifier

An end-to-end machine learning project for predicting diabetes risk using health and lifestyle indicators. The project covers the complete ML workflow, including exploratory data analysis, preprocessing, class-imbalance handling, model comparison, hyperparameter tuning, threshold optimization, explainability, and deployment with Streamlit.

## Live Demo

[Open the Disease Risk Classifier](https://disease-risk-classifier.streamlit.app/)

## Project Overview

Diabetes risk prediction is a binary classification problem with a highly imbalanced target distribution. In the dataset, approximately 84.2% of observations belong to the No Diabetes class while 15.8% belong to the At Risk class after converting the original target into a binary classification problem.

Because of this imbalance, accuracy alone is not a reliable evaluation metric. The project therefore focuses on ROC-AUC, PR-AUC, precision, recall, and F1-score, with additional threshold optimization to find a suitable balance between precision and recall.

The final model is an XGBoost classifier with an optimized classification threshold of 0.65.

## Dataset

The project uses the CDC Diabetes Health Indicators dataset.

The original dataset contains:

- 253,680 observations
- Health, lifestyle, demographic, and socioeconomic indicators
- A highly imbalanced diabetes-related target

### Features Used

The final model uses 21 features:

- BMI
- MentHlth
- PhysHlth
- Age
- HighBP
- HighChol
- CholCheck
- Smoker
- Stroke
- HeartDiseaseorAttack
- PhysActivity
- Fruits
- Veggies
- HvyAlcoholConsump
- AnyHealthcare
- NoDocbcCost
- DiffWalk
- Sex
- GenHlth
- Education
- Income

## Exploratory Data Analysis

EDA identified several important risk signals.

The strongest relationships with the target were observed for:

1. GenHlth
2. HighBP
3. BMI
4. DiffWalk
5. HighChol
6. Age
7. HeartDiseaseorAttack

People in the diabetes-risk group showed higher average BMI, higher prevalence of high blood pressure and high cholesterol, greater difficulty walking, and poorer self-reported general health.

Age and physical health also showed clear trends across the target groups.

Multicollinearity was not a major issue. The strongest feature correlation observed was between GenHlth and PhysHlth at approximately 0.52.

## Machine Learning Pipeline

The project follows this workflow:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Feature Preprocessing
     ↓
Class Imbalance Handling
     ↓
Model Training
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Model Comparison
     ↓
Threshold Optimization
     ↓
SHAP Explainability
     ↓
Streamlit Deployment
