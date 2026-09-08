# StreamSense

## Customer Churn Prediction, Risk Scoring and Retention Optimization

StreamSense is an end-to-end machine learning project designed to predict customer churn, estimate individual customer churn risk, and prioritize customers for retention interventions.

## Project Objectives

- Predict customer churn
- Compare multiple machine learning models
- Calibrate churn probabilities
- Assign customers to risk bands
- Estimate customer value
- Calculate retention priority
- Segment customers for retention strategies
- Simulate retention strategies
- Provide explainability using SHAP

## Machine Learning Models

The project evaluates:

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression achieved the highest original cross-validated ROC-AUC, while XGBoost provided competitive performance and was subsequently calibrated to improve the reliability of churn probability estimates.

## Pipeline

Customer Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Model Comparison
↓
XGBoost
↓
Probability Calibration
↓
Churn Probability
↓
Risk Band
↓
Customer Value
↓
Retention Priority Score
↓
Retention Strategy

## Risk Bands

| Churn Probability | Risk |
|---|---|
| < 0.30 | Low |
| 0.30–0.59 | Medium |
| 0.60–0.79 | High |
| ≥ 0.80 | Critical |

## Retention Priority

Customers are ranked using a retention priority score combining:

- Churn probability
- Customer value

This allows the system to prioritize customers who are both likely to churn and valuable to retain.

## Explainability

SHAP is used to identify the features that contribute most strongly to churn predictions.

## Project Structure

```text
StreamSense/
├── data/
├── notebooks/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore