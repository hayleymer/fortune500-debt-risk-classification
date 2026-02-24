# Fortune 500 Debt Risk Classification
## Overview
This project develops a supervised machine learning model to classify Fortune 500 companies into debt-risk bands using financial fundamentals.

The objective was to determine whether leverage risk can be predicted from indirect financial indicators, while intentionally excluding direct leverage components from model features to avoid leakage.

This project demonstrates an end-to-end data science workflow including data preparation, feature engineering, model training, evaluation, and interpretation.

## Business Problem
Debt risk is a critical factor in investment screening, credit analysis, and corporate risk assessment.

However, direct leverage measures (such as total liabilities and total assets) may not always be available, reliable, or appropriate for predictive modelling.

This project explores whether underlying financial performance indicators can be used to classify companies into leverage risk categories.

This approach simulates real-world conditions where analysts must infer risk from incomplete financial information.

## Key modelling decisions:
Excluded Total Assets and Liabilities from predictors to avoid leakage

Engineered debt ratio only for target creation

Compared multiple classifiers

Used cross-validation for robust evaluation

## Dataset
Source: Fortune 500 financial fundamentals dataset

Each observation represents a company, including features such as:
Revenue
Profit margin
Operating income
Interest expense
Market value
Sector
Other financial performance indicators
Target variable (engineered):
Debt Ratio = Total Liabilities / Total Assets

This ratio was used to assign companies into four risk bands:
Low risk
Moderate risk
High risk
Very high risk

To prevent data leakage, Total Assets and Total Liabilities were excluded from the model predictors.

## Approach
## 1. Data Preparation

Loaded and cleaned financial dataset
Handled missing values
Created engineered target variable (Debt Ratio risk bands)
Removed leakage variables from predictors

## 2. Feature Selection

Financial performance indicators were used as predictors, including:
Interest expense
Profitability metrics
Revenue and income measures
Market and operational indicators

## 3. Model Training

Multiple classification algorithms were evaluated:
Random Forest
Gradient Boosting
K-Nearest Neighbours
Data was split into training and test sets for unbiased evaluation.

## 4. Model Evaluation

Models were evaluated using:
Accuracy
Macro F1 score
Recall
Cross-validation was used to improve robustness.

## 5. Model Interpretation

Feature importance analysis was conducted to understand which financial indicators most strongly influence debt risk classification.

## Key Findings

Interest expense emerged as one of the strongest predictors of leverage risk, suggesting that financing burden provides an indirect signal of company debt exposure.

This finding supports the hypothesis that leverage risk can be inferred from structural financial indicators, even when direct leverage measures are excluded.

## Tech Stack

Python

pandas

numpy

scikit-learn

matplotlib

