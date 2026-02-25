# Fortune 500 Debt Risk Classification
## Business Problem

Corporate leverage risk is a critical factor in financial risk assessment, credit analysis, and investment screening. However, direct leverage measures such as Total Liabilities and Total Assets can introduce target leakage when used as model predictors.

This project develops a supervised machine learning model to classify Fortune 500 companies into debt-risk bands using indirect financial indicators, excluding direct leverage components from the predictors.

The objective is to evaluate whether structural financial characteristics such as financing burden, profitability, and operational scale can reliably predict leverage risk.

Initial hypothesis: lower profitability would associate with higher leverage risk.
Key finding: financing burden (Interest Expense) emerged as a stronger structural predictor.

## Dataset

### The dataset consists of financial fundamentals for Fortune 500 companies over a five-year period (2012–2016), comprising:

1,776 company-year observations

83 financial variables

Financial performance, scale, and sector classification features

### Key predictor variables used:

Total Revenue

Interest Expense

Depreciation

Profit Margin

Year

GICS Sector (one-hot encoded)

### Target variable:

Debt Ratio = Total Liabilities / Total Assets

### Companies were classified into four balanced risk categories based on quartiles:

* Very Low
* Low
* High
* Very High

Each class contains 444 observations.

Direct leverage variables were excluded from predictors to prevent data leakage.

## Approach
### Data Preparation

Merged fundamentals and securities datasets using ticker symbol

Filtered data to consistent five-year analysis window

Engineered Debt Ratio target variable

Converted categorical sector data using one-hot encoding

Removed variables with substantial missing data

### Exploratory Data Analysis

Examined Debt Ratio distribution and outliers

Analysed leverage variation across sectors

Evaluated relationships between financial indicators and leverage risk

Sector analysis showed structurally higher leverage in Financials, Utilities, and Telecommunications sectors.

### Model Development

#### Three supervised classification models were trained and compared:

Random Forest Classifier

Gradient Boosting Classifier

K-Nearest Neighbours Classifier

### Pipeline included:

Train/test split (70/30) with stratified sampling

Feature scaling applied for KNN

Cross-validated hyperparameter tuning using GridSearchCV

### Model Evaluation

#### Models were evaluated using:

Accuracy

Macro Recall

Macro F1 Score

These metrics ensure balanced performance evaluation across all risk classes.

## Results

### Model performance on test data:

Model	Accuracy	Macro Recall	Macro F1
Random Forest	66.2%	66.2%	66.2%
Gradient Boosting	59.1%	59.1%	59.1%
K-Nearest Neighbours	41.5%	41.5%	41.9%

Random Forest demonstrated the strongest performance and most reliable classification across all risk bands.

Hyperparameter tuning confirmed the baseline Random Forest configuration was near optimal.

## Key Findings

### Interest Expense was the strongest predictor of debt risk classification, followed by:

Total Revenue

Depreciation

Profit Margin

Sector membership also contributed meaningful predictive signal.

Feature importance analysis showed that financing burden and firm scale were the most influential structural indicators.

Statistical validation using one-way ANOVA confirmed that Interest Expense differs significantly across debt risk categories (p < 0.001), supporting its predictive relevance.

This demonstrates that leverage risk can be inferred from indirect financial indicators without directly using leverage variables.

## Tech Stack

Python
pandas
numpy
scikit-learn
matplotlib
scipy

## Machine learning techniques:

Supervised classification

Feature engineering

One-hot encoding

Hyperparameter tuning (GridSearchCV)

Feature importance analysis

Statistical hypothesis testing (ANOVA)

## How to Run

### Clone repository:

git clone https://github.com/hayleymer/fortune500-debt-risk-classification.git

### Navigate to project folder:

cd fortune500-debt-risk-classification

### Install dependencies:

pip install -r requirements.txt

Run the notebook or training script to reproduce analysis.
