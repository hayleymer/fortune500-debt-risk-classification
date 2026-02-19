Fortune 500 Debt Risk Classification
This project builds a supervised machine‑learning model to classify Fortune 500 companies into debt‑risk bands using financial fundamentals. The analysis intentionally excludes direct leverage measures (Total Assets, Total Liabilities) to test whether indirect indicators can meaningfully predict leverage‑based risk.

The dataset covers 1,776 company‑year observations from 2012–2016 and includes full data preparation, exploratory analysis, feature engineering, model development, and statistical validation.

Key Features
Construction of four debt‑risk bands using Debt Ratio quartiles

Comprehensive EDA including sector‑level leverage patterns

Correlation analysis using only numeric predictors

Random Forest, Gradient Boosting, and KNN model comparison

Balanced train/test split with stratification

Random Forest selected as the final model (~66% accuracy)

Feature‑importance analysis showing Interest Expense, Total Revenue, and Depreciation as top predictors

Supplementary ANOVA confirming significant differences in Interest Expense across risk bands

Repository Structure
Code
├── notebook.ipynb        # Full analysis and modelling workflow
├── fundamentals.csv      # Financial statement data (if included)
├── securities.csv        # Company metadata (if included)
├── README.md             # Project overview
└── requirements.txt      # Python dependencies
Methods & Tools
Python (NumPy, pandas, matplotlib)

scikit‑learn (RandomForestClassifier, GradientBoostingClassifier, KNN, GridSearchCV)

Statistical testing (SciPy ANOVA)

One‑hot encoding for sector features

Balanced multi‑class evaluation using macro recall and macro F1

Results Summary
Random Forest achieved the strongest performance:

Accuracy: 0.66

Macro Recall: 0.66

Macro F1: 0.66

Gradient Boosting performed moderately (~0.59)

KNN struggled with the multi‑class structure (~0.41)

Feature importances highlight financing burden and operational scale as the most influential predictors of leverage risk.

License
This project is released for educational and research purposes.
Feel free to fork, modify, and build upon it.
