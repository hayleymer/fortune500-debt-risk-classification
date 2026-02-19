Fortune 500 Debt Risk Classification
This project builds a supervised machine‑learning model to classify Fortune 500 companies into debt‑risk bands using financial fundamentals. The analysis intentionally excludes direct leverage measures (Total Assets, Total Liabilities) to test whether indirect indicators can meaningfully predict leverage‑based risk. The dataset covers 1,776 company‑year observations from 2012–2016.

Key Features
Construction of four debt‑risk bands using Debt Ratio quartiles

Full data cleaning, feature engineering, and exploratory analysis

Sector‑level leverage comparisons across five years

Correlation analysis using only numeric predictors

Model comparison: Random Forest, Gradient Boosting, KNN

Balanced evaluation using accuracy, macro recall, and macro F1

Random Forest selected as the final model (~66% accuracy)

Feature‑importance analysis highlighting financing burden and scale

ANOVA confirming significant differences in Interest Expense across risk bands

Methods & Tools
Python (NumPy, pandas, matplotlib)

scikit‑learn (RandomForestClassifier, GradientBoostingClassifier, KNN, GridSearchCV)

One‑hot encoding for categorical variables

Stratified train/test split

Statistical testing with SciPy

Results Summary
Random Forest achieved the strongest performance:

Accuracy: 0.66

Macro Recall: 0.66

Macro F1: 0.66

Gradient Boosting performed moderately (~0.59)

KNN performed weakest (~0.41)

Feature importances show that Interest Expense, Total Revenue, and Depreciation are the most influential predictors, reflecting financing burden, firm scale, and asset intensity.

Repository Structure
Code
├── notebook.ipynb        # Full analysis and modelling workflow
├── fundamentals.csv      # Financial statement data (if included)
├── securities.csv        # Company metadata (if included)
├── README.md             # Project overview
└── requirements.txt      # Python dependencies
How to Run
Install dependencies:
pip install -r requirements.txt

Open the notebook:
jupyter notebook notebook.ipynb

Run all cells to reproduce the analysis.

License
This project is available for educational and research use.

Author
Analysis, modelling, and documentation by Hayley.
