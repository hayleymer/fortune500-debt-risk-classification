# Fortune 500 Debt Risk Classification

A supervised **multi-class classification** project that assigns Fortune 500 companies to **debt-risk bands** using financial fundamentals—while **intentionally excluding direct leverage components** (e.g., *Total Liabilities*, *Total Assets*) from the model features to test whether *indirect indicators* can still predict leverage risk. :contentReference[oaicite:0]{index=0}

> **Note:** This is a modelling / analytics portfolio project and is **not financial advice**.

---

## Overview

### Business objective
Build a model that classifies companies into **four debt-risk bands** based on an engineered **Debt Ratio** target:
- Debt Ratio = `Total Liabilities / Total Assets` :contentReference[oaicite:1]{index=1}

To reduce leakage, **Total Assets** and **Total Liabilities** are used **only** to create the target, and are **excluded** from predictors. :contentReference[oaicite:2]{index=2}

### Initial hypothesis → what actually drove the signal
- Hypothesis: **lower profitability** would associate with **higher leverage risk**
- Finding: **financing burden (Interest Expense)** emerged as a stronger structural driver than Profit Margin alone :contentReference[oaicite:3]{index=3}

---

## Data

### Inputs
This project expects two CSVs:
- `fundamentals.csv`
- `securities.csv` :contentReference[oaicite:4]{index=4}

### Filtering window
- Five-year window: **2012–2016** :contentReference[oaicite:5]{index=5}

### Dataset shape (post-filter)
- **1,776 rows** (company-year observations)
- **83 variables** (mixed numeric + categorical) :contentReference[oaicite:6]{index=6}

### Missing data handling (high level)
- Dropped frequently-missing liquidity ratios (Quick/Current/Cash)
- Dropped sparse shareholder metrics (e.g., EPS, Estimated Shares Outstanding)
- Dropped “For Year” as redundant with Period Ending
- Negative values were inspected and treated as valid for the variables used :contentReference[oaicite:7]{index=7}

---

## Target: Debt Risk Bands

Debt Ratio quartiles are converted into **four balanced classes**:
- Very Low, Low, High, Very High (444 rows each) :contentReference[oaicite:8]{index=8}

This framing supports multi-class classification while keeping class sizes balanced.

---

## Exploratory Analysis Highlights

- Debt Ratio is **right-skewed** with high-leverage outliers :contentReference[oaicite:9]{index=9}  
- Debt ratios differ **structurally by sector**:
  - Higher leverage: Financials, Telecom, Utilities, Real Estate
  - Lower leverage: Information Technology (and other consumer/service industries) :contentReference[oaicite:10]{index=10}
- Profit Margin alone shows **no strong linear relationship** with Debt Ratio :contentReference[oaicite:11]{index=11}
- Leverage levels are broadly stable over time, with some mid-window risk concentration :contentReference[oaicite:12]{index=12}

---

## Features Used (Leakage-Aware)

Predictors were chosen to reflect profitability, operating performance, scale, time, and sector structure:

- **Total Revenue**
- **Interest Expense**
- **Depreciation**
- **Profit Margin**
- **Year**
- **GICS Sector** (one-hot encoded) :contentReference[oaicite:13]{index=13}

> **Excluded to avoid structural leakage:** Total Assets, Total Liabilities :contentReference[oaicite:14]{index=14}

---

## Modelling

### Train / test split
- 70/30 split with **stratification** to preserve class balance :contentReference[oaicite:15]{index=15}

### Models evaluated
- Random Forest
- Gradient Boosting
- K-Nearest Neighbours (with StandardScaler) :contentReference[oaicite:16]{index=16}

### Metrics
- Accuracy
- Macro Recall (≈ balanced accuracy)
- Macro F1 :contentReference[oaicite:17]{index=17}

---

## Results

| Model | Accuracy | Macro Recall | Macro F1 |
|------|----------|--------------|----------|
| Random Forest | **0.662** | **0.662** | **0.662** |
| Gradient Boosting | 0.591 | 0.591 | 0.591 |
| KNN | 0.415 | 0.415 | 0.419 | :contentReference[oaicite:18]{index=18}

**Takeaway:** Tree-based models outperform KNN for this multi-class, mixed-signal structure; **Random Forest** is the most reliable overall performer. :contentReference[oaicite:19]{index=19}

---

## Random Forest: Tuning + Interpretability

### Hyperparameter search
A GridSearchCV run confirmed the baseline RF settings were already near-optimal:
- Best params included `n_estimators=200`, default depth, and default split/leaf settings :contentReference[oaicite:20]{index=20}

### Feature importance (Top drivers)
Random Forest feature importance shows the strongest predictors are:
1. **Interest Expense**
2. **Total Revenue**
3. **Depreciation**
4. **Profit Margin**
…with smaller but meaningful contributions from **Year** and sector indicators (e.g., Financials, Utilities). :contentReference[oaicite:21]{index=21}

Interpretation: The model separates risk bands primarily through **financing burden and firm scale**, rather than profitability alone. :contentReference[oaicite:22]{index=22}

---

## Statistical Cross-Check (ANOVA)

A one-way ANOVA tested whether **Interest Expense** differs by debt-risk band:
- Result: **significant differences across groups** (p-value ≪ 0.001), supporting the model’s emphasis on Interest Expense. :contentReference[oaicite:23]{index=23}

---

## How to Run

### 1) Environment
This project uses:
- Python
- numpy, pandas, matplotlib
- scikit-learn
- scipy :contentReference[oaicite:24]{index=24}

Install dependencies (example):
```bash
pip install numpy pandas matplotlib scikit-learn scipy

### 2) Data

Place the required CSVs where your code expects them:

- `fundamentals.csv`
- `securities.csv`

### 3) Execute

Run the notebook/script to reproduce:

- Data cleaning + engineering (Debt Ratio, sector merge, year filter)
- Debt risk labelling (quartiles)
- One-hot encoding
- Model training + evaluation
- Feature importance + ANOVA

## Project Notes / Next Improvements

If you extend this work, good next steps include:

- Use cross-validation (not just a single train/test split) for more stable estimates
- Try a time-based split (train on earlier years, test on later years) to better reflect real deployment
- Add richer features (cashflow-based metrics, coverage ratios) with careful leakage checks
- Calibrate class probabilities and explore cost-sensitive errors (e.g., misclassifying “Very High” as “Very Low” is worse than adjacent mistakes)

## Acknowledgements

Dataset sources and sector labels are derived from the provided `fundamentals.csv` and `securities.csv` inputs used in the analysis.
