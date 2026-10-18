# Applied Machine Learning & Algorithmic Foundations

Practical implementations and comparative studies of foundational machine learning algorithms applied to synthetic and structured real-world datasets.

This repository organizes laboratory works exploring core predictive modeling techniques: regression, probabilistic classification, regularized linear classification, and ensemble learning.

---

## 📚 Laboratory Projects Overview

### [01. Linear Regression — Salary Prediction](./01_linear_regression)
* **Task:** Univariate regression modeling to predict employee compensation based on age and experience.
* **Technique:** Ordinary Least Squares (OLS) via `sklearn.linear_model.LinearRegression`.
* **Key Findings:**
  * Derived model equation: $\hat{y} = 94,067.50 + 6,141.62 \cdot \text{Age}$.
  * Model achieved a coefficient of determination of $R^2 = 0.8926$, showing strong linear correlation between age/tenure and compensation levels.
* **Artifacts:**
  * [salary_linear_regression.ipynb](./01_linear_regression/salary_linear_regression.ipynb)
  * [data.csv](./01_linear_regression/data.csv)

---

### [02. Naive Bayes Classification — Customer Purchase Prediction](./02_naive_bayes_classification)
* **Task:** Binary classification to forecast whether an e-commerce visitor will complete a purchase based on session activity.
* **Features:** Visits last month, cart additions, average order value, total past orders, days since last purchase, and demographics.
* **Technique:** `sklearn.naive_bayes.GaussianNB`.
* **Key Findings:**
  * Evaluated across train/test splits (80/20, 70/30, 90/10).
  * Feature correlation highlighted `cart_adds` and `visits_last_month` as the primary discriminative signals.
  * Evaluated with classification reports, confusion matrix heatmaps, and probability thresholds.
* **Artifacts:**
  * [customer_purchase_prediction.ipynb](./02_naive_bayes_classification/customer_purchase_prediction.ipynb)
  * [data.csv](./02_naive_bayes_classification/data.csv)

---

### [03. Logistic Regression — Email Spam Detection](./03_logistic_regression_spam)
* **Task:** Binary classification to distinguish spam emails from legitimate messages based on structural email metadata.
* **Features:** Message length, link count, exclamation mark frequency, uppercase word counts, spam lexicon matches, currency symbols, and attachments.
* **Technique:** L2-regularized `LogisticRegression` paired with `StandardScaler` feature normalization.
* **Key Findings:**
  * Feature standardization ensured equal penalization across varying scales (text length vs. binary flags).
  * Extracted feature coefficient weights to interpret log-odds impact: `spam_word_count`, `has_currency`, and `num_links` exhibited highest positive weights toward spam classification.
* **Artifacts:**
  * [email_spam_classification.ipynb](./03_logistic_regression_spam/email_spam_classification.ipynb)
  * [data.csv](./03_logistic_regression_spam/data.csv)
  * [confusion_matrix.png](./03_logistic_regression_spam/confusion_matrix.png)

---

### [04. Random Forest Ensemble — Loan Default Risk Assessment](./04_random_forest_classification)
* **Task:** Binary risk classification to evaluate loan default probability from applicant financial indicators.
* **Features:** Credit score, annual income, loan amount, debt-to-income ratio, employment duration, and past delinquency history.
* **Technique:** `sklearn.ensemble.RandomForestClassifier` (100 trees, Out-of-Bag scoring) benchmarked against a baseline `DecisionTreeClassifier`.
* **Key Findings:**
  * Ensemble learning reduced variance: Random Forest boosted ROC-AUC from **0.6442** (single tree) to **0.7255**, and Precision from **60.0%** to **83.33%**.
  * Out-of-Bag (OOB) accuracy stabilized around ~80.2% as estimator count exceeded 50 trees.
  * Gini feature importance identified `credit_score` (30.1%) and `debt_to_income` (23.5%) as the dominant credit risk determinants.
* **Artifacts:**
  * [random_forest_loan_risk.ipynb](./04_random_forest_classification/random_forest_loan_risk.ipynb)
  * [data.csv](./04_random_forest_classification/data.csv)
  * [confusion_and_roc.png](./04_random_forest_classification/confusion_and_roc.png)

---

## 📈 Model Comparison Summary

| Module | Algorithm | Problem Type | Primary Metric | Baseline / Comparison |
|---|---|---|:---:|:---:|
| **01. Salary Prediction** | Linear Regression (OLS) | Regression | $R^2 = 0.8926$ | Mean baseline ($R^2 = 0.0$) |
| **02. Customer Purchases** | Gaussian Naive Bayes | Classification | Accuracy: $78.0\%$ | Class prior split testing |
| **03. Email Spam Detection** | L2 Logistic Regression | Classification | Accuracy: $89.0\%$, ROC-AUC: $0.942$ | Unscaled baseline |
| **04. Loan Default Risk** | Random Forest (100 Trees) | Ensemble Classification | ROC-AUC: $0.7255$, Precision: $83.3\%$ | Decision Tree (AUC: $0.6442$) |

---

## 📁 Repository Structure

```text
.
├── 01_linear_regression/
│   ├── data.csv
│   └── salary_linear_regression.ipynb
├── 02_naive_bayes_classification/
│   ├── data.csv
│   └── customer_purchase_prediction.ipynb
├── 03_logistic_regression_spam/
│   ├── data.csv
│   ├── confusion_matrix.png
│   └── email_spam_classification.ipynb
├── 04_random_forest_classification/
│   ├── confusion_and_roc.png
│   ├── data.csv
│   └── random_forest_loan_risk.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Darhansila/applied-ml-algorithms.git
cd applied-ml-algorithms
```

### 2. Set up virtual environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Lab
```bash
jupyter lab
```
Navigate to any of the lab directories and open the respective `.ipynb` file.

---

## 👤 Author

**Darkhan** ([@Darhansila](https://github.com/Darhansila))  
Computer Science & Data Science Student
