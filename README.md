# Applied Machine Learning & Algorithmic Foundations

Practical implementations and comparative studies of foundational machine learning algorithms applied to synthetic and structured real-world datasets.

This repository organizes laboratory works exploring core predictive modeling techniques: regression, probabilistic classification, and linear classification with regularization.

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
