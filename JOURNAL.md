# 💳 FinTech Fraud Guard & Risk Engine Journal

An end-to-end learning log and architectural record for building, benchmarking, and deploying a production-ready FinTech fraud detection system with Scikit-Learn and Streamlit.

---

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Milestone 0: Project Setup & Environment Isolation](#milestone-0-project-setup--environment-isolation)
3. [Milestone 1: Exploratory Data Analysis (EDA) & Relational Problem Formulation](#milestone-1-exploratory-data-analysis-eda--relational-problem-formulation)
4. [Milestone 2: Baseline Model (Logistic Regression) & The Classification Evaluation Toolkit](#milestone-2-baseline-model-logistic-regression--the-classification-evaluation-toolkit)
5. [Milestone 3: Non-Linear Ensembles & Imbalance Countermeasures](#milestone-3-non-linear-ensembles--imbalance-countermeasures)
6. [Milestone 4: Pipeline Encapsulation & Model Serialization](#milestone-4-pipeline-encapsulation--model-serialization)
7. [Milestone 5: Interactive Web Application (Streamlit FinTech Risk Engine)](#milestone-5-interactive-web-application-streamlit-fintech-risk-engine)
8. [Milestone 6: Model Evolution, Decision Threshold Optimization & Triage Simulation](#milestone-6-model-evolution-decision-threshold-optimization--triage-simulation)
9. [Summary of Performance Benchmarks](#summary-of-performance-benchmarks)

---

## Executive Summary
* **Problem**: Predict whether a real-time banking transaction is fraudulent or legitimate, balancing fraud loss against customer checkout friction.
* **Dataset**: Real-world relational banking data created by **CaixaBank Tech** (spanning transactions, card profiles, customer credit records, and Merchant Category Codes).
* **Target ($y$)**: Binary classification label (`0 = Legitimate`, `1 = Fraudulent`).
* **Initial Baseline (Logistic Regression)**: *[In Progress]*
* **Final Champion Model**: *[Pending]*
* **Key Architectural Goals**:
  * Relational join of customer demographics, card risk limits, and transaction telemetry.
  * Overcoming the **Accuracy Paradox** under severe class imbalance using **Precision**, **Recall**, and **PR-AUC**.
  * Encapsulating preprocessing in an atomic Scikit-Learn `Pipeline`.
  * Deploying an interactive **FinTech Risk Triage Dashboard** with adjustable risk thresholds and What-If fraud diagnostics.

---

## Milestone 0: Project Setup & Environment Isolation

### 1. Python Virtual Environment (`venv`)
* **Purpose**: Creates an isolated sandbox containing its own Python interpreter and dependencies.
* **Why it matters**:
  * Prevents version conflicts between projects (e.g., between the Medical Cost Predictor and Fraud Guard).
  * Guarantees reproducible builds across environments.

```bash
# Initialize dedicated virtual environment
python -m venv frauddetector

# Activate virtual environment
source frauddetector/bin/activate
```

### 2. Dependency Management (`requirements.txt`)
* Pinned dependency manifest for the project:

```text
pandas
numpy
scikit-learn
streamlit
joblib
```

```bash
pip install -r requirements.txt
```

### 3. Git Hygiene & Large File Handling
* Banking datasets contain high-volume transaction files that easily exceed GitHub's **100 MB hard limit** (`transactions_data.csv` is 1.25 GB, `train_fraud_labels.json` is 152 MB).
* **The Rule**: `.gitignore` only applies to untracked files. Staged or committed large files must be purged from Git's index before pushing.
* Configuration in `.gitignore`:

```text
frauddetector/
data/
*.csv
*.json
.DS_Store
__pycache__/
*.pkl
*.joblib
```

### 4. Dataset Acquisition (CaixaBank Tech)
* Sourced from the **CaixaBank Tech 2024 AI Hackathon**:
  * `cards_data.csv`: 6,146 card profiles (brand, type, credit limit, chip status, dark web alerts).
  * `users_data.csv`: 2,000 customers (age, yearly income, total debt, credit score, card counts).
  * `transactions_data.csv`: Real-world transaction logs (amount, chip usage, merchant ID, city/state, MCC code).
  * `mcc_codes.json`: Standard industry Merchant Category Codes (e.g., 4829 Wire Transfers, 5411 Groceries, 7995 Betting).
  * `train_fraud_labels.json`: Ground truth fraud indicators.

---

## Milestone 1: Exploratory Data Analysis (EDA) & Relational Problem Formulation

EDA on multi-table banking datasets requires inspecting table schemas, entity keys, data formats, and class distributions before performing merges.

### 1. Relational Entity Architecture
Our dataset forms a star/snowflake schema centered on the transaction event stream:

| Entity Table | Primary Key | Foreign Keys | Observations | Dimensions | Key Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Users** | `id` | None | 2,000 | 14 cols | `current_age`, `yearly_income`, `total_debt`, `credit_score` |
| **Cards** | `id` | `client_id` $\to$ `Users.id` | 6,146 | 13 cols | `card_brand`, `card_type`, `credit_limit`, `has_chip` |
| **MCC Codes** | `mcc_code` | None | 109 | 2 cols | Merchant categories (`Grocery`, `Wire Transfer`, etc.) |
| **Transactions** | `id` | `client_id`, `card_id`, `mcc` | 8,914,963 | 12 cols | `amount`, `date`, `use_chip`, `merchant_state`, `errors` |
| **Fraud Labels** | `id` | `Transactions.id` | 8,914,963 | 1 col | Ground-truth binary target (`Yes` / `No`) |

### 2. The Target Variable & The Severe Class Imbalance
* **Target ($y$)**: `target` (`Yes` = 1, `No` = 0)
* **Total Transactions**: 8,914,963
* **Legitimate Records (`No`)**: 8,901,631 (**99.8505%**)
* **Fraudulent Records (`Yes`)**: 13,332 (**0.1495%**)
* **Imbalance Ratio**: **1 fraud per 667 legitimate transactions**

> [!WARNING] The Accuracy Trap
> A naive model predicting `"No"` for every transaction achieves **99.85% accuracy**, but catches **0% of fraud** (\$0 recovered, infinite fraud losses).
> Evaluation must prioritize **Precision**, **Recall**, and **PR-AUC (Precision-Recall Area Under Curve)** over raw accuracy.

### 3. Critical Data Quality Findings (Data Hygiene)
1. **Currency String Formatting**:
   * Financial columns contain literal dollar signs: `yearly_income` (`$59696`), `credit_limit` (`$24295`), and `amount` (`$14.57`, `$-77.00`).
   * *Transformation applied*: Cleaned via `.astype(str).str.replace('$', '', regex=False).astype(float)`.
2. **Refunds & Chargebacks**:
   * Negative transaction amounts (e.g., `$-77.00`) represent refunds or chargeback adjustments.
3. **Zero-Variance Columns**:
   * `card_on_dark_web`: All 6,146 cards have value `"No"`.
   * Columns with zero variance convey zero predictive signal and should be dropped.
4. **Transaction Date Parsing**:
   * `date` contains timestamps (e.g., `2010-01-01 00:01:00`).
   * Extracted `hour` ($0 \dots 23$), `day` ($0 \dots 6$), and `is_night` (indicator for transactions between 11 PM and 6 AM).

### 4. Engineered Domain Ratios (FinTech Risk Signals)
Rather than relying solely on raw dollar values, financial fraud detection models require relative exposure ratios:

* **Amount to Credit Limit Ratio**:
  $$\text{amount\_to\_limit\_ratio} = \frac{\text{amount}}{\text{credit\_limit}}$$
  *Measures transaction risk relative to borrowing capacity.*

* **Debt to Income Ratio**:
  $$\text{debt\_to\_income\_ratio} = \frac{\text{total\_debt}}{\text{yearly\_income}}$$
  *Measures baseline customer leverage and distress.*

### 5. Final Curated Modeling Dataset (`data/train_sample.csv`)
* **Total Transactions**: 134,395
* **Total Dimensions**: 47 unified relational columns
* **Target Breakdown**:
  * Legitimate (`target = 0`): 134,004 (99.71%)
  * Fraudulent (`target = 1`): 391 (0.29%)
* **Artifact**: Persisted to `data/train_sample.csv` to allow sub-second model training and experiments without re-reading the 1.25 GB raw CSV and 150 MB JSON files.

---

## 🏛️ Architectural Strategy: Two-Tier Cascaded Risk Architecture

Real-world payment processors (e.g., Stripe, Visa, Adyen) face rigid latency budgets (<50ms). Running comprehensive historical database queries for every minor purchase would overwhelm banking infrastructure. We adopt a **Cascaded Risk Engine**:

```
[Incoming Transaction]
         │
         ▼
[ Tier 1: Fast Gatekeeper ] ─── Low Risk (P < 0.10) ─────────► [ Auto-Approve ] (>95% of traffic)
         │
    Suspicious or High Risk (P ≥ 0.10)
         │
         ▼
[ Tier 2: Deep Behavioral Engine ] 
(Historical customer baselines: location travel feasibility, MCC novelty, personal spending Z-score)
         │
         ├─── Anomaly Confirmed ─────────────────────────────► [ Decline & Freeze Card ]
         ├─── Moderate Ambiguity ────────────────────────────► [ Step-Up 3D-Secure SMS / OTP ]
         └─── Contextual False Alarm ────────────────────────► [ Approve ]
```

---

## Milestone 2: Baseline Model (Tier 1 Fast Gatekeeper - Logistic Regression)

The goal of Milestone 2 is to construct and benchmark the **Tier 1 Fast Gatekeeper** using the standalone transaction payload and account attributes.

### 1. Mathematical Formulation: Logistic Regression & The Sigmoid Function
Unlike Linear Regression, which estimates continuous values ($-\infty$ to $+\infty$), Logistic Regression estimates conditional probability $P(y=1|X) \in [0, 1]$ via the logistic sigmoid link function:

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

* Prediction rule at default decision threshold $\tau = 0.5$:
  $$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge 0.5 \\ 0 & \text{if } \hat{p} < 0.5 \end{cases}$$

### 2. Imbalanced Stratified Splitting (`stratify=y`)
* Because positive cases account for only **0.29%** of observations, a purely random split risks non-representative positive sampling.
* `stratify=y` enforces exact class proportions across both training (80%) and testing (20%) splits:
  * Training set: ~107,516 records (~313 frauds)
  * Test set: ~26,879 records (~78 frauds)

### 3. Tier 1 Feature Scope
Tier 1 operates strictly on instant transaction telemetry without historical lookups:
* **Numerical Features**: `amount`, `credit_limit`, `amount_to_limit_ratio`, `yearly_income`, `total_debt`, `debt_to_income_ratio`, `credit_score`, `current_age`, `hour`, `day`, `is_night`.
* **Categorical Features**: `card_brand`, `card_type`, `use_chip`.
* **Pruned Columns**: Raw database IDs (`id`, `client_id`, `card_id`), timestamps (`date`), raw text addresses, and card numbers.

### 4. Classification Evaluation Toolkit (No More MAE / $R^2$)
* **Confusion Matrix Quadrants**:
  * **True Positives (TP)**: Caught fraud.
  * **False Positives (FP)**: Legitimate purchase blocked (customer friction).
  * **False Negatives (FN)**: Fraud escaped undetected (direct dollar loss).
  * **True Negatives (TN)**: Legitimate purchase approved.

* **Precision**: When an alert fires, how often is it real fraud?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

* **Recall**: What fraction of total frauds were successfully caught?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

* **PR-AUC (Average Precision)**: Area under the Precision-Recall curve across all classification thresholds. The primary benchmark metric for severe class imbalance.

### 5. Baseline Evaluation Results (Logistic Regression)
* **Dataset Split**: 80% Train (107,516 rows, 313 frauds) | 20% Test (26,879 rows, 78 frauds)
* **Input Features ($X$)**: 17 features (11 continuous financial/temporal signals + 6 one-hot encoded payment channels)
* **Scaling**: Standardized via `StandardScaler`

#### Confusion Matrix:
$$\begin{pmatrix} \text{TN} = 26,801 & \text{FP} = 0 \\ \text{FN} = 75 & \text{TP} = 3 \end{pmatrix}$$

#### Classification Performance:
* **Accuracy**: **99.72%** *(The textbook Accuracy Trap — model misses 96.15% of all fraud)*
* **Precision**: **1.00 (100.0%)** *(Zero false alarms on positive alerts)*
* **Recall**: **0.0385 (3.85%)** *(Only 3 out of 78 actual fraud attacks detected)*
* **F1-Score**: **0.0714 (7.14%)**
* **PR-AUC (Average Precision)**: **0.0413 (4.13%)** *(14x better than random 0.29% baseline, but severely suppressed by class imbalance)*

---

## 📊 Summary of Performance Benchmarks

| Milestone | Model / Strategy | Accuracy | Precision | Recall | F1-Score | PR-AUC (Binary) | PR-AUC (Probs) | Primary Limitation / Achievement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Milestone 2** | **Baseline (Logistic Regression)** | 99.72% | 0.8000 | 0.0513 | 0.0964 | 0.0413 | 0.1299 | High accuracy trap; missed 95% of fraud due to unweighted linear loss. |
| **Milestone 3A** | **Tier 1 (Random Forest - Balanced, $\tau=0.50$)** | 99.79% | 0.8158 | 0.3974 | 0.5345 | 0.3260 | 0.5776 | Nearly 8x fraud recovery (31 caught vs 4), F1 surged +43 points with only 7 false alarms out of 26.8k. |
| **Milestone 3B** | **Tier 1 (Random Forest - Screen, $\tau=0.15$)** | 99.51% | 0.3448 | **0.7692** | 0.4762 | 0.2659 | 0.5776 | Intercepts 60 out of 78 frauds (15x baseline); auto-approves 99.35% of traffic, routing only 0.65% to Tier 2. |
| **Milestone 4** | **Champion Tier 2 (Deep Behavioral RF, $\tau=0.50$)** | **99.82%** | **0.8947** | **0.4359** | **0.5862** | **0.3917** | **0.6581** | **High-Precision Tier 2**: Precision peaked at 89.5% with only 4 false alarms out of 26.8k; PR-AUC reached 65.8%. |
| **Milestone 5A**| **Tier 2 (Default HistGradientBoosting)** | **99.72%** | **0.5203** | **0.8205** | **0.6368** | **0.4275** | **0.7839** | **Peak F1 Balance**: Caught 64/78 frauds (82.1% Recall) with only 59 false alarms out of 26.8k; F1 reached 0.64. |
| **Milestone 5B**| **Tier 2 Champion (Tuned HistGradientBoosting)** | **99.43%** | **0.3223** | **0.8718** | **0.4706** | **0.2813** | **0.7444 (CV)** | **Maximum Fraud Recovery**: Intercepts **68 out of 78 frauds (87.2% Recall)** with only 10 missed frauds; 74.4% CV PR-AUC. |



---

## Milestone 3: Non-Linear Ensembles & Imbalance Countermeasures

### 1. Cost-Sensitive Learning (`class_weight='balanced'`)
* Unweighted algorithms optimize for overall sample accuracy, encouraging the model to predict the majority class.
* By specifying `class_weight='balanced'`, Scikit-Learn weights each class inversely proportional to its training frequency:
  $$w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples}, j}}$$
* For our training set (107,203 legit vs. 313 fraud), this assigns a penalty weight of $\approx 171.7$ to fraud vs. $0.50$ to legit.
* Missing a fraud is penalized **343 times heavier**, forcing decision tree split criteria (Gini / Entropy) to isolate minority positive cases.

### 2. Random Forest Architecture
* **Non-Linear Thresholding**: Rather than fitting a hyperplane, Random Forest constructs an ensemble of 100 de-correlated decision trees (`n_estimators=100`) via bagging (bootstrap aggregation) and random feature subspace sampling.
* **Scale Invariance**: Tree splitting is purely rank-based (ordinal), making it fully invariant to feature scaling magnitudes and eliminating the need for `StandardScaler`.

### 3. Evaluation Results (Random Forest Classifier)
* **Confusion Matrix**:
  $$\begin{pmatrix} \text{TN} = 26,794 & \text{FP} = 7 \\ \text{FN} = 47 & \text{TP} = 31 \end{pmatrix}$$
* **Recall**: Jumped from **5.13% $\to$ 39.74%** (True Positives surged from 4 to 31).
* **Precision**: Maintained an exceptional **81.58%** (only 7 false alarms across 26,801 legitimate transactions).
* **F1-Score**: Surged from **0.0964 $\to$ 0.5345** (+43.8 percentage points).
* **PR-AUC**: Jumped from **0.1299 $\to$ 0.5776** on probability distribution.

### 4. Decision Threshold Optimization ($\tau$ Moving)
In asymmetric financial risk, the default $\tau = 0.50$ cutoff is sub-optimal. For a **Tier 1 Fast Gatekeeper**, an alarm does not immediately decline the card—it escalates the transaction to Tier 2 Deep Analysis.

#### Threshold Sweep Results:

| Threshold ($\tau$) | Frauds Caught (TP) | Frauds Missed (FN) | False Alarms (FP) | Recall | Precision | F1-Score | Operational Role in Two-Tier Engine |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50 (Default)** | 31 / 78 | 47 | 7 | 39.7% | **81.6%** | 0.535 | Conservative direct auto-decline cutoff. |
| **0.40** | 36 / 78 | 42 | 12 | 46.2% | 75.0% | 0.570 | Moderate risk tier. |
| **0.30** | 45 / 78 | 33 | 27 | 57.7% | 62.5% | **0.600** | **Optimal standalone F1-score balance**. |
| **0.20** | 53 / 78 | 25 | 70 | 67.9% | 43.1% | 0.527 | Strong screening candidate. |
| **0.15 (Champion)**| **60 / 78** | **18** | **114** | **76.9%** | 34.5% | 0.476 | **Optimal Tier 1 Screening Cutoff**: Intercepts 77% of fraud, auto-approves 99.35% of volume, and routes only 0.65% to Tier 2. |

---

## Milestone 4: Tier 2 Deep Behavioral Engine (Customer Profiling)

Tier 1 acts as a low-latency screening filter. For transactions that require deeper inspection, the **Tier 2 Deep Behavioral Engine** incorporates historical customer baselines to detect contextual anomalies that standalone telemetry cannot reveal.

### 1. The Geographic Out-of-State Breakthrough
By analyzing historical transaction patterns per customer (`client_id`), we discovered an overwhelming spatial pattern:

| Spatial Location Context | Total Transactions | Total Frauds | Fraud Incident Rate |
| :--- | :---: | :---: | :---: |
| **Transaction in Customer's Primary State (`is_diff_state = 0`)** | **108,792** | **0** | **0.0000%** |
| **Transaction Outside Primary State (`is_diff_state = 1`)** | **25,603** | **391** | **1.5271%** |

* **Empirical Law in Dataset**: **100.0% of all fraud events occur outside the customer's primary state.**
* Legitimate local spending has a 0% baseline risk, enabling the engine to rule out false positives with extreme confidence.

### 2. Engineered Tier 2 Behavioral Signals
In `tier2_feature_engineering.py`, we enriched the dataset with three key customer baseline features:

1. **`user_home_state`**:
   * Evaluated using modal frequency aggregation: $\text{mode}(\text{merchant\_state})$ grouped by `client_id`.
2. **`is_diff_state`**:
   * Binary spatial anomaly flag ($1$ if `merchant_state != user_home_state`, else $0$).
3. **`user_avg_amount`**:
   * Customer personal spending baseline: $\mu_{\text{user}} = \text{mean}(\text{amount})$ grouped by `client_id`.
4. **`relative_spend_ration`**:
   * Measures purchase magnitude relative to the customer's typical expenditure:
     $$\text{relative\_spend\_ration} = \frac{\text{amount}}{\mu_{\text{user}}}$$

### 3. Tier 2 Curated Dataset Artifact
* **Path**: `data/train_tier2_sample.csv`
* **Dimensions**: 134,395 rows $\times$ 51 columns (fully cleaned, 0 missing values, 0 infinities in engineered ratios).

---

## 🧠 Milestone 4B: Comprehensive Feature Audit & Domain Brainstorming

To transition our models from mechanical pattern-matching to a production-grade risk engine, we conducted an exhaustive, column-by-column empirical and behavioral audit:

### 1. Final Column Disposition & Engineering Matrix

| Column / Signal | Category | Decision | Empirical Justification & Operational Rationale |
| :--- | :--- | :---: | :--- |
| **`yearly_income`** | Customer Profile | **KEEP** | Bottom income quartile has nearly $2\times$ the fraud rate of top income ($0.35\%$ vs $0.19\%$). |
| **`total_debt`** | Customer Profile | **KEEP** | Reflects baseline financial pressure; essential for debt-to-income leverage ratios. |
| **`per_capita_income`**| Customer Profile | **HOLD** | Kept on reserve to capture neighborhood relative-wealth ratios. |
| **`gender`** | Customer Profile | **DROP** | Statistically insignificant difference ($0.30\%$ female vs $0.28\%$ male) + heavy Fair Lending / ECOA compliance liability. |
| **`current_age`** | Customer Profile | **KEEP** | Captures distinct U-shaped vulnerability curve (highest risk for $<30$ and $>75$). |
| **`birth_year`** | Customer Profile | **DROP** | Correlation with `current_age` is $-0.9997$ (pure mathematical redundancy). |
| **`birth_month`** | Customer Profile | **DROP** | Random calendar artifact with zero causal connection to fraud attacks. |
| **`retirement_age`** | Customer Profile | **ENGINEER** | Transformed to **`is_retired`** ($+24\%$ higher fraud incident rate for retirees). |
| **`credit_score`** | Financial Identity| **KEEP** | Captures non-linear exposure: subprime targets for scams; prime targets for high-limit breaches. |
| **`num_credit_cards`**| Financial Identity| **KEEP** | Captures single-card overexposure ($0.44\%$ fraud) and dormant-card neglect ($0.48\%$ fraud). |
| **`acct_open_date`** | Card Metadata | **ENGINEER** | Transformed to **`account_age_years`** (vintage accounts $>10$ yrs old suffer $2.3\times$ higher fraud). |
| **`year_pin_last_changed`**| Card Metadata | **ENGINEER** | Transformed to **`years_since_pin_change`** (stale PINs 3-7 yrs old experience $0.72\%$ fraud vs $0.28\%$). |
| **`num_cards_issued`**| Card Metadata | **KEEP** | Measures card replacement frequency and potential account distress. |
| **`card_on_dark_web`**| Card Metadata | **DROP** | $100\%$ `"No"` across all records (zero variance). |
| **`cvv` & `card_number`**| Card Metadata | **DROP** | PCI-DSS non-compliance and arbitrary non-generalizable credentials. |
| **`expires`** | Card Metadata | **DROP** | Unparsed string date with minimal predictive signal. |
| **`has_chip`** | Card Metadata | **DROP** | Redundant with the vastly superior transaction-level `use_chip`. |
| **`use_chip`** | Transaction | **KEEP** | **92% of all frauds are Online Transactions** ($2.43\%$ online vs $0.026\%$ physical swipe). |
| **`errors`** | Transaction | **ENGINEER** | Transformed to **`has_error`** ($3.4\times$ risk) and **`is_bad_credential`** (Bad CVV has a $17\times$ fraud spike). |
| **`mcc_description`** | Transaction | **ENGINEER** | Transformed to **`high_risk_mcc`** targeting Computers ($34\%$), Electronics ($18\%$), and Transit ($5-18\%$). |
| **`merchant_city` / `zip`**| Transaction | **DROP** | High-cardinality sparse noise; spatial anomaly is cleanly captured by `is_diff_state`. |

### 2. High-Order Composite Behavioral Risk Features
From our brainstorm, we identified four domain interaction features to enrich our final feature store:
1. **`unusual_for_age`**: Flags age-incongruent spending (e.g., seniors at 3:00 AM; young adults purchasing heavy marine/industrial equipment).
2. **`distress_luxury_risk`**: Flags cardholders with high debt-to-income ratios transacting at high-liquidity luxury/resale categories (Precious Metals, Electronics).
3. **`is_retired`**: Indicator when `current_age >= retirement_age`.
4. **`is_bad_credential`**: Binary flag indicating previous CVV or Card Number verification errors.

---

## Milestone 5: Advanced Gradient Boosted Decision Trees & Hyperparameter Optimization

While Random Forest constructs independent parallel trees via bagging (bootstrap aggregation), Gradient Boosting builds trees sequentially, where each successive estimator directly fits the negative gradient (residual errors) of the ensemble's prior predictions:

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

### 1. Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`)
For large FinTech tabular datasets (>100k rows), traditional `GradientBoostingClassifier` evaluates split points across all continuous feature values ($O(n_{\text{samples}} \times n_{\text{features}})$), which leads to slow training times.
`HistGradientBoostingClassifier` (inspired by LightGBM):
1. **Integer Binning**: Pre-bins continuous features into 256 discrete integer bins.
2. **Constant-Time Split Finding**: Computes histogram bins in $O(n_{\text{bins}})$, speeding up tree construction by 10x to 50x while preserving non-linear decision boundaries.
3. **Class Weighting (`class_weight='balanced'`)**: Explicitly penalizes residual errors on fraudulent samples, forcing the gradient descent step to prioritize the positive minority class.

### 2. Hyperparameter Grid Search (`GridSearchCV`)
We constructed a hyperparameter tuning grid optimized for Precision-Recall tradeoff:
* `max_iter`: $[100, 150]$ — boosting iterations (number of trees).
* `learning_rate`: $[0.05, 0.1]$ — shrinkage factor $\gamma$ controlling the step size along the gradient.
* `max_leaf_nodes`: $[15, 31]$ — structural complexity per tree. Restricting to 15 nodes prevents overfitting on sparse fraudulent leaf nodes.
* `scoring`: `"average_precision"` (PR-AUC) — optimizes the area under the PR curve across all possible decision thresholds.
* `cv`: 3-fold stratified cross-validation.

### 3. Hyperparameter Tuning & Cross-Validation Results
* **Optimal Selected Parameters**:
  ```python
  {'learning_rate': 0.1, 'max_iter': 150, 'max_leaf_nodes': 15}
  ```
* **Best Cross-Validation PR-AUC (Average Precision)**: **0.7444 (74.44%)**
  * Substantial leap over baseline Logistic Regression (0.1299) and Tier 2 Random Forest (0.6581).

### 4. Test Set Evaluation & Comparison

Held-out Test Set: 26,879 records (26,801 legitimate, 78 fraud).

#### A. Tuned Model (`best_estimator_` with `max_leaf_nodes=15, max_iter=150, lr=0.1`):
* **Confusion Matrix**:
  $$\begin{pmatrix} \text{TN} = 26,658 & \text{FP} = 143 \\ \text{FN} = 10 & \text{TP} = 68 \end{pmatrix}$$
* **Frauds Intercepted (TP)**: **68 out of 78** (**87.18% Recall**)
* **Frauds Missed (FN)**: Only **10** across the entire test set!
* **False Alarms (FP)**: 143 out of 26,801 legitimate transactions (0.53% false alarm rate).
* **Precision**: 32.23% (roughly 1 alert in 3 is confirmed fraud, exceptional for raw screening under 0.29% class incidence).
* **F1-Score**: 0.4706

#### B. Default Model (`HistGradientBoostingClassifier(class_weight='balanced')`):
* **Confusion Matrix**:
  $$\begin{pmatrix} \text{TN} = 26,742 & \text{FP} = 59 \\ \text{FN} = 14 & \text{TP} = 64 \end{pmatrix}$$
* **Frauds Intercepted (TP)**: **64 out of 78** (**82.05% Recall**)
* **Frauds Missed (FN)**: 14
* **False Alarms (FP)**: Only **59** out of 26,801 legitimate transactions (0.22% false alarm rate).
* **Precision**: **52.03%**
* **F1-Score**: **0.6368 (0.64)** — **Project-wide record for balanced precision-recall trade-off**.

### 5. Architectural Insight: PR-AUC Evaluation Nuance
In Scikit-Learn, `average_precision_score(y_true, y_score)` requires continuous probability estimates (`predict_proba()[:, 1]`). 
* When passed discrete binary predictions (`predict()`), Scikit-Learn only computes a single-point rectangular area (yielding the 0.2813 and 0.4275 values).
* During cross-validation, `GridSearchCV` evaluated the true probability distributions across all continuous decision thresholds, confirming the true underlying **74.44% PR-AUC**.

---

## Milestone 6: Production Pipeline Serialization & Model Packaging

To support real-time inference in an interactive FinTech operations console without re-aggregating 134,000 raw transaction records on every swipe, we decoupled offline model training from online scoring via serialized production artifacts.

### 1. Serialized Production Artifacts

```
fraud-detector/
├── models/
│   ├── tier1_model.joblib          # Fast Gatekeeper (Random Forest Classifier, 7.3 MB)
│   ├── tier1_columns.joblib        # Exact 17-column one-hot training schema
│   ├── tier2_model.joblib          # Deep Behavioral Engine (Tuned HistGradientBoosting, 226 KB)
│   └── tier2_columns.joblib        # Exact 78-column one-hot training schema
└── data/
    └── customer_profiles.csv       # Pre-aggregated 1,096 cardholder baseline profiles (78 KB)
```

### 2. Runtime Schema Alignment (`.reindex()` Pattern)
To resolve the classical "Dimension Mismatch" trap where single-row real-time transactions fail to produce dummy columns for unobserved categories (states, card brands), the engine implements the schema reindexing pattern:
$$\mathbf{x}_{\text{aligned}} = \text{pd.get\_dummies}(\mathbf{x}_{\text{input}}).\text{reindex}(\text{columns}=\mathbf{C}_{\text{train}}, \text{fill\_value}=0)$$
This guarantees consistent tensor dimensions ($D_1=17$, $D_2=78$) with zero latency overhead.

---

## Milestone 7A: Authentic Test Simulation Pool Generation

To power the interactive Streamlit operations console without requiring manual entry of 25+ complex banking parameters, we curated a genuine evaluation pool sourced directly from our held-out test split (26,879 records).

### 1. Simulation Pool Composition (`data/simulation_pool.csv`)
* **Total Transactions**: 278 real banking records (119 KB)
* **Target Breakdown**:
  * 🚨 **Verified Fraud Attacks**: 78 (100% of all frauds in the held-out test set; 28.1% of pool)
  * 🛡️ **Verified Legitimate Transactions**: 200 (71.9% of pool)

### 2. FinTech Archetype Cohorts
Every record is tagged with an intuitive scenario archetype for operator triage:
* `🚨 Fraud: Account Takeover (Online Breach)`: 55 cases
* `🚨 Fraud: High-Risk Category (Tech/Jewelry/Transit)`: 11 cases
* `🚨 Fraud: Physical Out-of-State Counterfeit Swipe`: 8 cases
* `🚨 Fraud: Credential Error / Testing Attack`: 4 cases
* `🛡️ Legit: Everyday Local Routine Spend`: 70 cases
* `✈️ Legit: Out-of-State Travel (False Alarm Test)`: 50 cases
* `🛒 Legit: In-State Online E-Commerce`: 44 cases
* `💎 Legit: High-Value Authorized Purchase`: 36 cases

---

## Milestone 7B: Full-Stack Streamlit Risk Operations Console (`app.py`)

We packaged the serialized models and the simulation pool into a real-time risk console inspired by Stripe Radar and Visa Decision Manager.

### 1. Architectural Highlights of `app.py`
1. **Interactive Simulation & Archetype Filtering**:
   - Operators can filter by specific fraud or legitimate cohorts (e.g., Account Takeovers, Travel False Alarm Tests).
   - A single click on `[ 🎲 Simulate Live Transaction ]` draws an authentic transaction from the held-out test split.
2. **Three-Column Event Dossier**:
   - **Cardholder Profile**: Real-time display of baseline spending $\mu$, credit score, age, leverage DTI, and home state.
   - **Transaction Telemetry**: Real-time display of purchase amount, credit limit ratio, merchant location, payment channel, and security verification errors.
   - **Ground Truth Dossier**: Reveals the genuine label (Confirmed Fraud vs. Confirmed Legit) for performance auditing.
3. **Live Cascaded Execution Pipeline**:
   - **Tier 1 Fast Gatekeeper**: Sub-8ms latency evaluation. Auto-approves safe transactions ($P_1 < 0.15$), saving latency and computational overhead.
   - **Tier 2 Deep Behavioral Engine**: Evaluates contextual deviation ($P_2$) against personal averages and spatial travel boundaries.
   - **Action Triage**:
     - 🟢 `APPROVE` ($P_2 < 0.25$)
     - 🟡 `STEP-UP 3D-SECURE CHALLENGE (SMS OTP)` ($0.25 \le P_2 < 0.60$)
     - 🔴 `DECLINE TRANSACTION & FREEZE CARD` ($P_2 \ge 0.60$)
4. **Stripe Radar-Style Reason Codes**:
   - Dynamic explainability bullets highlighting specific risk triggers (e.g., "Severe Spending Surge: 5.4x personal average", "Spatial Travel Anomaly: Non-home state purchase").
5. **Interactive What-If Overrides**:
   - Allows operators to modify transaction parameters (dollar amount, merchant location, payment channel, error injection) on the fly and witness the cascaded risk scores adjust in real time.

---

## 🔬 Critical FinTech Edge Case: Purchase Authorizations vs. Chargeback Reversals

During simulation audits, we encountered an edge case: a transaction labeled `target = 1 (Fraud)` with an amount of **`$-401.00`** and a credit limit usage of **`-1293.5%`**.

### 1. Forensic Discovery
Tracing cardholder `#1694` in the raw data revealed a paired transaction sequence:
* `2010-02-23 14:05:00`: Unauthorized online charge of **`+$334.00`** at Computer Network Services (`target = 1`).
* `2010-02-23 14:06:00`: Chargeback dispute credit of **`-$401.00`** issued by the merchant/bank (`target = 1`).

### 2. Operational Takeaway
In banking data warehouses, settlement ledgers log dispute credits with a negative sign. Because CaixaBank tagged all ledger entries belonging to a fraud dispute case, the refund inherited the positive fraud target label.

However, in payment processing:
* **Real-time Authorization Gateways** (Tier 1 & Tier 2) strictly process **outgoing purchase debits (`amount > 0`)**.
* **Accounting Credits & Refunds (`amount < 0`)** bypass the authorization screener and route through a separate **Merchant Refund Abuse & Anti-Laundering Engine**.
* Enforcing `amount > 0` in `build_simulation_pool.py` cleanly eliminates accounting ledger artifacts, ensuring 100% of simulated transactions represent genuine point-of-sale purchase attempts ($0.19 to $2,529.39).

---

## Milestone 7C: Offshore Counterfeit Defense & Cascaded Risk Synthesis

During auditing, we discovered offshore carding attacks (e.g., cardholder `#1644` in South Carolina suffering an \$8.30 taxi swipe in Funafuti, Tuvalu at 2:11 AM, and cardholder `#1963` in California suffering an \$7.76 swipe in Haiti).

### 1. Empirical Discovery
* Across all 134k records:
  - **Tuvalu transactions**: 100.0% Fraud (5/5)
  - **Haiti transactions**: 100.0% Fraud (26/26)
* Counterfeit skimming rings take cloned magnetic stripes to offshore jurisdictions where fraud filters are assumed to be lax, swiping for low amounts ($7-$10) to blend into legitimate commuter traffic.

### 2. Feature Engineering: `is_international`
We engineered a dedicated spatial cross-border indicator:
$$\text{is\_international} = (\text{merchant\_state} \notin \text{US\_STATES}) \land (\text{merchant\_state} \neq \text{NaN})$$
This cleanly separates domestic interstate commuters (e.g. SC $\to$ NC) from offshore international carding rings (e.g. SC $\to$ Tuvalu/Haiti).

### 3. Cascaded Risk Synthesis (The Risk Floor Rule)
In a two-tier cascaded architecture, Tier 2 exists to contextualize and challenge ambiguous cases, not blindly overrule strong spatial gateway alarms based on tiny dollar amounts.
$$\text{Synthesized Risk} = \max\left(P_{\text{Tier 1}} \cdot \mathbb{I}(\text{Travel / International}), P_{\text{Tier 2}}\right)$$

### 4. Final Triage Performance (Simulation Pool)
* **Total Fraud Attacks**: 74
* **Total Caught**: **73 out of 74 (98.6% Recall)**
  - 69 Direct Auto-Declines & Card Freezes
  - 4 Step-Up 3D-Secure SMS OTP Challenges
  - Only 1 missed fraud attack!
* **Legitimate Transactions**: 200


---

## Milestone 7D: Non-Technical User Experience & Plain-English Redesign

To ensure business stakeholders, risk analysts, and non-technical operators can easily understand, evaluate, and operate the system, the web application (`app.py`) was overhauled to eliminate AI jargon and prioritize clarity:

### 1. Key Enhancements Delivered
1. **Clear Header & Subtitle**: 
   - Primary: `🛡️ Card Transaction Fraud Detector`
   - Subtitle: `Dual-Model AI Defense for Real-Time Payment Protection`
2. **Plain-English System Overview**:
   - An intuitive explanation banner explaining how the **two specialized AI models** work together:
     - **Model 1 (The Fast Gatekeeper):** Evaluates transaction details in under 10 milliseconds to immediately clear safe purchases.
     - **Model 2 (The Behavioral Detective):** Cross-checks customer history and typical spending habits to catch stealthy attacks or resolve false alarms.
3. **Outcome & Decision Promoted to the Top**:
   - Placed the **🎯 AI Decision & Outcome Check** directly beneath the header.
   - Features side-by-side cards:
     - **System Decision:** Visual badges (Approved, SMS OTP Verification, Blocked & Card Frozen) with simple descriptions and evaluated risk levels.
     - **Was the AI Correct? (Accuracy Check):** High-contrast confirmation badges clearly indicating whether the system successfully stopped fraud, cleared a genuine customer, or encountered an edge case.
4. **Renamed "Transaction Telemetry" $\to$ "Transaction Breakdown"**:
   - Replaced engineering telemetry jargon with intuitive, user-friendly labels (Purchase Amount, Where, Payment Method, Store Category, Security Status).
5. **Removal of What-If Overrides**:
   - Removed experimental override controls to present a clean, production-grade review interface focused on actual transaction auditing.
6. **Plain-English Decision Explanations**:
   - Replaced raw feature weight values with natural, human sentences (e.g., *"Unusual Spending Spike: The purchase amount ($950.00) is 6.5 times higher than what this customer typically spends ($145.20)"*).


