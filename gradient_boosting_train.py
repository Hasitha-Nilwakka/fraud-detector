# Train gradient boosting model using HistGradientBoostingClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

#load the enriched behavioral dataset
print("Loading enriched Tier 2 behavioral dataset...")
df = pd.read_csv('data/train_tier2_sample.csv')

#target(y)
y = df["target"]

#features to use for training
feature_cols = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income",
    "total_debt", "debt_to_income_ratio", "credit_score", "current_age",
    "hour", "day", "is_night", "use_chip", "card_brand", "card_type",
    "num_credit_cards", "num_cards_issued", "account_age_years", "years_since_pin_change",
    "user_home_state", "is_diff_state", "user_avg_amount", "relative_spend_ration",
    "is_retired", "has_error", "is_bad_credential", "is_high_risk_mcc",
    "distress_luxury_risk", "unusual_for_age"
]

X = df[feature_cols].copy()

#clean 0 division in amount_to_limit_ratio
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].fillna(0)
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].replace([np.inf, -np.inf], 0)

#convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)

#split into train and test sets with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train baseline model with balanced weights
print("\nTraining Default Baseline HistGradientBoosting...")
hgb_model = HistGradientBoostingClassifier(
    class_weight='balanced',
    random_state=42
)
hgb_model.fit(X_train, y_train)

#grid search to find best parameters
param_grid = {
    "max_iter": [100, 150],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31]
}

print("Running Grid Search CV (cv=3, scoring='average_precision')...")
gcs_model = GridSearchCV(
    estimator=HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=-1,
    refit=True
)
gcs_model.fit(X_train, y_train)
best_model = gcs_model.best_estimator_

#predictions for default model
hgb_pred = hgb_model.predict(X_test)
hgb_proba = hgb_model.predict_proba(X_test)[:, 1]

#predictions for tuned grid search model
grid_pred = best_model.predict(X_test)
grid_proba = best_model.predict_proba(X_test)[:, 1]

#print results
print("\n" + "=" * 50)
print("Grid Search Results")
print("=" * 50)
print(f"Best Parameters: {gcs_model.best_params_}")
print(f"Best Cross-Validation PR-AUC: {gcs_model.best_score_:.4f}")

print("\n" + "=" * 50)
print("Default Model Results")
print("=" * 50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, hgb_pred))
print("\nClassification Report:")
print(classification_report(y_test, hgb_pred))
print(f"PR-AUC (Binary): {average_precision_score(y_test, hgb_pred):.4f}")
print(f"PR-AUC (Probabilities): {average_precision_score(y_test, hgb_proba):.4f}")

print("\n" + "=" * 50)
print("Tuned Model Results")
print("=" * 50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, grid_pred))
print("\nClassification Report:")
print(classification_report(y_test, grid_pred))
print(f"PR-AUC (Binary): {average_precision_score(y_test, grid_pred):.4f}")
print(f"PR-AUC (Probabilities): {average_precision_score(y_test, grid_proba):.4f}")
