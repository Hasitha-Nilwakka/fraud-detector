# Train tier 2 random forest model with customer behavioral history

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

#load the enriched dataset
df = pd.read_csv('data/train_tier2_sample.csv')

#target(y)
y = df["target"]
#features(X)
X = df.drop(columns=["target"])

#clean 0 division in amount_to_limit_ratio
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].fillna(0)
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].replace([np.inf, -np.inf], 0)

#select basic features plus customer home state and spending history
feature_list = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income",
    "total_debt", "debt_to_income_ratio", "credit_score", "current_age",
    "hour", "day", "is_night", "use_chip", "card_brand", "card_type",
    "user_home_state", "is_diff_state", "user_avg_amount", "relative_spend_ration"
]
X = X[feature_list]

#convert categories into dummy variables
X = pd.get_dummies(X, drop_first=True)

#split data into train and test sets with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train random forest with balanced weights
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
).fit(X_train, y_train)

#predict on test data
y_pred = model.predict(X_test)
pp = model.predict_proba(X_test)

#evaluate the model
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
aps_binary = average_precision_score(y_test, y_pred)
aps_probs = average_precision_score(y_test, pp[:, 1])

print("--- Confusion Matrix (Default Threshold tau = 0.50) ---")
print(cm)

print("\n--- Classification Report ---")
print(cr)

print(f"--- PR-AUC (Binary Predictions): {aps_binary:.4f} ---")
print(f"--- PR-AUC (Continuous Probabilities): {aps_probs:.4f} ---")