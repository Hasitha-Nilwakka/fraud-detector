# Train random forest model for tier 1 fast fraud screening

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score

#load the merged dataset
df = pd.read_csv('data/train_sample.csv')

#target(y) is fraud or not
y = df["target"]
#features(X) remove target from data
X = df.drop(columns=["target"])

#handle 0 division when amount or credit limit is 0
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].fillna(0)
X["amount_to_limit_ratio"] = X["amount_to_limit_ratio"].replace([np.inf, -np.inf], 0)

#select basic transaction and customer features for fast screening
feature_list = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income",
    "total_debt", "debt_to_income_ratio", "credit_score", "current_age",
    "hour", "day", "is_night", "use_chip", "card_brand", "card_type"
]
X = X[feature_list]

#convert categories into dummy variables
#drop_first removes redundant column to save memory
X = pd.get_dummies(X, drop_first=True)

#split data into train and test sets with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

#train random forest with balanced weights so it pays more attention to rare fraud cases
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
).fit(X_train, y_train)

#predict on test data
y_pred = model.predict(X_test)
pp = model.predict_proba(X_test)

#evaluate the results
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
aps_binary = average_precision_score(y_test, y_pred)
aps_probs = average_precision_score(y_test, pp[:, 1])

#Threshhold simulation
fraud_index = list(model.classes_).index(1)
y_prob = model.predict_proba(X_test)[:, fraud_index]
threshholds = [0.40,0.30,0.20,0.15]
results = []

for t in threshholds : 
    pred = (y_prob >= t).astype(int)
    results.append({
        "Confusion Matrix" : confusion_matrix(y_test, pred),
        "Classification Report" : classification_report(y_test, pred),
        "APS" : average_precision_score(y_test, pred)
    })

print("-" * 10)
print(results)
print("-" * 10)

print("--- Sample Predicted Probabilities [P(Legit), P(Fraud)] ---")
print(pp[:5])

print("\n--- Confusion Matrix (Default Threshold tau = 0.50) ---")
print(cm)

print("\n--- Classification Report ---")
print(cr)

print(f"--- PR-AUC (Binary Predictions): {aps_binary:.4f} ---")
print(f"--- PR-AUC (Continuous Probabilities): {aps_probs:.4f} ---")