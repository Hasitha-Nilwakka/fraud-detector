# Train baseline logistic regression model for fast fraud filtering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

#split data into train and test sets
#use stratify because fraud is only 0.29% of the data so both sets have the same ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

#scale numerical features so large numbers like income don't dominate small numbers like age
numeric_features = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income",
    "total_debt", "debt_to_income_ratio", "credit_score", "current_age",
    "hour", "day"
]

scaler = StandardScaler()
#fit on train and scale both train and test
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

#train logistic regression model
model = LogisticRegression(random_state=42).fit(X_train, y_train)

#predict on test data and get probabilities
y_pred = model.predict(X_test)
predict_probability = model.predict_proba(X_test)

#evaluate the model
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
aps = average_precision_score(y_test, predict_probability[:, 1])

print("--- Sample Predicted Probabilities [P(Legit), P(Fraud)] ---")
print(predict_probability[:5])

print("\n--- Confusion Matrix ---")
print(cm)

print("\n--- Classification Report ---")
print(cr)

print(f"--- PR-AUC (Average Precision Score): {aps:.4f} ---")