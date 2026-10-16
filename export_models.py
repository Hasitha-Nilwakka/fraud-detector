# Train and save models and columns for the web app

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

#load enriched dataset
print("Loading enriched dataset...")
df = pd.read_csv('data/train_tier2_sample.csv')

#save customer profile information for quick lookup
print("Exporting customer profiles...")
profile_cols = [
    "client_id", "yearly_income", "total_debt", "debt_to_income_ratio", "credit_score", 
    "current_age", "retirement_age", "is_retired", "num_credit_cards", "user_home_state", "user_avg_amount"
]
customer_profiles = df[profile_cols].copy()
customer_profiles = customer_profiles.drop_duplicates(subset=['client_id'])
customer_profiles.to_csv('data/customer_profiles.csv', index=False)
print(f"Saved customer profiles ({len(customer_profiles)} unique customers)")

#fill missing values and clean inf
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

#tier 1 model training and export
print("\nTraining Tier 1 model...")
feature_list_tier1 = [
    "amount", "credit_limit", "amount_to_limit_ratio", "yearly_income", "total_debt", 
    "debt_to_income_ratio", "credit_score", "current_age", "hour", "day", "is_night", 
    "use_chip", "card_brand", "card_type", "is_diff_state", "is_international", "user_home_state"
]

y = df["target"]
X_tier1 = df[feature_list_tier1].copy()

#one-hot encode and save column names for runtime alignment
X_tier1 = pd.get_dummies(X_tier1, drop_first=True)
tier1_columns = X_tier1.columns.tolist()

#split data
X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_tier1, y, test_size=0.2, stratify=y, random_state=42
)

#train tier 1 model
t1_model = HistGradientBoostingClassifier(
    class_weight='balanced',
    learning_rate=0.1,
    max_iter=150,
    max_leaf_nodes=15,
    random_state=42
).fit(X_train_t1, y_train_t1)

#save tier 1 model and column list
joblib.dump(t1_model, 'models/tier1_model.joblib')
joblib.dump(tier1_columns, 'models/tier1_columns.joblib')
print(f"Saved tier 1 model and {len(tier1_columns)} columns")

#tier 2 model training and export
print("\nTraining Tier 2 model...")
feature_cols_tier2 = [
    "user_avg_amount", "relative_spend_ration",
    "account_age_years", "years_since_pin_change", "is_retired", "has_error", 
    "is_bad_credential", "is_high_risk_mcc", "distress_luxury_risk", "unusual_for_age", 
    "num_cards_issued"
] + feature_list_tier1

X_tier2 = df[feature_cols_tier2].copy()

#one-hot encode and save column names
X_tier2 = pd.get_dummies(X_tier2, drop_first=True)
tier2_columns = X_tier2.columns.tolist()

#split data
X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    X_tier2, y, test_size=0.2, stratify=y, random_state=42
)

#train tier 2 model
hgb_model = HistGradientBoostingClassifier(
    class_weight='balanced', 
    learning_rate=0.1, 
    max_iter=150, 
    max_leaf_nodes=15, 
    random_state=42
).fit(X_train_t2, y_train_t2)

#save tier 2 model and column list
joblib.dump(hgb_model, 'models/tier2_model.joblib')
joblib.dump(tier2_columns, 'models/tier2_columns.joblib')
print(f"Saved tier 2 model and {len(tier2_columns)} columns")
print("\nDone exporting models!")
