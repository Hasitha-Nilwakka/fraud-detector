# Feature engineering for tier 2 behavioral model

import pandas as pd
import numpy as np

#load the merged dataset
print("Loading dataset...")
df = pd.read_csv('data/train_sample.csv')

#find user home state from most frequent merchant state
df['user_home_state'] = df.groupby('client_id')['merchant_state'].transform(
    lambda s: s.mode().iloc[0] if not s.empty else 'Unknown'
)
#check if transaction happened in a different state
df['is_diff_state'] = (df['user_home_state'] != df['merchant_state']).astype(int)

#check if merchant is international outside US states
us_states = set(df['user_home_state'].dropna().unique())
df['is_international'] = (df['merchant_state'].notna() & ~df['merchant_state'].isin(us_states)).astype(int)

#calculate customer average spend and relative spend ratio
df['user_avg_amount'] = df.groupby('client_id')['amount'].transform('mean')
df['relative_spend_ration'] = (df['amount'] / df['user_avg_amount']).replace([np.inf, -np.inf], 0).fillna(1.0)

#check if customer is retired
df['is_retired'] = (df['current_age'] >= df['retirement_age']).astype(int)

#account age in years and years since pin change
tx_year = pd.to_datetime(df['date']).dt.year
open_year = pd.to_datetime(df['acct_open_date'], format='%m/%Y').dt.year
df['account_age_years'] = (tx_year - open_year).clip(lower=0).fillna(0)
df['years_since_pin_change'] = (tx_year - df['year_pin_last_changed']).clip(lower=0).fillna(0)

#check errors like bad cvv or bad card number
df['has_error'] = df['errors'].notna().astype(int)
df['is_bad_credential'] = df['errors'].astype(str).str.contains('Bad CVV|Bad Card Number', case=False).astype(int)

#high risk store categories
high_risk_cats = [
    'Computers, Computer Peripheral Equipment',
    'Electronics Stores',
    'Digital Goods - Games',
    'Furniture, Home Furnishings, and Equipment Stores',
    'Bus Lines',
    'Precious Stones and Metals'
]
df['is_high_risk_mcc'] = df['mcc_description'].isin(high_risk_cats).astype(int)

#combine risk factors: high debt or low income with expensive or high risk store
df['distress_luxury_risk'] = (
    ((df['debt_to_income_ratio'] > 1.5) | (df['yearly_income'] < 35000)) &
    ((df['is_high_risk_mcc'] == 1) | (df['amount'] > 500))
).astype(int)

#unusual purchase for age group like senior shopping late night or young customer spending over 500
senior_night = (df['current_age'] >= 65) & (df['is_night'] == 1)
young_outlier = (df['current_age'] <= 25) & ((df['amount'] > 500) | (df['is_high_risk_mcc'] == 1))
df['unusual_for_age'] = (senior_night | young_outlier).astype(int)

#save the new dataset
print(f"Feature engineering complete. Dataset shape: {df.shape}")
df.to_csv('data/train_tier2_sample.csv', index=False)
print("Saved enriched dataset to data/train_tier2_sample.csv")