# Build simulation pool dataset for testing in streamlit

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#load enriched dataset
print("Loading dataset...")
df = pd.read_csv('data/train_tier2_sample.csv')

#split into test set using same 20% split and random_state 42
_, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['target']
)
test_df = test_df.copy()

#filter out refund credits or negative amounts because we only want purchase debits
test_df = test_df[test_df['amount'] > 0].copy()
print(f"Test records for simulation: {len(test_df)}")

#get all fraud cases in the test set
frauds = test_df[test_df['target'] == 1].copy()

#sample different types of legit transactions for testing
legit_pool = test_df[test_df['target'] == 0].copy()

#out of state physical purchases to test false alarms
legit_travel = legit_pool[
    (legit_pool['is_diff_state'] == 1) & 
    (legit_pool['use_chip'] != 'Online Transaction')
].sample(n=50, random_state=42)

#online e-commerce purchases
legit_online = legit_pool[
    (legit_pool['use_chip'] == 'Online Transaction')
].sample(n=45, random_state=42)

#high value purchases over 250
legit_high_val = legit_pool[
    (legit_pool['amount'] > 250) & 
    (legit_pool['is_diff_state'] == 0) &
    (~legit_pool.index.isin(legit_online.index))
].sample(n=35, random_state=42)

#routine everyday local purchases under 100
legit_routine = legit_pool[
    (legit_pool['is_diff_state'] == 0) & 
    (legit_pool['amount'] <= 100) & 
    (legit_pool['use_chip'] != 'Online Transaction') &
    (~legit_pool.index.isin(legit_travel.index)) & 
    (~legit_pool.index.isin(legit_high_val.index))
].sample(n=70, random_state=42)

#combine legit samples
legit_combined = pd.concat([legit_travel, legit_online, legit_high_val, legit_routine])
print(f"Sampled legit transactions: {len(legit_combined)}")

#assign scenario tag for display in app
def assign_scenario_tag(row):
    if row['target'] == 1:
        if row['is_bad_credential'] == 1 or row['has_error'] == 1:
            return "🚨 Fraud: Credential Error / Testing Attack"
        elif row['is_high_risk_mcc'] == 1:
            return "🚨 Fraud: High-Risk Category (Tech/Jewelry/Transit)"
        elif row['distress_luxury_risk'] == 1:
            return "🚨 Fraud: High-Leverage Luxury Surge"
        elif row['use_chip'] == 'Online Transaction':
            return "🚨 Fraud: Account Takeover (Online Breach)"
        elif row['is_diff_state'] == 1:
            return "🚨 Fraud: Physical Out-of-State Counterfeit Swipe"
        else:
            return "🚨 Fraud: Compromised Card Attack"
    else:
        if row['is_diff_state'] == 1 and row['use_chip'] != 'Online Transaction':
            return "✈️ Legit: Out-of-State Travel (False Alarm Test)"
        elif row['amount'] > 250:
            return "💎 Legit: High-Value Authorized Purchase"
        elif row['use_chip'] == 'Online Transaction':
            return "🛒 Legit: In-State Online E-Commerce"
        else:
            return "🛡️ Legit: Everyday Local Routine Spend"

#combine fraud and legit samples and shuffle
sim_pool = pd.concat([frauds, legit_combined]).copy()
sim_pool['scenario_tag'] = sim_pool.apply(assign_scenario_tag, axis=1)
sim_pool = sim_pool.sample(frac=1.0, random_state=42).reset_index(drop=True)

#save simulation pool
output_path = 'data/simulation_pool.csv'
sim_pool.to_csv(output_path, index=False)

print(f"Total simulation records: {len(sim_pool)}")
print(f"Frauds: {(sim_pool['target'] == 1).sum()}")
print(f"Legit: {(sim_pool['target'] == 0).sum()}")
print(f"Saved simulation pool to {output_path}")
