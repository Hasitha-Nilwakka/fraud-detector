#preparing the dataset for training

import pandas as pd

#load cards data and clean columns with $
cards_data = pd.read_csv('data/cards_data.csv')
cards_data["credit_limit"] = cards_data["credit_limit"].astype(str).str.replace("$", "", regex=False).astype(float)

#load user_data as clean columns with $
users_data = pd.read_csv('data/users_data.csv')
users_data["per_capita_income"] = users_data["per_capita_income"].astype(str).str.replace("$", "", regex=False).astype(float)
users_data["yearly_income"] = users_data["yearly_income"].astype(str).str.replace("$", "", regex=False).astype(float)
users_data["total_debt"] = users_data["total_debt"].astype(str).str.replace("$", "", regex=False).astype(float)

#load mcc_code
mcc_codes = pd.read_json('data/mcc_codes.json', orient='index')
mcc_codes = mcc_codes.reset_index()
mcc_codes.columns = ['mcc_code', 'mcc_description']

#load train fraud data
train_fraud_data = pd.read_json('data/train_fraud_labels.json')

#load transactions data and clean columns with $
transactions_data = pd.read_csv('data/transactions_data.csv', nrows=200_000)
transactions_data["amount"] = transactions_data["amount"].astype(str).str.replace("$", "", regex=False).astype(float)

#map the fraud_data to transaction data -> transaction id against fraud data : id
transactions_data["target"] = transactions_data["id"].map(train_fraud_data["target"])

#the original trainfraud lables dataset has 67% transactions labled, rest was kept for the competition
#we need to drop those transaction since we dont have a use for them
#now we have mapped the fraud lable to transaction id we drop the NaN
transactions_data = transactions_data.dropna(subset=["target"])
#convert the fraud detection lable to int Fraud = 1 and No Fraud = 0
transactions_data["target"] = (transactions_data["target"].str.lower() == 'yes').astype(int)

#convert date to further sub columns
tx_dates = pd.to_datetime(transactions_data["date"])
transactions_data["hour"] = tx_dates.dt.hour
transactions_data["day"] = tx_dates.dt.day_of_week
transactions_data["is_night"] = ((transactions_data["hour"] >= 23) | (transactions_data["hour"] < 6)).astype(int)

#merge transaction data with mcc codes
merged_df = pd.merge(transactions_data, mcc_codes, left_on="mcc", right_on="mcc_code", how='left')

#merge card_data with transaction data
merged_df = pd.merge(merged_df, cards_data, left_on='card_id', right_on='id', how='left', suffixes=("", "_card"))

#merge user data with transactions
merged_df = pd.merge(merged_df, users_data, left_on='client_id', right_on='id', how='left', suffixes=("", "_user"))

#calculate engineered data
merged_df["amount_to_limit_ratio"] = merged_df["amount"] / merged_df["credit_limit"]
merged_df["debt_to_income_ratio"] = merged_df["total_debt"] / merged_df["yearly_income"]

merged_df.to_csv("data/train_sample.csv", index=False)
print("Saved the merged dataset")