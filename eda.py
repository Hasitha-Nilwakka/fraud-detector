#Explotary Data Analysis
import pandas as pd
import json

#user data
user_data = pd.read_csv('data/users_data.csv')
##clean currency data - remove the $ sign
user_data["total_debt"] = user_data["total_debt"].astype(str).str.replace("$", "", regex=False).astype(float)
user_data["per_capita_income"] = user_data["per_capita_income"].astype(str).str.replace("$", "", regex=False).astype(float)
user_data["yearly_income"] = user_data["yearly_income"].astype(str).str.replace("$", "", regex=False).astype(float)

print("-" * 10)
print("User Data")
print(user_data.head(3))
print(user_data.shape)

#card data
card_data = pd.read_csv('data/cards_data.csv')
##clean currency data - remove the $ sign
card_data["credit_limit"] = card_data["credit_limit"].astype(str).str.replace("$", "", regex=False).astype(float)

print('-' * 10)
print("Card Data")
print(card_data.head(3))
print(card_data.shape)
print(card_data.value_counts("card_on_dark_web"))

#mcc codes
mcc_codes_data = pd.read_json('data/mcc_codes.json', typ='series').reset_index()
mcc_codes_data.columns = ['mcc_code', 'mcc_description']

print('-' * 10)
print("MCC Codes Data")
print(mcc_codes_data.head(3))
print(mcc_codes_data.shape)

#train fraud data
fraud_data = pd.read_json('data/train_fraud_labels.json')
print('-' * 10)
print("Fraud Data")
print(fraud_data.head(3))
print(fraud_data.shape)
print(fraud_data["target"].value_counts())

#transaction data
transaction_data = pd.read_csv("data/transactions_data.csv", nrows=5)
##clean currency data - remove the $ sign
transaction_data["amount"] = transaction_data["amount"].astype(str).str.replace("$", "", regex=False).astype(float)
##convert transactions to variables usable for training
transaction_data["hour"] = pd.to_datetime(transaction_data["date"]).dt.hour
transaction_data["day"] = pd.to_datetime(transaction_data["date"]).dt.day_of_week
transaction_data["is_night"] = ((transaction_data["hour"] >= 23) | (transaction_data["hour"] < 6)).astype(int)

print('-' * 10)
print("Transaction Data")
print(transaction_data.head())
print(transaction_data.shape)

#Merge card_data and transaction data
merged_df = pd.merge(card_data, transaction_data, left_on='id', right_on='card_id')
#Merge with user data
merged_df = pd.merge(merged_df, user_data, left_on='client_id_x', right_on='id')

#create new columns

#amount to credit limit
merged_df["amount_to_limit_ratio"] = merged_df["amount"] / merged_df["credit_limit"]
#debt to income
merged_df["debt_to_income_ratio"] = merged_df["total_debt"] / merged_df["yearly_income"]

print(merged_df.head(5))