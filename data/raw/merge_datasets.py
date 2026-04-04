import pandas as pd
import os

# ===============================
# STEP 1: Load Mockaroo files
# ===============================

mockaroo_files = [
    "data/raw/MOCK_DATA (1).csv",
    "data/raw/MOCK_DATA (2).csv",
    "data/raw/MOCK_DATA (3).csv",
    "data/raw/MOCK_DATA (4).csv",
    "data/raw/MOCK_DATA (5).csv"
]

mock_dfs = []

for file in mockaroo_files:
    df = pd.read_csv(file)
    mock_dfs.append(df)

mockaroo_df = pd.concat(mock_dfs, ignore_index=True)

print("Mockaroo merged shape:", mockaroo_df.shape)


# ===============================
# STEP 2: Fix column names (IMPORTANT)
# ===============================

mockaroo_df.rename(columns={
    "txn_id": "transaction_id",
    "merchant_cat": "merchant_category"
}, inplace=True)

# Add missing columns to match generated dataset
mockaroo_df["customer_id"] = "UNKNOWN"
mockaroo_df["is_international"] = 0
mockaroo_df["transaction_status"] = "approved"
mockaroo_df["fraud_reasons"] = "none"


# ===============================
# STEP 3: Load programmatic dataset
# ===============================

generated_df = pd.read_csv("data/raw/transactions.csv")

print("Generated dataset shape:", generated_df.shape)


# ===============================
# STEP 4: Align columns (VERY IMPORTANT)
# ===============================

# Keep only common columns
common_cols = list(set(mockaroo_df.columns) & set(generated_df.columns))

mockaroo_df = mockaroo_df[common_cols]
generated_df = generated_df[common_cols]


# ===============================
# STEP 5: Merge both datasets
# ===============================

final_df = pd.concat([generated_df, mockaroo_df], ignore_index=True)

print("Final dataset shape:", final_df.shape)


# ===============================
# STEP 6: Save merged dataset
# ===============================

output_path = "final_merged_dataset.csv"
final_df.to_csv(output_path, index=False)

print(f"✅ Final dataset saved at: {output_path}")