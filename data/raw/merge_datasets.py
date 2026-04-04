import argparse
from pathlib import Path
import pandas as pd


def merge_datasets(mockaroo_files, generated_path, output_path):
    mock_dfs = [pd.read_csv(str(p)) for p in mockaroo_files]
    mockaroo_df = pd.concat(mock_dfs, ignore_index=True)
    mockaroo_df = mockaroo_df.rename(columns={
        "txn_id": "transaction_id",
        "merchant_cat": "merchant_category",
    })
    mockaroo_df["customer_id"] = "UNKNOWN"
    mockaroo_df["is_international"] = 0
    mockaroo_df["transaction_status"] = "approved"
    mockaroo_df["fraud_reasons"] = "none"

    generated_df = pd.read_csv(str(generated_path))
    common_cols = list(set(mockaroo_df.columns) & set(generated_df.columns))
    merged = pd.concat(
        [generated_df[common_cols], mockaroo_df[common_cols]],
        ignore_index=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Merge generated and Mockaroo datasets.")
    parser.add_argument(
        "--generated",
        default="data/raw/transactions.csv",
        help="Path to generated transactions CSV.",
    )
    parser.add_argument(
        "--mockaroo",
        nargs="+",
        default=[
            "data/raw/MOCK_DATA (1).csv",
            "data/raw/MOCK_DATA (2).csv",
            "data/raw/MOCK_DATA (3).csv",
            "data/raw/MOCK_DATA (4).csv",
            "data/raw/MOCK_DATA (5).csv",
        ],
        help="List of Mockaroo CSV files to merge.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/final_merged_dataset.csv",
        help="Output merged CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merged = merge_datasets(args.mockaroo, args.generated, args.output)
    print(f"Mockaroo + generated merged shape: {merged.shape}")
    print(f"✅ Final dataset saved at: {args.output}")


if __name__ == "__main__":
    main()
