# Data

The raw dataset is not stored in this repository — it's ~2.5GB combined and exceeds GitHub's practical file size limits, and it's freely available directly from the source.

## Download instructions

1. Go to the [IEEE-CIS Fraud Detection competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) on Kaggle (a free Kaggle account is required).
2. Download `train_transaction.csv` and `train_identity.csv` from the Data tab.
3. Place both files in this `data/` folder before running the notebooks, or update the `base_path` variable at the top of each notebook to point wherever you saved them.

## Files expected here

- `train_transaction.csv` (590,540 rows × 394 columns)
- `train_identity.csv` (144,233 rows × 41 columns)

The notebooks join these two on `TransactionID` and derive all subsequent processed files (e.g. `train_transaction_identity.parquet`) from them.
