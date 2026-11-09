# data/clean

The cleaned dataset the model trains on. It is generated from `data/raw/` and can always be rebuilt from it.

## Contents

| File | Description |
|---|---|
| `eeg_eye_state_clean.parquet` | Output of `clean_data()` in `src/data.py` (Parquet, snappy compression, no index) |

## How it is created

`clean_data()` rewrites this file on every run of `python run_pipeline.py`. Starting from the raw snapshot, it:

1. Asserts the expected shape (14,980 × 15), column names, and no missing values
2. Renames `V1` … `V14` to electrode names
3. Maps the label to integers: `0` = open, `1` = closed
4. Drops rows with any channel outside [2,000, 6,500] µV (4 rows; see "Outlier removal" in the main README)
5. Resets the index to 0 … n-1, keeping recording order

The dropped row indices and counts are also saved in `results/metrics.json` under `cleaning`.

## Schema

14,976 rows × 15 columns, no missing values. Rows are in recording order.

| Column | Type | Description |
|---|---|---|
| `AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, `AF4` | float64 | EEG voltage in µV for each electrode |
| `Class` | int64 | Eye state: `0` = open (8,254 rows), `1` = closed (6,722 rows) |
