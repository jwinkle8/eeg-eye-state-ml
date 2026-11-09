# data/raw

The untouched download of the EEG Eye State dataset. The pipeline never modifies this file.

## Contents

| File | Description |
|---|---|
| `eeg_eye_state.parquet` | `fetch_openml(data_id=1471, as_frame=True).frame`, saved as is (Parquet, snappy compression) |

## How it is created

`load_data()` in `src/data.py` downloads the data from OpenML only if this file does not exist. Later runs read the cached copy, so every run trains on the same data even if the source changes or goes offline. To re-download, delete the file and run `python run_pipeline.py`.

## Schema

14,980 rows × 15 columns, no missing values. Rows are in recording order (one row per sample, ~8 ms apart).

| Column | Type | Description |
|---|---|---|
| `V1` … `V14` | float64 | EEG voltage in µV for each of the 14 electrodes, in this order: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 |
| `Class` | category (`"1"`, `"2"`) | Eye state: `"1"` = open (8,257 rows), `"2"` = closed (6,723 rows) |

Known issues (handled in `data/clean/`): generic column names, string label codes, and a few extreme spike values (up to ~716,000 µV).
