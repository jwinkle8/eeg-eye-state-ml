# EEG Eye State Classifier

Predict whether a person's eyes are open or closed from 14-channel EEG readings.

## Dataset
- **Source:** [EEG Eye State (OpenML ID 1471)](https://www.openml.org/d/1471), originally from the UCI Machine Learning Repository
- **Size:** ~15,000 samples, 14 EEG channels from one continuous recording
- **Target:** eye state (open / closed)
- **Access:** downloaded at runtime with `sklearn.datasets.fetch_openml`, so no manual download is needed

### Outlier removal

Raw Emotiv values carry a DC offset of ~4,100–4,200 µV that drifts slowly (Emotiv). Scalp EEG is typically 10–100 µV, and the largest normal artifacts (eye blinks and closures, strongest at frontal electrodes) reach several hundred µV (Behavior Research Methods paper on eye artifacts; COGS 179 lecture notes). I dropped any row with a channel outside [2,000, 6,500] µV. That is the offset ± ~2,000 µV, several times larger than any normal signal and well within the headset’s ~8,400 µV hardware range. This removed 4 rows. The thresholds come from domain knowledge rather than data statistics, so no test-set information leaks into training.

## How to run
```bash
pip install -r requirements.txt
python run_pipeline.py
```
Metrics are written to `results/metrics.json`. The raw download is saved to `data/raw/eeg_eye_state.parquet`.

## Pipeline
1. **Load:** fetch from OpenML
2. **Snapshot:** save the untouched data to Parquet so the exact training data is versioned with the code
3. **Clean:** TODO (how missing values and outliers were handled)
4. **Split:** chronological 60/20/20. The first 60% of the recording is for training, the next 20% for validation, and the last 20% for testing. Rows are never shuffled, because neighboring samples are nearly identical and shuffling would leak them across splits.
5. **Train:** logistic regression, with standardization inside a scikit-learn Pipeline so the scaler is fit on training rows only
6. **Evaluate:** accuracy, macro F1, and confusion matrix on validation, then once on test

## Results

| Split | Accuracy | Macro F1 |
|---|---|---|
| Validation | TODO | TODO |
| Test | TODO | TODO |

## Takeaways
TODO: 2–3 sentences. How well does logistic regression predict eye state on later parts of the recording, and how do validation and test compare? Why?
