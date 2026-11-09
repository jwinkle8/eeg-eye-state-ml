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
The pipeline writes three files:
- `data/raw/eeg_eye_state.parquet`: the untouched download (only fetched if missing)
- `data/clean/eeg_eye_state_clean.parquet`: the cleaned data the model trains on
- `results/metrics.json`: cleaning details, split sizes and class balance, and validation/test metrics

## Pipeline
1. **Load:** fetch from OpenML
2. **Snapshot:** save the untouched data to Parquet so the exact training data is versioned with the code
3. **Clean:** assert the expected shape (14,980 × 15) and no missing values, rename columns to electrode names, map labels to 0 = open / 1 = closed, and drop 4 artifact rows (see [Outlier removal](#outlier-removal))
4. **Split:** random 60/20/20 train/validation/test, stratified by class (`random_state=86`). Each part is 55.1% open / 44.9% closed.
5. **Train:** logistic regression with `class_weight="balanced"`, with standardization inside a scikit-learn Pipeline so the scaler is fit on training rows only
6. **Evaluate:** accuracy, macro F1, and confusion matrix on validation, then once on test

## Results

| Part | Rows | Accuracy | Macro F1 | Majority-class share | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|
| Validation | 2,995 | 0.599 | 0.598 | 0.551 | [[986, 665], [536, 808]] |
| Test | 2,996 | 0.653 | 0.651 | 0.551 | [[1087, 564], [476, 869]] |

"Majority-class share" is the accuracy of always predicting the most common class (open), the baseline the model has to beat. 1 = closed is the positive class in the confusion matrix.

### Choices made along the way
- **Class weights:** with an earlier chronological split, `class_weight="balanced"` raised validation macro F1 from 0.263 to 0.306 over default weights, so I kept it. It was not re-compared on the random split, where the classes are nearly balanced.
- **Split:** I first tried a chronological split (first 60% train, next 20% validation, last 20% test). Class balance shifted across the recording (train 61% closed, test 91% open), and the model scored below the majority-class baseline (test accuracy 0.237, macro F1 0.235 vs. a 0.906 baseline). I switched to the random split. Those chronological numbers come from an earlier version of this pipeline, not from the current `metrics.json`.

## Takeaways
TODO: 2–3 sentences. How much does logistic regression beat the majority-class baseline, and what does the gap between the random and chronological results suggest?

## Limitations
- **The random split leaks across time.** Rows are sampled at ~128 Hz, so most test rows have a near-identical neighbor (~8 ms apart) in the training set. The scores above are an optimistic upper bound for this one recording, not an estimate of performance on new recordings or new people.
- **One subject, one session.** The whole dataset is a single ~117-second recording, so results may not generalize beyond it.
