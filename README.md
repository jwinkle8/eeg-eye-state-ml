# EEG Eye State Classifier

Predict whether a person's eyes are open or closed from 14-channel EEG readings.

## Dataset
- **Source:** [EEG Eye State (OpenML ID 1471)](https://www.openml.org/d/1471), originally from the UCI Machine Learning Repository
- **Size:** ~15,000 samples, 14 EEG channels from one continuous recording
- **Target:** eye state (open / closed)
- **Access:** downloaded at runtime with `sklearn.datasets.fetch_openml`, so no manual download is needed

### Outlier removal

Raw Emotiv values carry a DC offset of ~4,100–4,200 µV that drifts slowly (Emotiv). Scalp EEG is typically 10–100 µV, and the largest normal artifacts (eye blinks and closures, strongest at frontal electrodes) reach several hundred µV (Behavior Research Methods paper on eye artifacts; COGS 179 lecture notes). I dropped any row with a channel outside [2,000, 6,500] µV. That is the offset ± ~2,000 µV, several times larger than any normal signal and well within the headset’s ~8,400 µV hardware range. This removed N rows, consistent with the 3 outliers reported in prior work (arXiv 2209.01023). The thresholds come from domain knowledge rather than data statistics, so no test-set information leaks into training.

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
4. **Split:** 80/20, done two ways:
   - *Random:* shuffled, stratified
   - *Chronological:* first 80% of the recording for training, last 20% for testing
5. **Train:** majority-class baseline, logistic regression, random forest
6. **Evaluate:** test accuracy and F1

## Results

| Split | Model | Test Accuracy | F1 |
|---|---|---|---|
| Random | Baseline | TODO | TODO |
| Random | Logistic Regression | TODO | TODO |
| Random | Random Forest | TODO | TODO |
| Chronological | Baseline | TODO | TODO |
| Chronological | Logistic Regression | TODO | TODO |
| Chronological | Random Forest | TODO | TODO |

## Takeaways
TODO: 2–3 sentences. Which model won, and how much did accuracy change between the random and chronological splits? Why?
