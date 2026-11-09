# EEG Eye State Classifier

Predict whether a person's eyes are open or closed from 14-channel EEG readings.

## Data card

| | |
|---|---|
| **Source** | [EEG Eye State (OpenML ID 1471)](https://www.openml.org/d/1471), originally from the UCI Machine Learning Repository |
| **Access** | Downloaded with `sklearn.datasets.fetch_openml(data_id=1471)` on the first run, then cached as Parquet in `data/raw/` |
| **Collection** | One subject, one continuous recording with an Emotiv EEG headset. Eye state was labeled by hand from video of the session. |
| **Target** | Eye state. OpenML codes it `1` = open (8,257 rows) and `2` = closed (6,723 rows); the pipeline maps these to `0` = open, `1` = closed. |
| **Volume** | 14,980 rows × 14 EEG channels + 1 label (about 277 KB as Parquet) |
| **Variety** | Structured, numeric sensor time series. All 14 features are voltages in µV, one per electrode (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4). No missing values. |
| **Velocity** | Static historical dataset, not a stream. The recording ran 117 seconds, so 14,980 samples works out to about 128 samples per second (~8 ms apart). |
| **Veracity** | A few single-sample spikes far outside the normal range (up to ~716,000 µV) look like sensor glitches; 4 rows are removed (see below). Readings carry a large, slowly drifting baseline offset. Labels were assigned by hand, and the data covers one person. |

Folder-level details are in [`data/raw/README.md`](data/raw/README.md) and [`data/clean/README.md`](data/clean/README.md).

### Outlier removal

I chose the bounds from data statistics: each channel's min, median, and max over the full recording. Channel medians sit around 4,000–4,600 µV, while a handful of readings jump to 100,000–700,000 µV and a few dip well below 2,000 µV. I dropped any row with a channel outside [2,000, 6,500] µV, which removed 4 rows (raw row indices 898, 10386, 11509, and 13179).

Domain knowledge is a sanity check on those bounds. Raw Emotiv values carry a DC offset of ~4,100–4,200 µV that drifts slowly (Emotiv). Scalp EEG is typically 10–100 µV, and the largest normal artifacts (eye blinks and closures, strongest at frontal electrodes) reach several hundred µV (Behavior Research Methods paper on eye artifacts; COGS 179 lecture notes). The bounds are the offset ± ~2,000 µV: several times larger than any normal signal and within the headset's ~8,400 µV hardware range.

Because the bounds were picked by looking at the whole dataset, including rows that later land in validation and test, this is a mild form of leakage. Its effect is small: the rule removes 4 of 14,980 rows, and nothing the model learns is fit on those statistics.

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
I started with a chronological split because the rows are a time series: shuffling them drops their order and lets near-identical neighboring samples land in both train and test, so keeping the order seemed like the more honest test of prediction. The results were poor. On test the model reached 0.237 accuracy and 0.235 macro F1, far below the 0.906 majority-class baseline, and it predicted "closed" for most rows even with balanced class weights. Eye state is not spread evenly through the recording (train was 61% closed while test was 91% open), and that class imbalance between segments was detrimental to training the model.

With the random stratified split, every part has the same 55/45 balance, and the model beats the baseline: 0.653 test accuracy and 0.651 macro F1 against 0.551. That gain comes with the leakage described under Limitations, so these scores are an upper bound for this recording rather than evidence that the model would work on new data.

## Limitations
- **The random split leaks across time.** Rows are sampled at ~128 Hz, so most test rows have a near-identical neighbor (~8 ms apart) in the training set. The scores above are an optimistic upper bound for this one recording, not an estimate of performance on new recordings or new people.
- **One subject, one session.** The whole dataset is a single ~117-second recording, so results may not generalize beyond it.

## Next steps
- **Leak-resistant evaluation:** split the recording into contiguous blocks (with a small gap between train and test blocks) and use blocked cross-validation, so test rows have no near-identical neighbors in training while each fold still contains both eye states.
- **Handle baseline drift:** remove the slowly drifting offset (for example, high-pass filtering or subtracting a rolling mean) so the model sees signal changes rather than where in the recording a sample falls.
- **Windowed features:** compute features over short windows instead of single samples, such as power in the alpha band (8–12 Hz), which typically rises over the occipital electrodes (O1, O2) when the eyes close.
- **Re-check model settings on the final split:** the `class_weight="balanced"` choice was made on the chronological split; compare it with default weights, and try a few regularization strengths (`C`), on the random split's validation set.
- **Fit the outlier bounds on training data only:** choose the thresholds after splitting, from training rows, to remove the mild leakage described under Outlier removal.
- **Compare other models:** try a nonlinear model such as a random forest under the leak-resistant split to see whether the gap over the baseline holds.
- **More data:** add recordings from other subjects or sessions to test whether the model generalizes beyond one person.

## Use of AI
This project was generated with the help of a generative AI coding assistant while making the project decisions myself.

- **Planning:** an initial plan was drafted with Claude Desktop. I then had Claude Code check it against my course notes and revised it, choosing a three-way split and a narrower scope (one model, no hypothesis testing).
- **Code review and debugging:** Claude reviewed my code for bugs and leakage (for example, a column rename that had no effect, unmapped labels, and a validation split drawn from the full dataset instead of the remaining rows) and explained errors as they came up.
- **Code writing:** at my request, Claude wrote parts of the code, including fixes in `clean_data`, the split functions, `build_model`, `evaluate`/`save_results`, and changes to `run_pipeline.py`. I reviewed each change.
- **Documentation:** Claude drafted parts of this README and the `data/raw` and `data/clean` READMEs, based on my direction and my interpretation of the results.
- **My decisions:** the outlier thresholds, the move from a chronological to a random split, keeping balanced class weights, the metrics to report, the random seed, and the conclusions in Takeaways. All reported numbers come from my own pipeline runs.
