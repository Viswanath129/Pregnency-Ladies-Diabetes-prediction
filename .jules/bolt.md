## 2023-10-27 - Scikit-Learn Inference Bottleneck with Pandas
**Learning:** Initializing a pandas.DataFrame just to process a single row for scikit-learn models introduces measurable overhead in high-throughput endpoints.
**Action:** Always bypass pandas for single-row/real-time inference by using 2D numpy arrays instead. Wrap the prediction block in `warnings.catch_warnings()` if the models emit `UserWarning` about missing feature names (since they were likely trained on pandas DataFrames).
