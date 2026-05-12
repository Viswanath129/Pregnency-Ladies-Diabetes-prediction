## 2023-10-24 - Bypass pandas DataFrame instantiation for single-row inference
**Learning:** For scikit-learn models, initializing a single-row `pandas.DataFrame` purely to feed it into inference/transform methods introduces significant overhead.
**Action:** Always bypass `pandas.DataFrame` instantiation by providing a 2D `numpy.array` directly when inferring on single records to save on CPU time. Wrap inference blocks in `warnings.catch_warnings()` filtering `UserWarning` to cleanly avoid feature names missing warnings, and make sure to explicitly cast to standard Python float when needed.
