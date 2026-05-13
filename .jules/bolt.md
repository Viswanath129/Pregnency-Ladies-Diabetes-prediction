## 2024-05-13 - [FastAPI ML Inference Pandas Bottleneck]
**Learning:** [Using `pandas.DataFrame` for single-row ML inference in a FastAPI loop introduces a massive serialization/instantiation bottleneck. Switching to raw `numpy.array` provided a ~6.8x speedup locally but requires handling JSON serialization of numpy floats and suppressing sklearn feature name warnings.]
**Action:** [For real-time single-row endpoints, avoid `pd.DataFrame` entirely. Use `np.array`, catch warnings natively, and ensure casting with standard `float()`.]
