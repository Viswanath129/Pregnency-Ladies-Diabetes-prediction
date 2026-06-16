## 2025-02-20 - [FastAPI ML Inference Overhead]
**Learning:** In fast API loops, using `pandas.DataFrame` and NumPy scalar functions (like `np.clip` or `np.std`) for single-row inference introduces significant latency.
**Action:** Always replace DataFrame instantiations with `numpy.array` and use pure Python equivalents for scalar operations (`min/max`, `random.gauss`, manual variance calculation) in latency-sensitive endpoints.
