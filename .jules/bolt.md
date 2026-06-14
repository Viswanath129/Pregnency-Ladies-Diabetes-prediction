## 2024-05-18 - [FastAPI single-row inference performance]
**Learning:** Instantiating `pandas.DataFrame` for single-row inference and using `numpy` functions for scalar operations (e.g., `np.random.normal`, `np.std`) introduces significant and measurable overhead in fast API loops.
**Action:** Always prefer 2D `numpy.array` and native pure Python operations (`math`, `random`) for micro-optimizations in high-throughput endpoints. Ensure synchronous ML/CPU-bound inference runs with standard `def` in FastAPI to utilize external thread pools effectively.
