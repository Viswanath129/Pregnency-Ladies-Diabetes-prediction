
## 2024-06-04 - [Single-Row Inference Bottlenecks]
**Learning:** In fast FastAPI loops, using `pd.DataFrame` and NumPy functions (`np.clip`, `np.std`, `np.random.normal`) for single-row inference or scalar values introduces massive overhead. Also, CPU-bound endpoints (like scikit-learn predictions) should use synchronous `def` rather than `async def` to avoid blocking the event loop and to allow FastAPI to execute them in an external thread pool.
**Action:** Always bypass pandas by using 2D `numpy.array` wrapped in `warnings.catch_warnings()` for scikit-learn predictions, use pure Python functions for simple scalar math, and ensure CPU-bound endpoints use synchronous `def`.
