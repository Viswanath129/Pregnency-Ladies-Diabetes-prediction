## 2024-05-08 - CPU-bound operations and Pandas DataFrame overhead

**Learning:** FastAPI's `async def` runs on the main event loop. For CPU-bound operations like Scikit-learn or other ML inference, using `async def` blocks the event loop, causing poor concurrency and increasing response time. Also, constructing a `pandas.DataFrame` for single-row inference introduces a massive overhead (up to ~200x slower) compared to a simple 2D `numpy.array`.

**Action:** For ML inference endpoints in FastAPI, always use standard `def` (which runs in a separate threadpool), and use 2D `numpy.array` instead of `pandas.DataFrame` for single-row predictions. Wrap the call in `warnings.catch_warnings()` to suppress missing feature names warnings and explicitly cast the output to a python `float()` for type consistency.
