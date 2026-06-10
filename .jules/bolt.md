## 2024-05-24 - [Avoid numpy standard functions for small array/scalar ops]
**Learning:** In fast loops or tight API endpoints, using NumPy standard math functions (like `np.clip`, `np.std`, or `np.random.normal`) for scalars or small 1D arrays introduces significant overhead compared to plain Python equivalents (like `min/max`, manual variance, and `random.gauss`), likely due to numpy's underlying C-API transition overhead on trivial amounts of data.
**Action:** Use Python's built-in scalar math or the `math`/`random` standard libraries instead of numpy for tiny/scalar calculations.

## 2024-05-24 - [Scikit-learn inference with Numpy vs Pandas DataFrames]
**Learning:** `pandas.DataFrame` instantiation has non-trivial overhead. For single-row predictions via scikit-learn models (like `predict`, `predict_proba`), feeding a 2D numpy array directly (e.g., `np.array([vitals])`) skips the DataFrame creation overhead and is notably faster.
**Action:** Always bypass pandas for single-row ML inference in APIs. Note that doing so requires suppressing `UserWarning` about missing feature names and often explicitly casting outputs to standard python floats using `float()`.

## 2024-05-24 - [FastAPI Threading for CPU-bound operations]
**Learning:** Scikit-learn inference (and other ML operations) are typically synchronous and CPU-bound. Placing these in an `async def` FastAPI endpoint blocks the main asyncio event loop, tanking throughput under concurrent load.
**Action:** Define CPU-bound API endpoints with standard `def` (instead of `async def`) so FastAPI automatically dispatches them to a thread pool via AnyIO.
