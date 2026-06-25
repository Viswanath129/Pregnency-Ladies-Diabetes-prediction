## 2024-03-24 - [FastAPI Thread Pool Warnings]
**Learning:** `warnings.catch_warnings()` is not thread-safe. When changing an `async def` FastAPI route to `def` so it runs in an external thread pool, wrapping inference code in `warnings.catch_warnings()` to silence sklearn's `UserWarning` (missing feature names) can cause race conditions that permanently suppress warnings globally.
**Action:** Use global `warnings.filterwarnings("ignore", category=UserWarning)` at the module level instead of `catch_warnings` for thread-pooled API endpoints.

## 2024-03-24 - [Python Micro-optimizations]
**Learning:** For extremely fast, sub-millisecond API loops, instantiating `pandas.DataFrame` or using `numpy` functions for scalar operations (e.g., `np.std` for 3 values, `np.clip`, `np.random.normal`) adds measurable overhead (e.g. going from 0.68s down to 0.01s for 1000 iterations).
**Action:** Use `numpy.array` instead of `pandas.DataFrame` for single-row inference, and standard library equivalents (`random.gauss`, `max`/`min`, manual math formulas) for scalar operations in tight loops.
