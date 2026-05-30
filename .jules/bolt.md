## 2024-05-18 - [FastAPI Event Loop Blocking & Pandas Overhead]
**Learning:** CPU-bound operations like single-row ML inference in FastAPI `async def` endpoints block the event loop, tanking concurrency. Additionally, `pandas.DataFrame` instantiation adds massive overhead relative to pure NumPy arrays during single-row inferences.
**Action:** Use `def` (sync) for endpoints running scikit-learn models so FastAPI delegates to a threadpool. Swap `pandas.DataFrame` for `numpy.array` when transforming single inputs, suppressing missing feature names with `warnings.catch_warnings()`.
