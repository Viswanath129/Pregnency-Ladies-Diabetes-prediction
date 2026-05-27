## 2025-02-20 - Fast Single-Row Inference in Scikit-Learn / FastAPI CPU-bound Blocking

**Learning:**
1) In FastAPI, CPU-bound endpoints (like Scikit-Learn predictions) should use standard `def` instead of `async def` so they are executed in an external threadpool and do not block the main event loop.
2) Single-row scikit-learn model inferences have significant overhead if using `pandas.DataFrame`. Bypassing this by using a 2D `numpy.array` improves inference speed.

**Action:**
1) Use standard `def` for FastAPI endpoints containing CPU-intensive or blocking tasks.
2) For single-row predictions in Scikit-learn, use 2D `numpy.array` instead of `pandas.DataFrame`. Wrap inference calls in `warnings.catch_warnings()` to safely suppress `UserWarning` about missing feature names, and explicitly cast numpy outputs back to standard Python types like `float()`.
