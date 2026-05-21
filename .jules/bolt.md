
## 2024-05-18 - Fast API Sync vs Async CPU Bound Performance
**Learning:** In FastAPI, using `async def` for endpoints that perform synchronous, CPU-bound operations (like `joblib` machine learning model inferences) blocks the main event loop, severely degrading performance under load. `pandas` DataFrame instantiation for single rows during inference is also surprisingly slow.
**Action:** Always use `def` (synchronous) instead of `async def` for endpoints performing CPU-bound work in FastAPI so it runs in an external thread pool. Bypass `pandas.DataFrame` instantiation by converting single rows directly to 2D `numpy.array` before passing to scikit-learn model `predict`/`predict_proba` functions. Suppress scikit-learn missing feature warnings with `warnings.catch_warnings()`.
