## 2024-06-27 - [Avoid async def for CPU-bound routes in FastAPI]
**Learning:** The FastAPI `/predict` route uses `async def` but performs CPU-bound scikit-learn model inference synchronously. This blocks the main event loop, severely degrading throughput and increasing latency under load.
**Action:** Change `async def predict_risk` to `def predict_risk`. FastAPI will automatically run synchronous handlers in a threadpool, preventing event loop blocking.

## 2024-06-27 - [Avoid dataframe instantiations for single row prediction]
**Learning:** Creating pandas dataframes inside a prediction loop is very slow. It is faster to use a 2d numpy array if we don't care about the feature names warning.
**Action:** Use 2D numpy arrays and global warning filter instead of pandas dataframe.

## 2024-06-27 - [Avoid np functions for small scalars]
**Learning:** `np.clip`, `np.std`, `np.random.normal` introduce overhead when dealing with scalar values or very small lists, pure Python operations are faster.
**Action:** Replace `np.clip` with `min/max`, replace `np.std` with manual variance calc, and `np.random.normal` with `random.gauss` where appropriate.
