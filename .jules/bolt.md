
## 2024-05-23 - [FastAPI inference bottleneck]
**Learning:** In a FastAPI app, CPU-bound operations like scikit-learn ML inference inside an `async def` function will block the entire main event loop, severely limiting concurrency. Changing the function to `def` allows FastAPI to automatically execute it in a separate external threadpool, making the system much more concurrent and drastically reducing latency.
**Action:** Always make API endpoints that use CPU-bound operations, like ML inference or synchronous blocking processes, `def` instead of `async def` in FastAPI.

## 2024-05-23 - [pandas overhead for single-row inference]
**Learning:** Instantiating a `pd.DataFrame` inside the hotpath for a single-row inference comes with huge performance overhead compared to directly passing a 2D `numpy.array` to the scikit-learn models. Passing raw arrays reduces overhead and significantly speeds up real-time inference latency. Note that doing this will cause scikit-learn to complain via a `UserWarning` if the model was trained on dataframes with feature names.
**Action:** Use `np.array([features])` instead of `pd.DataFrame` for fast single-row inference, and wrap the call in `with warnings.catch_warnings(): warnings.simplefilter("ignore", UserWarning)` to safely suppress the missing-feature-names warning without affecting application logic. Always explicitely cast numpy scalar outputs to float for JSON parsing compatibility.
