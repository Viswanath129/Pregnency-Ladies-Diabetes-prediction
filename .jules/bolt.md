## 2025-02-28 - FastAPI async endpoint blocks thread pool with CPU bound code
**Learning:** In FastAPI, using `async def` for an endpoint means it runs directly in the asyncio event loop. If the endpoint performs CPU-bound operations (like ML inference, calculating probabilities), it will block the event loop, causing severe latency for all other concurrent requests.
**Action:** Use standard `def` instead of `async def` for endpoints that perform CPU-bound tasks (like model prediction). This allows FastAPI to run the handler in an external thread pool, preventing the main event loop from being blocked.

## 2025-02-28 - NumPy overhead for scalar values
**Learning:** In fast API loops, using NumPy functions (like `np.std`, `np.clip`, or `np.random.normal`) for scalar values or very small arrays introduces significant overhead compared to pure Python equivalents.
**Action:** Use pure Python equivalents (e.g., manual variance calculations, `min`/`max`, and `random.gauss`) instead of NumPy for scalar/small array math in hot paths.
