## 2023-10-27 - [FastAPI ML Inference Blocking]
**Learning:** [Using `async def` for CPU-bound tasks like scikit-learn model inference in FastAPI blocks the event loop, causing poor performance under load.]
**Action:** [Use standard `def` for route handlers that perform synchronous blocking work so FastAPI can offload the execution to an external threadpool.]

## 2023-10-27 - [NumPy Overhead on Scalars]
**Learning:** [Using numpy functions like `np.clip`, `np.std`, or `np.random.normal` for single scalar values or very small arrays introduces significant function-call overhead that dwarfs the actual mathematical operation.]
**Action:** [Use native Python modules (`math`, `random`) or explicit manual calculations for operations on individual scalar values.]
