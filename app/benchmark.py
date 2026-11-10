"""Measure single-request latency of the running API (start uvicorn first)."""
import json
import statistics
import time
import urllib.request

from src.data import load_data, time_split

N_WARMUP, N_MEASURE = 20, 200

_, val, _ = time_split(load_data())
row = val.iloc[0]
cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
body = json.dumps({c: float(row[c]) for c in cols}).encode()


def one_call():
    req = urllib.request.Request("http://127.0.0.1:8000/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    urllib.request.urlopen(req).read()
    return (time.perf_counter() - start) * 1000   # milliseconds


for _ in range(N_WARMUP):
    one_call()
times = sorted(one_call() for _ in range(N_MEASURE))
print(f"requests: {N_MEASURE}")
print(f"mean {statistics.mean(times):.1f} ms | median {statistics.median(times):.1f} ms | "
      f"p95 {times[int(0.95 * N_MEASURE) - 1]:.1f} ms | max {times[-1]:.1f} ms")
