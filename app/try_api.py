"""Send one real validation transaction to the running API."""
import json
import urllib.request

from src.data import load_data, time_split

_, val, _ = time_split(load_data())
row = val[val["Class"] == 1].iloc[0]
cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
payload = {c: float(row[c]) for c in cols}

req = urllib.request.Request("http://127.0.0.1:8000/predict",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
