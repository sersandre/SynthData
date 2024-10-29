import json


data = "results_MVIP_synthetic_real_mixed.json"

with open(data, "r") as f:
    out = json.load(f)

for _ in out.keys():
    print(f"{out[_]['output']},")