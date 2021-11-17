import json


with open(
    "./training_data/variant_1.json",
) as f:
    parsed = json.load(f)


print(json.dumps(parsed, indent=4))
