
import json, random, os

input_files = ["team.json", "player.json"]
output_all = "combined.json"
output_train = "train.json"
output_val = "val.json"
output_test = "test.json"
split_ratio = (0.8, 0.1, 0.1)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Combine
dataset = []
for file in input_files:
    if os.path.exists(file):
        dataset += load_json(file)
    else:
        print(f"{file} not found")

random.shuffle(dataset)
with open(output_all, "w") as f:
    json.dump(dataset, f, indent=2)
print(f"Combined dataset saved to {output_all} with {len(dataset)} examples.")

# Split into train/val/test (80-10-10)
n = len(dataset)
n_train = int(n * split_ratio[0])
n_val = int(n * split_ratio[1])
train = dataset[:n_train]
val = dataset[n_train:n_train + n_val]
test = dataset[n_train + n_val:]

with open(output_train, "w") as f:
    json.dump(train, f, indent=2)
with open(output_val, "w") as f:
    json.dump(val, f, indent=2)
with open(output_test, "w") as f:
    json.dump(test, f, indent=2)
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
