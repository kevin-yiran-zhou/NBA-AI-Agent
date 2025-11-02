
import json, random, os

input_files = ["team.json", "player.json"]
output_all = "combined.json"
output_train = "train.json"
output_dev = "dev.json"
split_ratio = (0.8, 0.2)

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

# Split into train/dev
n = len(dataset)
n_train = int(n * split_ratio[0])
train = dataset[:n_train]
dev = dataset[n_train:]

with open(output_train, "w") as f:
    json.dump(train, f, indent=2)
with open(output_dev, "w") as f:
    json.dump(dev, f, indent=2)
print(f"Train: {len(train)} | Dev: {len(dev)}")
