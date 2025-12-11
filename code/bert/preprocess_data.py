import json
import os
from collections import defaultdict

def preprocess_dataset(all_json_path, output_dir, train_per_combo=16, val_per_combo=2, test_per_combo=2):
    """Split dataset into train/val/test sets by intent-attribute combination."""
    # Load all data
    with open(all_json_path, 'r') as f:
        data = json.load(f)
    
    # Group examples by (intent, attribute) combination
    grouped = defaultdict(list)
    for example in data:
        intent = example["intent"]
        attr = example["slots"]["attribute"]
        key = (intent, attr)
        grouped[key].append(example)
    
    # Split each group
    train_data = []
    val_data = []
    test_data = []
    
    for (intent, attr), examples in sorted(grouped.items()):
        n_examples = len(examples)
        expected = train_per_combo + val_per_combo + test_per_combo
        
        if n_examples != expected:
            print(f"⚠️  Warning: {intent}-{attr} has {n_examples} examples (expected {expected})")
            # Adjust split if needed
            actual_train = min(train_per_combo, n_examples)
            remaining = n_examples - actual_train
            actual_val = min(val_per_combo, remaining)
            actual_test = remaining - actual_val
        else:
            actual_train = train_per_combo
            actual_val = val_per_combo
            actual_test = test_per_combo
        
        # Split by order
        train_data.extend(examples[:actual_train])
        val_data.extend(examples[actual_train:actual_train + actual_val])
        test_data.extend(examples[actual_train + actual_val:actual_train + actual_val + actual_test])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    n_total = len(train_data) + len(val_data) + len(test_data)
    n_combos = len(grouped)
    
    print(f"✅ Dataset split complete:")
    print(f"   Total examples: {n_total}")
    print(f"   Intent-attribute combinations: {n_combos}")
    print(f"   Train: {len(train_data)} examples ({len(train_data)/n_total*100:.1f}%)")
    print(f"   Val:   {len(val_data)} examples ({len(val_data)/n_total*100:.1f}%)")
    print(f"   Test:  {len(test_data)} examples ({len(test_data)/n_total*100:.1f}%)")
    print(f"\n   Split per combination: {train_per_combo} train, {val_per_combo} val, {test_per_combo} test")
    print(f"\n   Saved to:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")
    
    # Verify each combination appears in all splits
    print(f"\n   Verifying splits...")
    train_combos = set((ex["intent"], ex["slots"]["attribute"]) for ex in train_data)
    val_combos = set((ex["intent"], ex["slots"]["attribute"]) for ex in val_data)
    test_combos = set((ex["intent"], ex["slots"]["attribute"]) for ex in test_data)
    
    all_combos = set(grouped.keys())
    missing_train = all_combos - train_combos
    missing_val = all_combos - val_combos
    missing_test = all_combos - test_combos
    
    if missing_train or missing_val or missing_test:
        print(f"   ⚠️  Warning: Some combinations missing from splits!")
        if missing_train:
            print(f"      Missing from train: {missing_train}")
        if missing_val:
            print(f"      Missing from val: {missing_val}")
        if missing_test:
            print(f"      Missing from test: {missing_test}")
    else:
        print(f"   ✓ All {n_combos} combinations present in all splits!")
    
    return train_path, val_path, test_path

if __name__ == "__main__":
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "dataset")
    
    all_json_path = os.path.join(dataset_dir, "all.json")
    
    if not os.path.exists(all_json_path):
        print(f"❌ Error: {all_json_path} not found!")
        exit(1)
    
    preprocess_dataset(all_json_path, dataset_dir)

