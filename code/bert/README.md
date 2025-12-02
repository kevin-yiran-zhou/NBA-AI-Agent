# BERT Training and Inference

This module contains code for training and using a BERT model for intent and attribute classification.

## Files

- `preprocess_data.py` - Split dataset into train/val/test (0.8/0.1/0.1)
- `train_bert.py` - Train BERT model for intent + attribute classification
- `bert.py` - BERT predictor class for inference
- `test_bert.py` - Evaluate model on test set

## Usage

### 1. Preprocess Dataset
```bash
cd code/bert
python preprocess_data.py
```
This splits `dataset/all.json` into `train.json`, `val.json`, and `test.json`.

### 2. Train Model
```bash
python train_bert.py
```
This will:
- Load train/val splits
- Train BERT model for 3 epochs
- Save model to `models/bert_multi/`

### 3. Test Model
```bash
python test_bert.py
```
This evaluates the model on the test set and prints classification reports.

### 4. Use Model in Code
```python
from bert import BertPredictor

# Initialize predictor
predictor = BertPredictor(model_dir="../../models/bert_multi")

# Predict on real query (with real entity names)
result = predictor.predict("What position does Stephen Curry play?")
print(result)
# {'intent': 'player_info', 'attr': 'position', 'input': 'Stephen Curry', ...}

# Predict on test data (already has <name> placeholder)
result = predictor.predict("What position does <name> play?", preprocess=False)
```

## Model Architecture

- **Input**: Text with `<name>` placeholder (or real text that gets preprocessed)
- **Output**: 
  - Intent: `team_info` or `player_info`
  - Attribute: One of 15 attributes (conference, division, city, etc.)

The model uses BERT-base-uncased with two classification heads (intent + attribute).

