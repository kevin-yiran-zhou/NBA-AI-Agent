"""
Train BERT model for intent and attribute classification.
Simplified version: only predicts intent and attribute (no input prediction).
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import os

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
def load_data(path):
    """Load data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    texts = [ex["text"] for ex in data]
    intents = [ex["intent"] for ex in data]
    attributes = [ex["slots"]["attribute"] for ex in data]
    return texts, intents, attributes

# ----------------------------------------------------
# 2. Dataset class
# ----------------------------------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, texts, intents, attrs, tokenizer, max_len=64):
        self.texts = texts
        self.intents = intents
        self.attrs = attrs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_label": torch.tensor(self.intents[idx], dtype=torch.long),
            "attr_label": torch.tensor(self.attrs[idx], dtype=torch.long)
        }

# ----------------------------------------------------
# 3. Multi-task model definition (Intent + Attribute only)
# ----------------------------------------------------
class BertForIntentAndAttr(nn.Module):
    def __init__(self, num_intents, num_attrs):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.attr_head = nn.Linear(hidden_size, num_attrs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, intent_labels=None, attr_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        intent_logits = self.intent_head(pooled)
        attr_logits = self.attr_head(pooled)

        loss = None
        if intent_labels is not None and attr_labels is not None:
            loss_intent = self.loss_fn(intent_logits, intent_labels)
            loss_attr = self.loss_fn(attr_logits, attr_labels)
            loss = loss_intent + loss_attr  # equal weighting
        # Return 4 values for compatibility with end_to_end.py (which expects input_logits as 3rd arg)
        return intent_logits, attr_logits, None, loss

# ----------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------
def evaluate(model, loader, device, intent_encoder, attr_encoder):
    model.eval()
    preds_intent, preds_attr, trues_intent, trues_attr = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out_intent, out_attr, _, _ = model(input_ids, mask)
            preds_intent.extend(out_intent.argmax(1).cpu().numpy())
            preds_attr.extend(out_attr.argmax(1).cpu().numpy())
            trues_intent.extend(batch["intent_label"].numpy())
            trues_attr.extend(batch["attr_label"].numpy())

    print("\n" + "="*60)
    print("Intent classification:")
    print("="*60)
    print(classification_report(trues_intent, preds_intent, target_names=intent_encoder.classes_, digits=4, zero_division=0))
    print("\n" + "="*60)
    print("Attribute classification:")
    print("="*60)
    print(classification_report(trues_attr, preds_attr, target_names=attr_encoder.classes_, digits=4, zero_division=0))

# ----------------------------------------------------
# Main training script
# ----------------------------------------------------
if __name__ == "__main__":
    # Set up paths - script is in code/bert/, dataset and models are at project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "dataset")
    models_dir = os.path.join(project_root, "models", "bert_multi")
    
    # Load data
    train_path = os.path.join(dataset_dir, "train.json")
    val_path = os.path.join(dataset_dir, "val.json")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ Error: train.json or val.json not found!")
        print("   Run preprocess_data.py first to create train/val/test splits.")
        exit(1)
    
    train_texts, train_intents, train_attrs = load_data(train_path)
    val_texts, val_intents, val_attrs = load_data(val_path)

    # Encode labels
    intent_encoder = LabelEncoder()
    attr_encoder = LabelEncoder()
    intent_encoder.fit(train_intents + val_intents)
    attr_encoder.fit(train_attrs + val_attrs)

    num_intents = len(intent_encoder.classes_)
    num_attrs = len(attr_encoder.classes_)

    y_intent_train = intent_encoder.transform(train_intents)
    y_intent_val = intent_encoder.transform(val_intents)
    y_attr_train = attr_encoder.transform(train_attrs)
    y_attr_val = attr_encoder.transform(val_attrs)

    print(f"✅ Loaded {len(train_texts)} train, {len(val_texts)} val samples")
    print(f"Intents: {list(intent_encoder.classes_)}")
    print(f"Attributes: {list(attr_encoder.classes_)}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Create datasets and loaders
    train_ds = MultiTaskDataset(train_texts, y_intent_train, y_attr_train, tokenizer)
    val_ds   = MultiTaskDataset(val_texts,   y_intent_val,   y_attr_val,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")
    model = BertForIntentAndAttr(num_intents, num_attrs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 15
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    print(f"\n📊 Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)
            attr_labels = batch["attr_label"].to(device)

            optimizer.zero_grad()
            _, _, _, loss = model(input_ids, mask, intent_labels, attr_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"\nEpoch {epoch+1} average loss: {total_loss/len(train_loader):.4f}")
        print("Val evaluation:")
        evaluate(model, val_loader, device, intent_encoder, attr_encoder)

    # Save model + encoders
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "model.pt"))
    tokenizer.save_pretrained(models_dir)
    np.save(os.path.join(models_dir, "intent_encoder.npy"), intent_encoder.classes_)
    np.save(os.path.join(models_dir, "attr_encoder.npy"), attr_encoder.classes_)

    print(f"\n✅ Model saved to {models_dir}/model.pt")

