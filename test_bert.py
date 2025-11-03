from transformers import BertTokenizer
import torch, numpy as np
from train_bert import BertForIntentAndAttr  # import your class

model_dir = "models/bert_multi"
tokenizer = BertTokenizer.from_pretrained(model_dir)
intent_labels = np.load(f"{model_dir}/intent_encoder.npy", allow_pickle=True)
attr_labels = np.load(f"{model_dir}/attr_encoder.npy", allow_pickle=True)
input_labels = np.load(f"{model_dir}/input_encoder.npy", allow_pickle=True)

model = BertForIntentAndAttr(num_intents=len(intent_labels),
                             num_attrs=len(attr_labels),
                             num_inputs=len(input_labels))
model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        intent_logits, attr_logits, input_logits, _ = model(enc["input_ids"], enc["attention_mask"])
    intent_idx = torch.argmax(intent_logits, dim=1).item()
    attr_idx = torch.argmax(attr_logits, dim=1).item()
    input_idx = torch.argmax(input_logits, dim=1).item()
    intent = intent_labels[intent_idx]
    attr = attr_labels[attr_idx]
    input_val = input_labels[input_idx]
    return intent, attr, input_val

while True:
    text = input("🗣️ You: ").strip()
    if text.lower() in ["exit", "quit"]: break
    intent, attr, input_val = predict(text)
    print(f"🤖 Intent: {intent} | Attribute: {attr} | Input: {input_val}\n")
