from transformers import BertTokenizer
import torch, numpy as np
import time
import spacy
from train_bert import BertForIntentAndAttr  # import your class

model_dir = "models/bert_multi"
tokenizer = BertTokenizer.from_pretrained(model_dir)
intent_labels = np.load(f"{model_dir}/intent_encoder.npy", allow_pickle=True)
attr_labels = np.load(f"{model_dir}/attr_encoder.npy", allow_pickle=True)

model = BertForIntentAndAttr(num_intents=len(intent_labels),
                             num_attrs=len(attr_labels))
model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load spaCy NER once
nlp = spacy.load("en_core_web_trf")

def extract_entity_spacy(text, intent):
    t0 = time.perf_counter()
    doc = nlp(text)
    if "player" in str(intent).lower():
        ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    elif "team" in str(intent).lower():
        ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
    else:
        ents = [ent.text for ent in doc.ents]
    spacy_ms = (time.perf_counter() - t0) * 1000.0
    return (ents[0] if ents else "Unknown"), spacy_ms

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        intent_logits, attr_logits, _ = model(enc["input_ids"], enc["attention_mask"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    bert_ms = (time.perf_counter() - t0) * 1000.0
    intent_idx = torch.argmax(intent_logits, dim=1).item()
    attr_idx = torch.argmax(attr_logits, dim=1).item()
    intent = intent_labels[intent_idx]
    attr = attr_labels[attr_idx]
    extracted_input, spacy_ms = extract_entity_spacy(text, intent)
    return intent, attr, extracted_input, bert_ms, spacy_ms

while True:
    text = input("🗣️ You: ").strip()
    if text.lower() in ["exit", "quit"]: break
    intent, attr, extracted_input, bert_ms, spacy_ms = predict(text)
    print(f"🤖 Intent: {intent} | Attribute: {attr} | Input: {extracted_input}")
    print(f"⏱️ BERT: {bert_ms:.2f} ms | spaCy: {spacy_ms:.2f} ms\n")
