from transformers import BertTokenizer
import torch
import numpy as np
import time
import spacy
try:
    from .train_bert import BertForIntentAndAttr
except ImportError:
    from train_bert import BertForIntentAndAttr


class BertPredictor:
    """BERT predictor for intent and attribute classification."""
    
    def __init__(self, model_dir="../../models/bert_multi", device=None):
        self.model_dir = model_dir
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and label encoders
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.intent_labels = np.load(f"{model_dir}/intent_encoder.npy", allow_pickle=True)
        self.attr_labels = np.load(f"{model_dir}/attr_encoder.npy", allow_pickle=True)
        
        # Load model
        self.model = BertForIntentAndAttr(
            num_intents=len(self.intent_labels),
            num_attrs=len(self.attr_labels)
        )
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
        self.model.eval()
        self.model.to(self.device)
        
        # Load spaCy NER for entity extraction and preprocessing
        self.nlp = spacy.load("en_core_web_trf")
    
    def preprocess_text(self, text: str) -> str:
        """Replace entities with <name> placeholder."""
        doc = self.nlp(text)
        processed_text = text
        
        # Replace entities with <name> (in reverse order to preserve indices)
        entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
        for ent in entities:
            # Replace entity with <name>
            processed_text = processed_text[:ent.start_char] + "<name>" + processed_text[ent.end_char:]
        
        return processed_text
    
    def extract_entity_spacy(self, text: str, intent: str) -> tuple:
        """Extract entity using spaCy NER."""
        t0 = time.perf_counter()
        doc = self.nlp(text)
        if "player" in str(intent).lower():
            ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        elif "team" in str(intent).lower():
            ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
        else:
            ents = [ent.text for ent in doc.ents]
        spacy_ms = (time.perf_counter() - t0) * 1000.0
        return (ents[0] if ents else "Unknown"), spacy_ms
    
    def predict(self, text: str, extract_entity: bool = True, preprocess: bool = True):
        """Predict intent and attribute for given text."""
        # Preprocess text: replace entities with <name>
        if preprocess:
            processed_text = self.preprocess_text(text)
        else:
            processed_text = text
        
        # Tokenize and encode
        enc = self.tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(self.device)
        
        # BERT inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            intent_logits, attr_logits, _, _ = self.model(enc["input_ids"], enc["attention_mask"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bert_ms = (time.perf_counter() - t0) * 1000.0
        
        # Get predictions
        intent_idx = torch.argmax(intent_logits, dim=1).item()
        attr_idx = torch.argmax(attr_logits, dim=1).item()
        intent = self.intent_labels[intent_idx]
        attr = self.attr_labels[attr_idx]
        
        result = {
            'intent': intent,
            'attr': attr,
            'bert_ms': bert_ms
        }
        
        # Extract entity if requested (use original text, not preprocessed)
        if extract_entity:
            extracted_input, spacy_ms = self.extract_entity_spacy(text, intent)
            result['input'] = extracted_input
            result['spacy_ms'] = spacy_ms
        else:
            result['spacy_ms'] = 0.0
        
        return result

