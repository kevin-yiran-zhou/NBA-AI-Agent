from transformers import BertTokenizer
import torch
import numpy as np
import time
import spacy
from train_bert import BertForIntentAndAttr


class BertPredictor:
    """BERT-based predictor for intent and attribute classification with spaCy entity extraction."""
    
    def __init__(self, model_dir="../../models/bert_multi", device=None):
        """
        Initialize the BERT predictor.
        
        Args:
            model_dir: Path to the model directory containing model.pt and encoders
            device: torch device (auto-detected if None)
        """
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
        
        # Load spaCy NER
        self.nlp = spacy.load("en_core_web_trf")
    
    def extract_entity_spacy(self, text, intent):
        """
        Extract entity from text using spaCy NER based on intent.
        
        Args:
            text: Input text
            intent: Predicted intent (used to filter entity types)
            
        Returns:
            tuple: (entity_text, time_ms)
        """
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
    
    def predict(self, text, extract_entity=True):
        """
        Predict intent and attribute for given text.
        
        Args:
            text: Input text to predict
            extract_entity: Whether to extract entity using spaCy (default: True)
            
        Returns:
            dict: {
                'intent': predicted intent,
                'attr': predicted attribute,
                'input': extracted entity (if extract_entity=True),
                'bert_ms': BERT inference time in milliseconds,
                'spacy_ms': spaCy inference time in milliseconds (if extract_entity=True)
            }
        """
        # Tokenize and encode
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        # BERT inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            intent_logits, attr_logits, _ = self.model(enc["input_ids"], enc["attention_mask"])
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
        
        # Extract entity if requested
        if extract_entity:
            extracted_input, spacy_ms = self.extract_entity_spacy(text, intent)
            result['input'] = extracted_input
            result['spacy_ms'] = spacy_ms
        else:
            result['spacy_ms'] = 0.0
        
        return result

