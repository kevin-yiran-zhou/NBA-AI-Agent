from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
import time
from transformers import BertTokenizer
import spacy

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from API.api_service import NBAApiService
from API.entity_linker import EntityLinker
from API.api_router import APIRouter
from API.response_formatter import ResponseFormatter
from bert.train_bert import BertForIntentAndAttr
from mock.mock_predictor import MockPredictor


class EndToEndAgent:
    """End-to-end agent for processing natural language queries."""
    
    def __init__(self, model=None, tokenizer=None, intent_labels=None, 
                 attr_labels=None, input_labels=None,  # input_labels kept for compatibility but not used
                 api_service: Optional[NBAApiService] = None,
                 entity_linker: Optional[EntityLinker] = None,
                 api_router: Optional[APIRouter] = None,
                 response_formatter: Optional[ResponseFormatter] = None,
                 mock_predictor: Optional[MockPredictor] = None,
                 device: Optional[torch.device] = None):
        # Initialize services (create if not provided)
        if api_service is None:
            api_service = NBAApiService()
        if entity_linker is None:
            entity_linker = EntityLinker(api_service)
        if api_router is None:
            api_router = APIRouter(api_service, entity_linker)
        if response_formatter is None:
            response_formatter = ResponseFormatter()
        
        self.api_service = api_service
        self.entity_linker = entity_linker
        self.api_router = api_router
        self.response_formatter = response_formatter
        
        # Determine prediction mode
        if mock_predictor is not None:
            self.use_mock = True
            self.mock_predictor = mock_predictor
            self.model = None
            self.tokenizer = None
            self.intent_labels = None
            self.attr_labels = None
            self.input_labels = None
            self.device = None
        elif model is not None and tokenizer is not None:
            self.use_mock = False
            self.mock_predictor = None
            self.model = model
            self.tokenizer = tokenizer
            self.intent_labels = intent_labels
            self.attr_labels = attr_labels
            self.input_labels = input_labels
            
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
            
            self.model.to(self.device)
            self.model.eval()
            
            self.nlp = spacy.load("en_core_web_trf")
        else:
            raise ValueError(
                "Must provide either (model, tokenizer, labels) or mock_predictor. "
                "Use EndToEndAgent.from_model_dir() to load from saved model, "
                "or EndToEndAgent.with_mock_predictor() to use mock predictor."
            )
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process query through end-to-end pipeline."""
        # Step 1: Model prediction
        intent, slots = self.predict(user_query)
        
        # Step 2: Entity Linking (if needed)
        linked_slots = self.link_entities(intent, slots)
        
        # Step 3: API call
        api_result = self.api_router.route(intent, linked_slots)
        
        # Step 4: Format response
        formatted_response = self.format_response(intent, api_result, slots)
        
        return {
            "intent": intent,
            "slots": slots,
            "linked_slots": linked_slots,
            "api_result": api_result,
            "formatted_response": formatted_response
        }
    
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
    
    def extract_entity_spacy(self, text: str, intent: str) -> str:
        """Extract entity using spaCy NER."""
        doc = self.nlp(text)
        if "player" in str(intent).lower():
            ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        elif "team" in str(intent).lower():
            ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
        else:
            ents = [ent.text for ent in doc.ents]
        return ents[0] if ents else "Unknown"
    
    def predict(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Predict intent and slots from text."""
        if self.use_mock:
            return self.mock_predictor.predict(text)
        else:
            processed_text = self.preprocess_text(text)
            enc = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            ).to(self.device)
            
            with torch.no_grad():
                intent_logits, attr_logits, _, _ = self.model(
                    enc["input_ids"],
                    enc["attention_mask"]
                )
            
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            attr_idx = torch.argmax(attr_logits, dim=1).item()
            
            intent = self.intent_labels[intent_idx]
            attr = self.attr_labels[attr_idx]
            input_val = self.extract_entity_spacy(text, intent)
            slots = {
                "input": input_val,
                "attribute": attr
            }
            
            return intent, slots
    
    def link_entities(self, intent: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Link entity names in slots to their API IDs."""
        return slots.copy()
    
    def format_response(self, intent: str, api_result: Dict[str, Any],
                       slots: Optional[Dict[str, Any]] = None) -> str:
        """Format API result into natural language response."""
        return self.response_formatter.format(intent, api_result, slots)
    
    @classmethod
    def from_model_dir(cls, model_dir: str, api_service: Optional[NBAApiService] = None,
                       entity_linker: Optional[EntityLinker] = None,
                       api_router: Optional[APIRouter] = None,
                       response_formatter: Optional[ResponseFormatter] = None,
                       device: Optional[torch.device] = None) -> 'EndToEndAgent':
        """Create an EndToEndAgent from a saved model directory."""
        # Load tokenizer and labels
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        intent_labels = np.load(f"{model_dir}/intent_encoder.npy", allow_pickle=True)
        attr_labels = np.load(f"{model_dir}/attr_encoder.npy", allow_pickle=True)
        
        model = BertForIntentAndAttr(
            num_intents=len(intent_labels),
            num_attrs=len(attr_labels)
        )
        model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
        model.eval()
        
        # Initialize services if not provided
        if api_service is None:
            api_service = NBAApiService()
        if entity_linker is None:
            entity_linker = EntityLinker(api_service)
        if api_router is None:
            api_router = APIRouter(api_service, entity_linker)
        if response_formatter is None:
            response_formatter = ResponseFormatter()
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            intent_labels=intent_labels,
            attr_labels=attr_labels,
            input_labels=None,
            api_service=api_service,
            entity_linker=entity_linker,
            api_router=api_router,
            response_formatter=response_formatter,
            device=device
        )
    
    @classmethod
    def with_mock_predictor(cls, api_service: Optional[NBAApiService] = None,
                           entity_linker: Optional[EntityLinker] = None,
                           api_router: Optional[APIRouter] = None,
                           response_formatter: Optional[ResponseFormatter] = None) -> 'EndToEndAgent':
        """Create an EndToEndAgent using a mock predictor."""
        mock_predictor = MockPredictor()
        
        # Initialize services if not provided
        if api_service is None:
            api_service = NBAApiService()
        if entity_linker is None:
            entity_linker = EntityLinker(api_service)
        if api_router is None:
            api_router = APIRouter(api_service, entity_linker)
        if response_formatter is None:
            response_formatter = ResponseFormatter()
        
        return cls(
            model=None,
            tokenizer=None,
            intent_labels=None,
            attr_labels=None,
            input_labels=None,
            api_service=api_service,
            entity_linker=entity_linker,
            api_router=api_router,
            response_formatter=response_formatter,
            mock_predictor=mock_predictor
        )

