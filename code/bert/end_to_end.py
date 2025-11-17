"""
End-to-End Agent Module

This module integrates model prediction, entity linking, API calls, and result formatting
into a complete pipeline.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
from transformers import BertTokenizer

from api_service import NBAApiService
from entity_linker import EntityLinker
from api_router import APIRouter
from response_formatter import ResponseFormatter
from train_bert import BertForIntentAndAttr
from mock_predictor import MockPredictor


class EndToEndAgent:
    """
    Complete end-to-end agent that processes natural language queries
    and returns formatted responses.
    """
    
    def __init__(self, model=None, tokenizer=None, intent_labels=None, 
                 attr_labels=None, input_labels=None,
                 api_service: Optional[NBAApiService] = None,
                 entity_linker: Optional[EntityLinker] = None,
                 api_router: Optional[APIRouter] = None,
                 response_formatter: Optional[ResponseFormatter] = None,
                 mock_predictor: Optional[MockPredictor] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the end-to-end agent.
        
        Can be initialized in two modes:
        1. With trained model: Provide model, tokenizer, and labels
        2. With mock predictor: Provide mock_predictor (for testing without trained model)
        
        Args:
            model: The trained BERT model (BertForIntentAndAttr) - optional if using mock_predictor
            tokenizer: The BERT tokenizer - optional if using mock_predictor
            intent_labels: Array of intent labels (from LabelEncoder) - optional if using mock_predictor
            attr_labels: Array of attribute labels (from LabelEncoder) - optional if using mock_predictor
            input_labels: Array of input labels (from LabelEncoder) - optional if using mock_predictor
            api_service: Instance of NBAApiService (will create if None)
            entity_linker: Instance of EntityLinker (will create if None)
            api_router: Instance of APIRouter (will create if None)
            response_formatter: Instance of ResponseFormatter (will create if None)
            mock_predictor: MockPredictor instance (for testing without model)
            device: PyTorch device (default: auto-detect)
        """
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
            # Use mock predictor (for testing without trained model)
            self.use_mock = True
            self.mock_predictor = mock_predictor
            self.model = None
            self.tokenizer = None
            self.intent_labels = None
            self.attr_labels = None
            self.input_labels = None
            self.device = None
        elif model is not None and tokenizer is not None:
            # Use trained model
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
        else:
            raise ValueError(
                "Must provide either (model, tokenizer, labels) or mock_predictor. "
                "Use EndToEndAgent.from_model_dir() to load from saved model, "
                "or EndToEndAgent.with_mock_predictor() to use mock predictor."
            )
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Complete end-to-end processing pipeline.
        
        Steps:
        1. Model prediction (intent + slots)
        2. Entity Linking (link entity names to IDs)
        3. API call (via APIRouter)
        4. Result formatting (natural language response)
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            Dictionary containing:
                - "intent": Predicted intent
                - "slots": Predicted slots
                - "linked_slots": Slots with entity IDs (if applicable)
                - "api_result": API call result
                - "formatted_response": Natural language response
        """
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
    
    def predict(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use the model or mock predictor to predict intent and slots from text.
        
        Args:
            text: Input text query
            
        Returns:
            Tuple of (intent, slots_dict)
            slots_dict format: {"input": ..., "attribute": ...}
        """
        if self.use_mock:
            # Use mock predictor
            return self.mock_predictor.predict(text)
        else:
            # Use trained model
            # Tokenize input
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                intent_logits, attr_logits, input_logits, _ = self.model(
                    enc["input_ids"],
                    enc["attention_mask"]
                )
            
            # Decode predictions
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            attr_idx = torch.argmax(attr_logits, dim=1).item()
            input_idx = torch.argmax(input_logits, dim=1).item()
            
            intent = self.intent_labels[intent_idx]
            attr = self.attr_labels[attr_idx]
            input_val = self.input_labels[input_idx]
            
            # Construct slots dictionary
            slots = {
                "input": input_val,
                "attribute": attr
            }
            
            return intent, slots
    
    def link_entities(self, intent: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Link entity names in slots to their API IDs.
        
        Currently, entity linking is handled by APIRouter internally,
        but this method can be used for preprocessing if needed.
        
        Args:
            intent: The predicted intent
            slots: The predicted slots
            
        Returns:
            Linked slots (currently just returns original slots,
            as linking happens in APIRouter)
        """
        # For now, entity linking is handled inside APIRouter
        # This method can be extended to pre-link entities if needed
        return slots.copy()
    
    def format_response(self, intent: str, api_result: Dict[str, Any],
                       slots: Optional[Dict[str, Any]] = None) -> str:
        """
        Format API result into natural language response.
        
        Args:
            intent: The predicted intent
            api_result: The API call result
            slots: Original slots (for context)
            
        Returns:
            Formatted natural language response
        """
        return self.response_formatter.format(intent, api_result, slots)
    
    @classmethod
    def from_model_dir(cls, model_dir: str, api_service: Optional[NBAApiService] = None,
                       entity_linker: Optional[EntityLinker] = None,
                       api_router: Optional[APIRouter] = None,
                       response_formatter: Optional[ResponseFormatter] = None,
                       device: Optional[torch.device] = None) -> 'EndToEndAgent':
        """
        Create an EndToEndAgent from a saved model directory.
        
        Args:
            model_dir: Path to the model directory (e.g., "models/bert_multi")
            api_service: Optional NBAApiService instance (creates new if None)
            entity_linker: Optional EntityLinker instance (creates new if None)
            api_router: Optional APIRouter instance (creates new if None)
            response_formatter: Optional ResponseFormatter instance (creates new if None)
            device: PyTorch device (default: auto-detect)
            
        Returns:
            Initialized EndToEndAgent instance
        """
        # Load tokenizer and labels
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        intent_labels = np.load(f"{model_dir}/intent_encoder.npy", allow_pickle=True)
        attr_labels = np.load(f"{model_dir}/attr_encoder.npy", allow_pickle=True)
        input_labels = np.load(f"{model_dir}/input_encoder.npy", allow_pickle=True)
        
        # Load model
        model = BertForIntentAndAttr(
            num_intents=len(intent_labels),
            num_attrs=len(attr_labels),
            num_inputs=len(input_labels)
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
            input_labels=input_labels,
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
        """
        Create an EndToEndAgent using a mock predictor (for testing without trained model).
        
        This is useful for testing the API integration before the model is trained.
        
        Args:
            api_service: Optional NBAApiService instance (creates new if None)
            entity_linker: Optional EntityLinker instance (creates new if None)
            api_router: Optional APIRouter instance (creates new if None)
            response_formatter: Optional ResponseFormatter instance (creates new if None)
            
        Returns:
            Initialized EndToEndAgent instance with mock predictor
        """
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

