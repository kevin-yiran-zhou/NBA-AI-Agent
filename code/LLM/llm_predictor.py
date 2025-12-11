from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json
from typing import Dict, Optional


class LLMPredictor:
    """LLM predictor using Qwen model."""
    
    VALID_INTENTS = ['player_info', 'team_info']
    VALID_ATTRIBUTES = [
        'abbreviation', 'city', 'college', 'conference', 'country', 'division',
        'draft_number', 'draft_round', 'draft_year', 'full_name', 'height',
        'jersey_number', 'position', 'team', 'weight'
    ]
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507-FP8", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📦 Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Model loaded successfully!")
    
    def predict(self, text: str) -> Dict:
        """Predict intent, attribute, and entity name."""
        import time
        
        # Group attributes by intent
        player_attrs = ['college', 'country', 'draft_number', 'draft_round', 'draft_year', 'height', 'jersey_number', 'position', 'team', 'weight']
        team_attrs = ['abbreviation', 'city', 'conference', 'division', 'full_name']
        
        prompt = f"""Question: {text}
        
        Classify this NBA question by identifying intent, attribute, and entity.

        CRITICAL RULES - MUST FOLLOW:
        1. If the question asks about a PLAYER (person's full name), intent MUST be "player_info" and attribute MUST be one of: {', '.join(player_attrs)}
        2. If the question asks about a TEAM (team name), intent MUST be "team_info" and attribute MUST be one of: {', '.join(team_attrs)}
        3. These combinations are FIXED - you CANNOT mix them:
        - player_info can ONLY use: {', '.join(player_attrs)}
        - team_info can ONLY use: {', '.join(team_attrs)}
        4. Extract the player or team name mentioned in the question (or "Unknown" if you cannot find the name)

        Analyze the question carefully. Determine if it's about a player or team, then select the matching attribute from the correct list.

        Respond ONLY with valid JSON: {{"intent": "...", "attribute": "...", "entity": "..."}}

        Response:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, temperature=0.1, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        llm_ms = (time.perf_counter() - t0) * 1000.0
        
        input_length = inputs['input_ids'].shape[1]
        raw_response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Parse JSON response
        intent, attr, entity = self.VALID_INTENTS[0], self.VALID_ATTRIBUTES[0], "Unknown"
        player_attrs = ['college', 'country', 'draft_number', 'draft_round', 'draft_year', 'height', 'jersey_number', 'position', 'team', 'weight']
        team_attrs = ['abbreviation', 'city', 'conference', 'division', 'full_name']
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                intent_raw = parsed.get('intent', '').strip().lower()
                attr_raw = parsed.get('attribute', '').strip().lower()
                entity = parsed.get('entity', '').strip()
                
                # Match intent
                for valid in self.VALID_INTENTS:
                    if valid.lower() == intent_raw:
                        intent = valid
                        break
                
                # Match attribute
                for valid in self.VALID_ATTRIBUTES:
                    if valid.lower() == attr_raw:
                        attr = valid
                        break
                
                # Validate combination
                if intent == 'player_info' and attr not in player_attrs:
                    attr = player_attrs[0]
                elif intent == 'team_info' and attr not in team_attrs:
                    attr = team_attrs[0]
                    
            except (json.JSONDecodeError, AttributeError):
                pass
        
        if not entity or entity.lower() == 'unknown' or '<name>' in entity:
            entity = 'Unknown'
        
        return {
            'intent': intent,
            'attr': attr,
            'input': entity,
            'llm_ms': llm_ms,
            'raw_response': raw_response
        }
