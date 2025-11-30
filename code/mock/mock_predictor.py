"""
Mock Predictor Module

This module provides a mock predictor that can be used to test the API integration
without requiring a trained model. It uses simple rule-based matching to predict
intent and slots.
"""

import re
from typing import Dict, Any, Tuple


class MockPredictor:
    """
    Mock predictor that uses simple rule-based matching to predict intent and slots.
    This allows testing the API integration without a trained model.
    """
    
    # Team keywords
    TEAM_KEYWORDS = [
        "lakers", "warriors", "celtics", "heat", "bulls", "knicks", "rockets",
        "nuggets", "clippers", "mavericks", "nets", "pistons", "pacers",
        "hornets", "cavaliers", "grizzlies", "jazz", "suns", "trail blazers",
        "raptors", "wizards", "thunder", "magic", "pelicans", "spurs", "kings",
        "76ers", "timberwolves", "hawks", "bucks"
    ]
    
    # Player keywords (common players)
    PLAYER_KEYWORDS = [
        "curry", "james", "durant", "embiid", "doncic", "antetokounmpo",
        "jokic", "tatum", "booker", "butler", "irving", "davis", "lillard",
        "morant", "young", "mitchell", "williamson", "brown", "brunson"
    ]
    
    # Intent patterns
    TEAM_INFO_PATTERNS = [
        r"which (conference|division|city|abbreviation)",
        r"what.*conference|division|city|abbreviation",
        r"tell me.*(conference|division|city|abbreviation|full name)",
        r"what is.*(conference|division|city|abbreviation|full name)"
    ]
    
    PLAYER_INFO_PATTERNS = [
        r"what (position|height|weight|jersey|college|country)",
        r"how (tall|heavy|much)",
        r"which (position|college|country)",
        r"tell me.*(position|height|weight|jersey|college|country)"
    ]
    
    # Attribute patterns
    ATTRIBUTE_MAP = {
        "conference": ["conference"],
        "division": ["division"],
        "city": ["city", "based", "located"],
        "full_name": ["full name", "fullname"],
        "abbreviation": ["abbreviation", "abbr", "code", "three letter"],
        "position": ["position", "plays"],
        "height": ["height", "tall"],
        "weight": ["weight", "weigh", "heavy"],
        "jersey_number": ["jersey", "number", "jersey number"],
        "college": ["college", "attended", "university"],
        "country": ["country", "from", "represent"]
    }
    
    def predict(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Predict intent and slots from text using simple rule-based matching.
        
        Args:
            text: Input text query
            
        Returns:
            Tuple of (intent, slots_dict)
            slots_dict format: {"input": ..., "attribute": ...}
        """
        text_lower = text.lower()
        
        # Determine intent
        intent = self._predict_intent(text_lower)
        
        # Extract attribute
        attribute = self._extract_attribute(text_lower)
        
        # Extract input (team or player name)
        input_val = self._extract_input(text_lower, intent)
        
        slots = {
            "input": input_val,
            "attribute": attribute
        }
        
        return intent, slots
    
    def _predict_intent(self, text: str) -> str:
        """Predict intent from text."""
        # Check for team info patterns
        for pattern in self.TEAM_INFO_PATTERNS:
            if re.search(pattern, text):
                return "team_info"
        
        # Check for player info patterns
        for pattern in self.PLAYER_INFO_PATTERNS:
            if re.search(pattern, text):
                return "player_info"
        
        # Default: try to determine by entity type
        for team in self.TEAM_KEYWORDS:
            if team in text:
                return "team_info"
        
        for player in self.PLAYER_KEYWORDS:
            if player in text:
                return "player_info"
        
        # Default to team_info
        return "team_info"
    
    def _extract_attribute(self, text: str) -> str:
        """Extract attribute from text."""
        for attr, keywords in self.ATTRIBUTE_MAP.items():
            for keyword in keywords:
                if keyword in text:
                    return attr
        
        # Default attributes
        if "team" in text or any(team in text for team in self.TEAM_KEYWORDS):
            return "conference"  # Default team attribute
        else:
            return "position"  # Default player attribute
    
    def _extract_input(self, text: str, intent: str) -> str:
        """Extract team or player name from text."""
        # Try to find team names
        if intent == "team_info":
            for team in self.TEAM_KEYWORDS:
                if team in text:
                    # Capitalize properly
                    if team == "trail blazers":
                        return "Trail Blazers"
                    elif team == "76ers":
                        return "76ers"
                    else:
                        return team.title()
            
            # Try common aliases
            if "laker" in text:
                return "Lakers"
            elif "warrior" in text:
                return "Warriors"
            elif "celtic" in text:
                return "Celtics"
        
        # Try to find player names
        elif intent == "player_info":
            # Common player name patterns
            player_patterns = [
                (r"stephen curry|steph curry", "Stephen Curry"),
                (r"lebron james|lebron", "LeBron James"),
                (r"kevin durant|durant", "Kevin Durant"),
                (r"joel embiid|embiid", "Joel Embiid"),
                (r"luka doncic|doncic", "Luka Doncic"),
                (r"giannis|antetokounmpo", "Giannis Antetokounmpo"),
                (r"nikola jokic|jokic", "Nikola Jokic"),
                (r"jayson tatum|tatum", "Jayson Tatum"),
                (r"devin booker|booker", "Devin Booker"),
                (r"jimmy butler|butler", "Jimmy Butler"),
            ]
            
            for pattern, name in player_patterns:
                if re.search(pattern, text):
                    return name
            
            # Try simple last name matching
            for player in self.PLAYER_KEYWORDS:
                if player in text:
                    # Try to find full name
                    for pattern, name in player_patterns:
                        if player in pattern:
                            return name
                    # Fallback: capitalize last name
                    return player.title()
        
        # Default fallback
        return "Unknown"
    
    def predict_with_confidence(self, text: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Predict with confidence score (for compatibility with real model).
        
        Args:
            text: Input text query
            
        Returns:
            Tuple of (intent, slots_dict, confidence)
        """
        intent, slots = self.predict(text)
        # Mock confidence (always high for rule-based)
        confidence = 0.9
        return intent, slots, confidence

