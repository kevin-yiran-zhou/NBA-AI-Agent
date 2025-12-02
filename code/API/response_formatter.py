"""
Response Formatter Module

This module formats API results into user-friendly natural language responses.
"""

from typing import Dict, Any, Optional


class ResponseFormatter:
    """
    Formats API results into natural language responses.
    """
    
    def format(self, intent: str, api_result: Dict[str, Any], slots: Optional[Dict[str, Any]] = None) -> str:
        """
        Format API result based on intent.
        
        Args:
            intent: The intent type
            api_result: The API result dictionary (from APIRouter)
            slots: The original slots (optional, for context)
            
        Returns:
            Formatted natural language response
        """
        if not api_result.get("success", False):
            error = api_result.get("error", "Unknown error")
            return f"I'm sorry, I couldn't find that information. {error}"
        
        data = api_result.get("data", {})
        
        # Route to appropriate formatter based on intent
        if intent == "team_info":
            return self.format_team_info(data, slots)
        elif intent == "player_info":
            return self.format_player_info(data, slots)
        elif intent == "game_lookup":
            return self.format_game_info(data)
        else:
            # Generic formatting for other intents
            return self.format_generic(intent, data)
    
    def format_team_info(self, data: Dict[str, Any], slots: Optional[Dict[str, Any]] = None) -> str:
        """
        Format team information response.
        
        Args:
            data: API result data containing team_id, team_name, attribute, value
            slots: Original slots (optional)
            
        Returns:
            Formatted response string
        """
        team_name = data.get("team_name", "the team")
        attribute = data.get("attribute", "")
        value = data.get("value", "")
        
        # Format based on attribute type
        if attribute == "conference":
            return f"{team_name} are in the {value}ern Conference."
        elif attribute == "division":
            return f"{team_name} are in the {value} Division."
        elif attribute == "city":
            return f"{team_name} are based in {value}."
        elif attribute == "full_name":
            return f"The full name is {value}."
        elif attribute == "abbreviation":
            return f"{team_name}'s abbreviation is {value}."
        else:
            return f"{team_name}'s {attribute} is {value}."
    
    def format_player_info(self, data: Dict[str, Any], slots: Optional[Dict[str, Any]] = None) -> str:
        """
        Format player information response.
        
        Args:
            data: API result data containing player_id, player_name, attribute, value
            slots: Original slots (optional)
            
        Returns:
            Formatted response string
        """
        player_name = data.get("player_name", "the player")
        attribute = data.get("attribute", "")
        value = data.get("value", "")
        
        # Handle None values
        if value is None:
            return f"I don't have information about {player_name}'s {attribute}."
        
        # Format based on attribute type
        if attribute == "position":
            return f"{player_name} plays the {value} position."
        elif attribute == "height":
            return f"{player_name} is {value} tall."
        elif attribute == "weight":
            return f"{player_name} weighs {value} pounds."
        elif attribute == "jersey_number":
            return f"{player_name} wears jersey number {value}."
        elif attribute == "college":
            if value:
                return f"{player_name} attended {value}."
            else:
                return f"{player_name} did not attend college."
        elif attribute == "country":
            return f"{player_name} is from {value}."
        elif attribute == "draft_year":
            if value:
                return f"{player_name} was drafted in {int(value)}."
            else:
                return f"{player_name} was not drafted."
        elif attribute == "draft_round":
            if value:
                return f"{player_name} was drafted in round {int(value)}."
            else:
                return f"{player_name} was not drafted."
        elif attribute == "draft_number":
            if value:
                return f"{player_name} was drafted as the {int(value)} pick."
            else:
                return f"{player_name} was not drafted."
        elif attribute == "team":
            # Handle team attribute - value is a team object
            if isinstance(value, dict):
                team_full_name = value.get("full_name", value.get("name", "Unknown"))
                return f"{player_name}'s team is {team_full_name}."
            elif value:
                return f"{player_name}'s team is {value}."
            else:
                return f"{player_name} is not currently on a team."
        else:
            return f"{player_name}'s {attribute} is {value}."
    
    def format_game_info(self, data: Dict[str, Any]) -> str:
        """
        Format game information response.
        
        Args:
            data: API result data containing game information
            
        Returns:
            Formatted response string
        """
        # If it's a list of games
        if "games" in data:
            count = data.get("count", 0)
            date = data.get("date", "")
            if count == 0:
                return f"No games were played on {date}."
            elif count == 1:
                game = data["games"][0]
                return self._format_single_game(game)
            else:
                games = data["games"]
                response = f"There were {count} games on {date}:\n"
                for i, game in enumerate(games[:5], 1):  # Limit to 5 games
                    home = game.get("home_team", {}).get("full_name", "Unknown")
                    visitor = game.get("visitor_team", {}).get("full_name", "Unknown")
                    home_score = game.get("home_team_score", 0)
                    visitor_score = game.get("visitor_team_score", 0)
                    response += f"{i}. {visitor} {visitor_score} - {home_score} {home}\n"
                if count > 5:
                    response += f"... and {count - 5} more games."
                return response
        else:
            # Single game
            return self._format_single_game(data)
    
    def _format_single_game(self, game: Dict[str, Any]) -> str:
        """Format a single game's information."""
        home_team = game.get("home_team", {}).get("full_name", "Unknown")
        visitor_team = game.get("visitor_team", {}).get("full_name", "Unknown")
        home_score = game.get("home_team_score", 0)
        visitor_score = game.get("visitor_team_score", 0)
        status = game.get("status", "Unknown")
        date = game.get("date", "")
        
        if status == "Final":
            return f"On {date}, {visitor_team} {visitor_score} - {home_score} {home_team} (Final)."
        else:
            return f"On {date}, {visitor_team} vs {home_team} - Status: {status}."
    
    def format_generic(self, intent: str, data: Dict[str, Any]) -> str:
        """
        Generic formatter for other intents.
        
        Args:
            intent: The intent type
            data: API result data
            
        Returns:
            Formatted response string
        """
        # For intents that are not yet fully implemented
        if isinstance(data, dict) and "value" in data:
            return f"The result is: {data['value']}"
        else:
            return f"Here's the information: {str(data)}"

