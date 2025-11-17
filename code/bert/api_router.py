"""
API Router Module

This module routes intent and slots to the appropriate API calls.
It acts as a bridge between the NLP model predictions and the API service.
"""

from typing import Optional, Dict, Any, List
from api_service import NBAApiService
from entity_linker import EntityLinker


class APIRouter:
    """
    Routes intent and slots to appropriate API calls.
    
    This class takes the predicted intent and slots from the NLP model,
    uses EntityLinker to resolve entity names to IDs, and then calls
    the appropriate API methods.
    """
    
    def __init__(self, api_service: NBAApiService, entity_linker: EntityLinker):
        """
        Initialize the API router.
        
        Args:
            api_service: Instance of NBAApiService for API calls
            entity_linker: Instance of EntityLinker for entity resolution
        """
        self.api_service = api_service
        self.entity_linker = entity_linker
    
    def route(self, intent: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route intent and slots to the appropriate API handler.
        
        Args:
            intent: The predicted intent (e.g., "team_info", "player_info")
            slots: The predicted slots (e.g., {"input": "Lakers", "attribute": "conference"})
            
        Returns:
            Dictionary containing:
                - "success": bool indicating if the API call succeeded
                - "data": The API response data (if successful)
                - "error": Error message (if failed)
                - "intent": The intent that was processed
                - "slots": The slots that were used
        """
        try:
            # Route to appropriate handler based on intent
            if intent == "team_info":
                return self.handle_team_info(slots)
            elif intent == "player_info":
                return self.handle_player_info(slots)
            elif intent == "game_lookup":
                return self.handle_game_lookup(slots)
            elif intent == "standings":
                return self.handle_standings(slots)
            elif intent == "leaders":
                return self.handle_leaders(slots)
            elif intent == "season_averages":
                return self.handle_season_averages(slots)
            elif intent == "box_scores":
                return self.handle_box_scores(slots)
            elif intent == "injuries":
                return self.handle_injuries(slots)
            elif intent == "odds":
                return self.handle_odds(slots)
            else:
                return {
                    "success": False,
                    "error": f"Unknown intent: {intent}",
                    "intent": intent,
                    "slots": slots
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing {intent}: {str(e)}",
                "intent": intent,
                "slots": slots
            }
    
    def handle_team_info(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle team_info intent.
        
        Required slots:
            - input: Team name (e.g., "Lakers", "Los Angeles Lakers")
            - attribute: The attribute to retrieve (e.g., "conference", "division", "city", "full_name", "abbreviation")
        
        Args:
            slots: Dictionary containing "input" (team name) and "attribute"
            
        Returns:
            Dictionary with API result containing the requested attribute
        """
        # Validate required slots
        if "input" not in slots or not slots["input"]:
            return {
                "success": False,
                "error": "Missing required slot: 'input' (team name)",
                "intent": "team_info",
                "slots": slots
            }
        
        if "attribute" not in slots or not slots["attribute"]:
            return {
                "success": False,
                "error": "Missing required slot: 'attribute'",
                "intent": "team_info",
                "slots": slots
            }
        
        team_name = slots["input"]
        attribute = slots["attribute"]
        
        # Link team name to team ID
        team_id = self.entity_linker.link_team(team_name)
        if not team_id:
            return {
                "success": False,
                "error": f"Team not found: {team_name}",
                "intent": "team_info",
                "slots": slots
            }
        
        # Get team data
        team_data = self.api_service.get_team_by_id(team_id)
        if not team_data:
            return {
                "success": False,
                "error": f"Failed to retrieve team data for ID: {team_id}",
                "intent": "team_info",
                "slots": slots
            }
        
        # Extract the requested attribute
        if attribute not in team_data:
            return {
                "success": False,
                "error": f"Attribute '{attribute}' not found in team data. Available attributes: {list(team_data.keys())}",
                "intent": "team_info",
                "slots": slots
            }
        
        return {
            "success": True,
            "data": {
                "team_id": team_id,
                "team_name": team_data.get("full_name", team_name),
                "attribute": attribute,
                "value": team_data[attribute]
            },
            "intent": "team_info",
            "slots": slots
        }
    
    def handle_player_info(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle player_info intent.
        
        Required slots:
            - input: Player name (e.g., "Stephen Curry", "LeBron James")
            - attribute: The attribute to retrieve (e.g., "position", "height", "weight", "jersey_number", "college", "country", "draft_year", etc.)
        
        Args:
            slots: Dictionary containing "input" (player name) and "attribute"
            
        Returns:
            Dictionary with API result containing the requested attribute
        """
        # Validate required slots
        if "input" not in slots or not slots["input"]:
            return {
                "success": False,
                "error": "Missing required slot: 'input' (player name)",
                "intent": "player_info",
                "slots": slots
            }
        
        if "attribute" not in slots or not slots["attribute"]:
            return {
                "success": False,
                "error": "Missing required slot: 'attribute'",
                "intent": "player_info",
                "slots": slots
            }
        
        player_name = slots["input"]
        attribute = slots["attribute"]
        
        # Link player name to player ID
        player_id = self.entity_linker.link_player(player_name)
        if not player_id:
            return {
                "success": False,
                "error": f"Player not found: {player_name}",
                "intent": "player_info",
                "slots": slots
            }
        
        # Get player data
        player_data = self.api_service.get_player_by_id(player_id)
        if not player_data:
            return {
                "success": False,
                "error": f"Failed to retrieve player data for ID: {player_id}",
                "intent": "player_info",
                "slots": slots
            }
        
        # Extract the requested attribute
        if attribute not in player_data:
            return {
                "success": False,
                "error": f"Attribute '{attribute}' not found in player data. Available attributes: {list(player_data.keys())}",
                "intent": "player_info",
                "slots": slots
            }
        
        return {
            "success": True,
            "data": {
                "player_id": player_id,
                "player_name": f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip(),
                "attribute": attribute,
                "value": player_data[attribute]
            },
            "intent": "player_info",
            "slots": slots
        }
    
    def handle_game_lookup(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle game_lookup intent.
        
        Required slots (one of):
            - date: Date in YYYY-MM-DD format
            - game_id: Game ID
        
        Args:
            slots: Dictionary containing "date" or "game_id"
            
        Returns:
            Dictionary with game data
        """
        if "game_id" in slots and slots["game_id"]:
            # Get game by ID
            game_id = slots["game_id"]
            try:
                game_id = int(game_id)
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "error": f"Invalid game_id: {game_id}",
                    "intent": "game_lookup",
                    "slots": slots
                }
            
            game_data = self.api_service.get_game_by_id(game_id)
            if not game_data:
                return {
                    "success": False,
                    "error": f"Game not found: {game_id}",
                    "intent": "game_lookup",
                    "slots": slots
                }
            
            return {
                "success": True,
                "data": game_data,
                "intent": "game_lookup",
                "slots": slots
            }
        
        elif "date" in slots and slots["date"]:
            # Get games by date
            date = slots["date"]
            games = self.api_service.get_games_by_date(date)
            
            if not games:
                return {
                    "success": False,
                    "error": f"No games found for date: {date}",
                    "intent": "game_lookup",
                    "slots": slots
                }
            
            return {
                "success": True,
                "data": {
                    "date": date,
                    "games": games,
                    "count": len(games)
                },
                "intent": "game_lookup",
                "slots": slots
            }
        else:
            return {
                "success": False,
                "error": "Missing required slot: 'date' or 'game_id'",
                "intent": "game_lookup",
                "slots": slots
            }
    
    def handle_standings(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle standings intent.
        
        Optional slots:
            - season: Season year (e.g., 2023)
        
        Args:
            slots: Dictionary optionally containing "season"
            
        Returns:
            Dictionary with standings data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        season = slots.get("season", 2023)
        
        return {
            "success": False,
            "error": "Standings endpoint not yet implemented. This may require a paid API tier.",
            "intent": "standings",
            "slots": slots
        }
    
    def handle_leaders(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle leaders intent.
        
        Required slots:
            - stat_type: Type of statistic (e.g., "pts", "reb", "ast")
            - season: Season year (e.g., 2023)
        
        Args:
            slots: Dictionary containing "stat_type" and "season"
            
        Returns:
            Dictionary with leaders data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        stat_type = slots.get("stat_type")
        season = slots.get("season", 2023)
        
        if not stat_type:
            return {
                "success": False,
                "error": "Missing required slot: 'stat_type'",
                "intent": "leaders",
                "slots": slots
            }
        
        return {
            "success": False,
            "error": "Leaders endpoint not yet implemented. This may require a paid API tier.",
            "intent": "leaders",
            "slots": slots
        }
    
    def handle_season_averages(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle season_averages intent.
        
        Required slots:
            - input: Player name
            - season: Season year (e.g., 2023)
        
        Args:
            slots: Dictionary containing "input" (player name) and "season"
            
        Returns:
            Dictionary with season averages data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        player_name = slots.get("input")
        season = slots.get("season", 2023)
        
        if not player_name:
            return {
                "success": False,
                "error": "Missing required slot: 'input' (player name)",
                "intent": "season_averages",
                "slots": slots
            }
        
        # Link player name to player ID
        player_id = self.entity_linker.link_player(player_name)
        if not player_id:
            return {
                "success": False,
                "error": f"Player not found: {player_name}",
                "intent": "season_averages",
                "slots": slots
            }
        
        return {
            "success": False,
            "error": "Season averages endpoint not yet implemented. This may require a paid API tier.",
            "intent": "season_averages",
            "slots": slots
        }
    
    def handle_box_scores(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle box_scores intent.
        
        Required slots:
            - date: Date in YYYY-MM-DD format
        
        Args:
            slots: Dictionary containing "date"
            
        Returns:
            Dictionary with box scores data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        date = slots.get("date")
        
        if not date:
            return {
                "success": False,
                "error": "Missing required slot: 'date'",
                "intent": "box_scores",
                "slots": slots
            }
        
        return {
            "success": False,
            "error": "Box scores endpoint not yet implemented. This may require a paid API tier.",
            "intent": "box_scores",
            "slots": slots
        }
    
    def handle_injuries(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle injuries intent.
        
        No required slots.
        
        Args:
            slots: Dictionary (may be empty)
            
        Returns:
            Dictionary with injuries data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        return {
            "success": False,
            "error": "Injuries endpoint not yet implemented. This may require a paid API tier.",
            "intent": "injuries",
            "slots": slots
        }
    
    def handle_odds(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle odds intent.
        
        Required slots:
            - date: Date in YYYY-MM-DD format
        
        Args:
            slots: Dictionary containing "date"
            
        Returns:
            Dictionary with odds data
        """
        # Note: This endpoint may not be available in the Free tier
        # Placeholder implementation
        date = slots.get("date")
        
        if not date:
            return {
                "success": False,
                "error": "Missing required slot: 'date'",
                "intent": "odds",
                "slots": slots
            }
        
        return {
            "success": False,
            "error": "Odds endpoint not yet implemented. This may require a paid API tier.",
            "intent": "odds",
            "slots": slots
        }

