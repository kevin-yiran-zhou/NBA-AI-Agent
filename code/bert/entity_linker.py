"""
Entity Linking Module

This module links entity names (teams/players) recognized by the model
to their corresponding API IDs using fuzzy matching and alias handling.
"""

from typing import Optional, Dict, List, Any
from difflib import SequenceMatcher
from api_service import NBAApiService


class EntityLinker:
    """
    Links entity names to API IDs using fuzzy matching and alias resolution.
    
    This class handles:
    - Team name variations (e.g., "Lakers" -> "Los Angeles Lakers")
    - Player name matching (full names, partial names)
    - Fuzzy matching for typos and variations
    - Caching for performance
    """
    
    # Common team aliases and variations
    TEAM_ALIASES = {
        "lakers": "Los Angeles Lakers",
        "lal": "Los Angeles Lakers",
        "warriors": "Golden State Warriors",
        "gsw": "Golden State Warriors",
        "celtics": "Boston Celtics",
        "bos": "Boston Celtics",
        "heat": "Miami Heat",
        "mia": "Miami Heat",
        "nuggets": "Denver Nuggets",
        "den": "Denver Nuggets",
        "suns": "Phoenix Suns",
        "phx": "Phoenix Suns",
        "clippers": "LA Clippers",
        "lac": "LA Clippers",
        "knicks": "New York Knicks",
        "nyk": "New York Knicks",
        "76ers": "Philadelphia 76ers",
        "sixers": "Philadelphia 76ers",
        "phi": "Philadelphia 76ers",
        "nets": "Brooklyn Nets",
        "bkn": "Brooklyn Nets",
        "raptors": "Toronto Raptors",
        "tor": "Toronto Raptors",
        "bulls": "Chicago Bulls",
        "chi": "Chicago Bulls",
        "bucks": "Milwaukee Bucks",
        "mil": "Milwaukee Bucks",
        "cavaliers": "Cleveland Cavaliers",
        "cavs": "Cleveland Cavaliers",
        "cle": "Cleveland Cavaliers",
        "hawks": "Atlanta Hawks",
        "atl": "Atlanta Hawks",
        "wizards": "Washington Wizards",
        "was": "Washington Wizards",
        "hornets": "Charlotte Hornets",
        "cha": "Charlotte Hornets",
        "magic": "Orlando Magic",
        "orl": "Orlando Magic",
        "pistons": "Detroit Pistons",
        "det": "Detroit Pistons",
        "pacers": "Indiana Pacers",
        "ind": "Indiana Pacers",
        "rockets": "Houston Rockets",
        "hou": "Houston Rockets",
        "mavericks": "Dallas Mavericks",
        "mavs": "Dallas Mavericks",
        "dal": "Dallas Mavericks",
        "spurs": "San Antonio Spurs",
        "sas": "San Antonio Spurs",
        "grizzlies": "Memphis Grizzlies",
        "mem": "Memphis Grizzlies",
        "pelicans": "New Orleans Pelicans",
        "nop": "New Orleans Pelicans",
        "thunder": "Oklahoma City Thunder",
        "okc": "Oklahoma City Thunder",
        "trail blazers": "Portland Trail Blazers",
        "blazers": "Portland Trail Blazers",
        "por": "Portland Trail Blazers",
        "timberwolves": "Minnesota Timberwolves",
        "wolves": "Minnesota Timberwolves",
        "min": "Minnesota Timberwolves",
        "jazz": "Utah Jazz",
        "uta": "Utah Jazz",
        "kings": "Sacramento Kings",
        "sac": "Sacramento Kings",
    }
    
    def __init__(self, api_service: NBAApiService):
        """
        Initialize the EntityLinker.
        
        Args:
            api_service: An instance of NBAApiService for API calls
        """
        self.api_service = api_service
        self._teams_cache: Optional[List[Dict[str, Any]]] = None
        self._players_cache: Optional[List[Dict[str, Any]]] = None
        self._fuzzy_threshold = 0.7  # Minimum similarity for fuzzy matching
    
    def _get_teams_cache(self) -> List[Dict[str, Any]]:
        """Get or load teams cache."""
        if self._teams_cache is None:
            self._teams_cache = self.api_service.list_all_teams()
        return self._teams_cache
    
    def _get_players_cache(self) -> List[Dict[str, Any]]:
        """Get or load players cache (limited to first 1000 for performance)."""
        if self._players_cache is None:
            # Load a reasonable number of players for caching
            # In production, you might want to load all players or use a more sophisticated caching strategy
            result = self.api_service.list_players(per_page=1000)
            self._players_cache = result['data']
            # Also try to load more players if needed (up to 2000)
            # This helps ensure we have popular players in cache
            if result.get('meta', {}).get('next_cursor'):
                try:
                    result2 = self.api_service.list_players(per_page=1000, cursor=result['meta']['next_cursor'])
                    self._players_cache.extend(result2['data'])
                except Exception:
                    pass
        return self._players_cache
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison (lowercase, strip, remove extra spaces)."""
        return ' '.join(name.lower().strip().split())
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def link_team(self, team_name: str) -> Optional[int]:
        """
        Link a team name to its team_id.
        
        Handles:
        - Exact matches (full_name, name, abbreviation)
        - Alias matching (e.g., "Lakers" -> "Los Angeles Lakers")
        - Fuzzy matching for typos
        
        Args:
            team_name: The team name to link
            
        Returns:
            team_id if found, None otherwise
        """
        if not team_name or not team_name.strip():
            return None
        
        teams = self._get_teams_cache()
        normalized_input = self._normalize_name(team_name)
        
        # Step 1: Check aliases
        if normalized_input in self.TEAM_ALIASES:
            canonical_name = self.TEAM_ALIASES[normalized_input]
            for team in teams:
                if self._normalize_name(team['full_name']) == self._normalize_name(canonical_name):
                    return team['id']
        
        # Step 2: Exact matches
        for team in teams:
            # Check full_name
            if self._normalize_name(team['full_name']) == normalized_input:
                return team['id']
            # Check name
            if self._normalize_name(team['name']) == normalized_input:
                return team['id']
            # Check abbreviation
            if self._normalize_name(team['abbreviation']) == normalized_input:
                return team['id']
            # Check city + name
            city_name = f"{team['city']} {team['name']}"
            if self._normalize_name(city_name) == normalized_input:
                return team['id']
        
        # Step 3: Partial matches (e.g., "Lakers" in "Los Angeles Lakers")
        for team in teams:
            full_name_norm = self._normalize_name(team['full_name'])
            name_norm = self._normalize_name(team['name'])
            
            # Check if input is contained in full_name or vice versa
            if normalized_input in full_name_norm or full_name_norm in normalized_input:
                return team['id']
            if normalized_input in name_norm or name_norm in normalized_input:
                return team['id']
        
        # Step 4: Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for team in teams:
            # Compare against full_name, name, and abbreviation
            scores = [
                self._similarity(normalized_input, team['full_name']),
                self._similarity(normalized_input, team['name']),
                self._similarity(normalized_input, team['abbreviation']),
                self._similarity(normalized_input, f"{team['city']} {team['name']}")
            ]
            max_score = max(scores)
            
            if max_score > best_score:
                best_score = max_score
                best_match = team
        
        if best_match and best_score >= self._fuzzy_threshold:
            return best_match['id']
        
        return None
    
    def link_player(self, player_name: str) -> Optional[int]:
        """
        Link a player name to its player_id.
        
        Handles:
        - Full name matching (e.g., "Stephen Curry")
        - Partial name matching (e.g., "Curry", "Stephen")
        - Fuzzy matching for typos
        
        Args:
            player_name: The player name to link (can be full name or partial)
            
        Returns:
            player_id if found, None otherwise
        """
        if not player_name or not player_name.strip():
            return None
        
        players = self._get_players_cache()
        normalized_input = self._normalize_name(player_name)
        input_parts = normalized_input.split()
        
        # Step 1: Exact full name match
        for player in players:
            full_name = f"{player['first_name']} {player['last_name']}"
            if self._normalize_name(full_name) == normalized_input:
                return player['id']
        
        # Step 2: Partial name matching (last name or first name)
        # If input is a single word, try matching against last_name
        if len(input_parts) == 1:
            for player in players:
                if self._normalize_name(player['last_name']) == normalized_input:
                    return player['id']
                # Also check first name
                if self._normalize_name(player['first_name']) == normalized_input:
                    return player['id']
        # If input has multiple words, try matching first and last name
        elif len(input_parts) >= 2:
            first_part = input_parts[0]
            last_part = ' '.join(input_parts[1:])
            
            for player in players:
                first_match = self._normalize_name(player['first_name']) == first_part
                last_match = self._normalize_name(player['last_name']) == last_part
                
                if first_match and last_match:
                    return player['id']
        
        # Step 3: Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for player in players:
            full_name = f"{player['first_name']} {player['last_name']}"
            scores = [
                self._similarity(normalized_input, full_name),
                self._similarity(normalized_input, player['last_name']),
                self._similarity(normalized_input, player['first_name'])
            ]
            max_score = max(scores)
            
            if max_score > best_score:
                best_score = max_score
                best_match = player
        
        # Use a slightly higher threshold for players (0.75) since names can be similar
        if best_match and best_score >= 0.75:
            return best_match['id']
        
        # Step 4: If cache doesn't have the player, try API search
        try:
            # If we have multiple words, try using get_player_by_name with first_name and last_name
            if len(input_parts) >= 2:
                first_name = input_parts[0]
                last_name = ' '.join(input_parts[1:])
                player_result = self.api_service.get_player_by_name(
                    first_name=first_name,
                    last_name=last_name
                )
                if player_result:
                    return player_result['id']
            
            # Try full name search
            search_results = self.api_service.search_players(player_name, per_page=10)
            
            # If full name search fails and we have multiple words, try searching by parts
            if not search_results and len(input_parts) >= 2:
                # Try searching by last name (usually more unique)
                last_name = input_parts[-1]
                search_results = self.api_service.search_players(last_name, per_page=10)
            
            if search_results:
                # Find the best match from search results
                best_match = None
                best_score = 0.0
                
                for result in search_results:
                    full_name = f"{result['first_name']} {result['last_name']}"
                    score = self._similarity(normalized_input, full_name)
                    
                    # For exact matches, return immediately
                    if self._normalize_name(full_name) == normalized_input:
                        return result['id']
                    
                    if score > best_score:
                        best_score = score
                        best_match = result
                
                # Return best match if similarity is high enough
                if best_match and best_score >= 0.75:
                    return best_match['id']
                
                # If we have a partial match (e.g., "Curry" matches "Stephen Curry"), return the first result
                if len(input_parts) == 1 and search_results:
                    # Single word input - check if it matches last name
                    for result in search_results:
                        if self._normalize_name(result['last_name']) == normalized_input:
                            return result['id']
                
                # For multi-word input, try matching first and last name separately
                if len(input_parts) >= 2 and search_results:
                    first_part = input_parts[0]
                    last_part = ' '.join(input_parts[1:])
                    for result in search_results:
                        result_first = self._normalize_name(result['first_name'])
                        result_last = self._normalize_name(result['last_name'])
                        if (result_first == first_part or 
                            self._similarity(result_first, first_part) >= 0.8):
                            if (result_last == last_part or 
                                self._similarity(result_last, last_part) >= 0.8):
                                return result['id']
        except Exception:
            pass
        
        return None
    
    def link_entity(self, entity_type: str, entity_name: str) -> Optional[int]:
        """
        Generic entity linking interface.
        
        Args:
            entity_type: Type of entity ("team" or "player")
            entity_name: Name of the entity
            
        Returns:
            entity_id if found, None otherwise
        """
        if entity_type.lower() == "team":
            return self.link_team(entity_name)
        elif entity_type.lower() == "player":
            return self.link_player(entity_name)
        else:
            return None
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._teams_cache = None
        self._players_cache = None

