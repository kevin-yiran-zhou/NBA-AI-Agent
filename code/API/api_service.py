from typing import Optional, List, Dict, Any
from balldontlie import BalldontlieAPI
from balldontlie.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    ServerError,
    BallDontLieException
)
import os


class NBAApiService:
    """Service class for interacting with the BALLDONTLIE NBA API."""
    
    def __init__(self, api_key_path: str = None):
        if api_key_path is None:
            # Default to API_KEY.txt in the same directory as this file
            api_key_path = os.path.join(os.path.dirname(__file__), 'API_KEY.txt')
        
        if not os.path.exists(api_key_path):
            raise FileNotFoundError(f"API key file not found: {api_key_path}")
        
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        
        if not api_key:
            raise ValueError("API key is empty")
        
        self.client = BalldontlieAPI(api_key=api_key)
        self._teams_cache: Optional[List[Dict[str, Any]]] = None
    
    def _handle_api_error(self, error: Exception, operation: str) -> None:
        """Handle API errors and raise appropriate exceptions."""
        if isinstance(error, AuthenticationError):
            raise AuthenticationError(
                f"Authentication failed for {operation}. "
                f"Please check your API key. Status: {error.status_code}"
            ) from error
        elif isinstance(error, RateLimitError):
            raise RateLimitError(
                f"Rate limit exceeded for {operation}. "
                f"Please wait before retrying. Status: {error.status_code}"
            ) from error
        elif isinstance(error, NotFoundError):
            raise NotFoundError(
                f"Resource not found for {operation}. "
                f"Status: {error.status_code}"
            ) from error
        elif isinstance(error, ValidationError):
            raise ValidationError(
                f"Invalid request parameters for {operation}. "
                f"Status: {error.status_code}"
            ) from error
        elif isinstance(error, ServerError):
            raise ServerError(
                f"API server error for {operation}. "
                f"Please try again later. Status: {error.status_code}"
            ) from error
        else:
            raise BallDontLieException(
                f"Unexpected error during {operation}: {str(error)}"
            ) from error
    
    # ==================== Team Methods ====================
    
    def list_all_teams(self, division: Optional[str] = None, 
                      conference: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all NBA teams."""
        try:
            response = self.client.nba.teams.list(
                division=division,
                conference=conference
            )
            teams = []
            for team in response.data:
                teams.append({
                    'id': team.id,
                    'conference': team.conference,
                    'division': team.division,
                    'city': team.city,
                    'name': team.name,
                    'full_name': team.full_name,
                    'abbreviation': team.abbreviation
                })
            return teams
        except BallDontLieException as e:
            self._handle_api_error(e, "list_all_teams")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in list_all_teams: {str(e)}") from e
    
    def get_team_by_id(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific team by ID."""
        try:
            response = self.client.nba.teams.get(team_id)
            team = response.data
            return {
                'id': team.id,
                'conference': team.conference,
                'division': team.division,
                'city': team.city,
                'name': team.name,
                'full_name': team.full_name,
                'abbreviation': team.abbreviation
            }
        except NotFoundError:
            return None
        except BallDontLieException as e:
            self._handle_api_error(e, f"get_team_by_id({team_id})")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in get_team_by_id: {str(e)}") from e
    
    def get_team_by_name(self, team_name: str) -> Optional[Dict[str, Any]]:
        """Get a team by name."""
        try:
            teams = self.list_all_teams()
            team_name_lower = team_name.lower().strip()
            
            for team in teams:
                # Check full_name, name, abbreviation, city+name
                if (team_name_lower == team['full_name'].lower() or
                    team_name_lower == team['name'].lower() or
                    team_name_lower == team['abbreviation'].lower() or
                    team_name_lower == f"{team['city']} {team['name']}".lower()):
                    return team
            
            return None
        except Exception as e:
            raise BallDontLieException(f"Error in get_team_by_name: {str(e)}") from e
    
    # ==================== Player Methods ====================
    
    def list_players(self, per_page: int = 25, cursor: Optional[int] = None,
                    search: Optional[str] = None,
                    first_name: Optional[str] = None,
                    last_name: Optional[str] = None,
                    team_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get a list of players with optional filters."""
        try:
            response = self.client.nba.players.list(
                per_page=per_page,
                cursor=cursor,
                search=search,
                first_name=first_name,
                last_name=last_name,
                team_ids=team_ids
            )
            
            players = []
            for player in response.data:
                player_dict = {
                    'id': player.id,
                    'first_name': player.first_name,
                    'last_name': player.last_name,
                    'position': player.position,
                    'height': player.height,
                    'weight': player.weight,
                    'jersey_number': player.jersey_number,
                    'college': player.college,
                    'country': player.country,
                    'draft_year': player.draft_year,
                    'draft_round': player.draft_round,
                    'draft_number': player.draft_number
                }
                # Add team info if available
                if hasattr(player, 'team') and player.team:
                    player_dict['team'] = {
                        'id': player.team.id,
                        'conference': player.team.conference,
                        'division': player.team.division,
                        'city': player.team.city,
                        'name': player.team.name,
                        'full_name': player.team.full_name,
                        'abbreviation': player.team.abbreviation
                    }
                players.append(player_dict)
            
            meta = {
                'next_cursor': response.meta.next_cursor if hasattr(response.meta, 'next_cursor') else None,
                'per_page': response.meta.per_page if hasattr(response.meta, 'per_page') else per_page
            }
            
            return {'data': players, 'meta': meta}
        except BallDontLieException as e:
            self._handle_api_error(e, "list_players")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in list_players: {str(e)}") from e
    
    def get_player_by_id(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific player by ID."""
        try:
            response = self.client.nba.players.get(player_id)
            player = response.data
            
            player_dict = {
                'id': player.id,
                'first_name': player.first_name,
                'last_name': player.last_name,
                'position': player.position,
                'height': player.height,
                'weight': player.weight,
                'jersey_number': player.jersey_number,
                'college': player.college,
                'country': player.country,
                'draft_year': player.draft_year,
                'draft_round': player.draft_round,
                'draft_number': player.draft_number
            }
            
            # Add team info if available
            if hasattr(player, 'team') and player.team:
                player_dict['team'] = {
                    'id': player.team.id,
                    'conference': player.team.conference,
                    'division': player.team.division,
                    'city': player.team.city,
                    'name': player.team.name,
                    'full_name': player.team.full_name,
                    'abbreviation': player.team.abbreviation
                }
            
            return player_dict
        except NotFoundError:
            return None
        except BallDontLieException as e:
            self._handle_api_error(e, f"get_player_by_id({player_id})")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in get_player_by_id: {str(e)}") from e
    
    def search_players(self, query: str, per_page: int = 25) -> List[Dict[str, Any]]:
        """Search for players by name."""
        try:
            result = self.list_players(search=query, per_page=per_page)
            return result['data']
        except Exception as e:
            raise BallDontLieException(f"Error in search_players: {str(e)}") from e
    
    def get_player_by_name(self, first_name: Optional[str] = None,
                          last_name: Optional[str] = None,
                          full_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a player by name."""
        try:
            if full_name:
                # Try to split full name
                parts = full_name.strip().split()
                if len(parts) >= 2:
                    first_name = parts[0]
                    last_name = ' '.join(parts[1:])
                else:
                    # If single word, search as last name
                    last_name = parts[0]
            
            result = self.list_players(
                first_name=first_name,
                last_name=last_name,
                per_page=100
            )
            
            players = result['data']
            if not players:
                return None
            
            # If we have exact matches, return the first one
            # Otherwise, return the first result
            return players[0] if players else None
        except Exception as e:
            raise BallDontLieException(f"Error in get_player_by_name: {str(e)}") from e
    
    # ==================== Game Methods ====================
    
    def list_games(self, per_page: int = 25, cursor: Optional[int] = None,
                  dates: Optional[List[str]] = None,
                  seasons: Optional[List[int]] = None,
                  team_ids: Optional[List[int]] = None,
                  postseason: Optional[bool] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get a list of games with optional filters."""
        try:
            response = self.client.nba.games.list(
                per_page=per_page,
                cursor=cursor,
                dates=dates,
                seasons=seasons,
                team_ids=team_ids,
                postseason=postseason,
                start_date=start_date,
                end_date=end_date
            )
            
            games = []
            for game in response.data:
                game_dict = {
                    'id': game.id,
                    'date': game.date,
                    'season': game.season,
                    'status': game.status,
                    'period': game.period,
                    'time': game.time,
                    'postseason': game.postseason,
                    'home_team_score': game.home_team_score,
                    'visitor_team_score': game.visitor_team_score,
                    'datetime': getattr(game, 'datetime', None)
                }
                
                # Add team info
                if hasattr(game, 'home_team') and game.home_team:
                    game_dict['home_team'] = {
                        'id': game.home_team.id,
                        'conference': game.home_team.conference,
                        'division': game.home_team.division,
                        'city': game.home_team.city,
                        'name': game.home_team.name,
                        'full_name': game.home_team.full_name,
                        'abbreviation': game.home_team.abbreviation
                    }
                
                if hasattr(game, 'visitor_team') and game.visitor_team:
                    game_dict['visitor_team'] = {
                        'id': game.visitor_team.id,
                        'conference': game.visitor_team.conference,
                        'division': game.visitor_team.division,
                        'city': game.visitor_team.city,
                        'name': game.visitor_team.name,
                        'full_name': game.visitor_team.full_name,
                        'abbreviation': game.visitor_team.abbreviation
                    }
                
                games.append(game_dict)
            
            meta = {
                'next_cursor': response.meta.next_cursor if hasattr(response.meta, 'next_cursor') else None,
                'per_page': response.meta.per_page if hasattr(response.meta, 'per_page') else per_page
            }
            
            return {'data': games, 'meta': meta}
        except BallDontLieException as e:
            self._handle_api_error(e, "list_games")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in list_games: {str(e)}") from e
    
    def get_game_by_id(self, game_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific game by ID."""
        try:
            response = self.client.nba.games.get(game_id)
            game = response.data
            
            game_dict = {
                'id': game.id,
                'date': game.date,
                'season': game.season,
                'status': game.status,
                'period': game.period,
                'time': game.time,
                'postseason': game.postseason,
                'home_team_score': game.home_team_score,
                'visitor_team_score': game.visitor_team_score,
                'datetime': getattr(game, 'datetime', None)
            }
            
            # Add team info
            if hasattr(game, 'home_team') and game.home_team:
                game_dict['home_team'] = {
                    'id': game.home_team.id,
                    'conference': game.home_team.conference,
                    'division': game.home_team.division,
                    'city': game.home_team.city,
                    'name': game.home_team.name,
                    'full_name': game.home_team.full_name,
                    'abbreviation': game.home_team.abbreviation
                }
            
            if hasattr(game, 'visitor_team') and game.visitor_team:
                game_dict['visitor_team'] = {
                    'id': game.visitor_team.id,
                    'conference': game.visitor_team.conference,
                    'division': game.visitor_team.division,
                    'city': game.visitor_team.city,
                    'name': game.visitor_team.name,
                    'full_name': game.visitor_team.full_name,
                    'abbreviation': game.visitor_team.abbreviation
                }
            
            return game_dict
        except NotFoundError:
            return None
        except BallDontLieException as e:
            self._handle_api_error(e, f"get_game_by_id({game_id})")
        except Exception as e:
            raise BallDontLieException(f"Unexpected error in get_game_by_id: {str(e)}") from e
    
    def get_games_by_date(self, date: str, per_page: int = 100) -> List[Dict[str, Any]]:
        """Get all games for a specific date."""
        try:
            result = self.list_games(dates=[date], per_page=per_page)
            return result['data']
        except Exception as e:
            raise BallDontLieException(f"Error in get_games_by_date: {str(e)}") from e
    
    def clear_cache(self) -> None:
        """Clear any cached data."""
        self._teams_cache = None

