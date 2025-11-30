"""
Test script for api_service.py

This script tests the NBAApiService class to ensure all methods work correctly.
"""

try:
    from .api_service import NBAApiService
except ImportError:
    # Allow running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api_service import NBAApiService
from balldontlie.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError
)


def test_teams(api_service: NBAApiService):
    """Test team-related methods."""
    print("\n" + "="*50)
    print("Testing Team Methods")
    print("="*50)
    
    # Test list_all_teams
    print("\n1. Testing list_all_teams()...")
    try:
        teams = api_service.list_all_teams()
        print(f"   ✓ Successfully retrieved {len(teams)} teams")
        if teams:
            print(f"   Sample team: {teams[0]['full_name']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_team_by_id
    print("\n2. Testing get_team_by_id(10)...")
    try:
        team = api_service.get_team_by_id(10)
        if team:
            print(f"   ✓ Found team: {team['full_name']} ({team['abbreviation']})")
        else:
            print("   ✗ Team not found")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_team_by_name
    print("\n3. Testing get_team_by_name('Lakers')...")
    try:
        team = api_service.get_team_by_name("Lakers")
        if team:
            print(f"   ✓ Found team: {team['full_name']} (ID: {team['id']})")
        else:
            print("   ✗ Team not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True


def test_players(api_service: NBAApiService):
    """Test player-related methods."""
    print("\n" + "="*50)
    print("Testing Player Methods")
    print("="*50)
    
    # Test list_players
    print("\n1. Testing list_players(per_page=5)...")
    try:
        result = api_service.list_players(per_page=5)
        players = result['data']
        print(f"   ✓ Successfully retrieved {len(players)} players")
        if players:
            player = players[0]
            print(f"   Sample player: {player['first_name']} {player['last_name']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_player_by_id
    print("\n2. Testing get_player_by_id(19)...")
    try:
        player = api_service.get_player_by_id(19)
        if player:
            print(f"   ✓ Found player: {player['first_name']} {player['last_name']}")
            if 'team' in player and player['team']:
                print(f"   Team: {player['team']['full_name']}")
        else:
            print("   ✗ Player not found")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test search_players
    print("\n3. Testing search_players('Curry')...")
    try:
        players = api_service.search_players("Curry", per_page=5)
        print(f"   ✓ Found {len(players)} players matching 'Curry'")
        if players:
            print(f"   Sample: {players[0]['first_name']} {players[0]['last_name']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_player_by_name
    print("\n4. Testing get_player_by_name(first_name='Stephen', last_name='Curry')...")
    try:
        player = api_service.get_player_by_name(first_name="Stephen", last_name="Curry")
        if player:
            print(f"   ✓ Found player: {player['first_name']} {player['last_name']}")
        else:
            print("   ✗ Player not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True


def test_games(api_service: NBAApiService):
    """Test game-related methods."""
    print("\n" + "="*50)
    print("Testing Game Methods")
    print("="*50)
    
    # Test list_games with date filter
    print("\n1. Testing list_games(dates=['2024-10-30'])...")
    try:
        result = api_service.list_games(dates=["2024-10-30"], per_page=10)
        games = result['data']
        print(f"   ✓ Successfully retrieved {len(games)} games")
        if games:
            game = games[0]
            home = game.get('home_team', {}).get('full_name', 'Unknown')
            visitor = game.get('visitor_team', {}).get('full_name', 'Unknown')
            print(f"   Sample game: {visitor} @ {home}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_game_by_id
    print("\n2. Testing get_game_by_id()...")
    try:
        result = api_service.list_games(dates=["2024-10-30"], per_page=1)
        if result['data']:
            game_id = result['data'][0]['id']
            game = api_service.get_game_by_id(game_id)
            if game:
                home = game.get('home_team', {}).get('full_name', 'Unknown')
                visitor = game.get('visitor_team', {}).get('full_name', 'Unknown')
                print(f"   ✓ Found game: {visitor} @ {home}")
                print(f"   Score: {game.get('visitor_team_score', 'N/A')} - {game.get('home_team_score', 'N/A')}")
            else:
                print("   ✗ Game not found")
                return False
        else:
            print("   ⚠ No games found to test get_game_by_id")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test get_games_by_date
    print("\n3. Testing get_games_by_date('2024-10-30')...")
    try:
        games = api_service.get_games_by_date("2024-10-30")
        print(f"   ✓ Successfully retrieved {len(games)} games for 2024-10-30")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*50)
    print("NBA API Service Test Suite")
    print("="*50)
    
    # Initialize API service
    try:
        api_service = NBAApiService()
        print("\n✓ API service initialized successfully")
    except FileNotFoundError as e:
        print(f"\n✗ Failed to initialize API service: {e}")
        print("   Please ensure API_KEY.txt exists in the API directory")
        return
    except Exception as e:
        print(f"\n✗ Failed to initialize API service: {e}")
        return
    
    # Run tests
    results = []
    
    results.append(("Teams", test_teams(api_service)))
    results.append(("Players", test_players(api_service)))
    results.append(("Games", test_games(api_service)))
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*50)


if __name__ == "__main__":
    main()

