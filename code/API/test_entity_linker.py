"""
Test script for entity_linker.py

This script tests the EntityLinker class to ensure entity linking works correctly.
"""

try:
    from .api_service import NBAApiService
    from .entity_linker import EntityLinker
except ImportError:
    # Allow running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api_service import NBAApiService
    from entity_linker import EntityLinker


def test_team_linking(linker: EntityLinker):
    """Test team linking functionality."""
    print("\n" + "="*50)
    print("Testing Team Linking")
    print("="*50)
    
    test_cases = [
        ("Los Angeles Lakers", "Exact full name"),
        ("Lakers", "Alias/partial name"),
        ("LAL", "Abbreviation"),
        ("Golden State Warriors", "Exact full name"),
        ("Warriors", "Partial name"),
        ("GSW", "Abbreviation"),
        ("Boston Celtics", "Exact full name"),
        ("Celtics", "Partial name"),
        ("Miami Heat", "Exact full name"),
        ("Heat", "Partial name"),
    ]
    
    passed = 0
    failed = 0
    
    for team_name, description in test_cases:
        try:
            team_id = linker.link_team(team_name)
            if team_id:
                # Verify by getting team info
                team_info = linker.api_service.get_team_by_id(team_id)
                if team_info:
                    print(f"✓ {description}: '{team_name}' -> ID {team_id} ({team_info['full_name']})")
                    passed += 1
                else:
                    print(f"[FAIL] {description}: '{team_name}' -> ID {team_id} (but team not found)")
                    failed += 1
            else:
                print(f"✗ {description}: '{team_name}' -> Not found")
                failed += 1
        except Exception as e:
            print(f"✗ {description}: '{team_name}' -> Error: {e}")
            failed += 1
    
    print(f"\nTeam Linking Results: {passed} passed, {failed} failed")
    return failed == 0


def test_player_linking(linker: EntityLinker):
    """Test player linking functionality."""
    print("\n" + "="*50)
    print("Testing Player Linking")
    print("="*50)
    
    test_cases = [
        ("Stephen Curry", "Full name"),
        ("Curry", "Last name only"),
        ("LeBron James", "Full name"),
        ("James", "Last name only"),
        ("Joel Embiid", "Full name"),
        ("Luka Doncic", "Full name"),
        ("Kevin Durant", "Full name"),
    ]
    
    passed = 0
    failed = 0
    
    for player_name, description in test_cases:
        try:
            player_id = linker.link_player(player_name)
            if player_id:
                # Verify by getting player info
                player_info = linker.api_service.get_player_by_id(player_id)
                if player_info:
                    full_name = f"{player_info['first_name']} {player_info['last_name']}"
                    print(f"✓ {description}: '{player_name}' -> ID {player_id} ({full_name})")
                    passed += 1
                else:
                    print(f"✗ {description}: '{player_name}' -> ID {player_id} (but player not found)")
                    failed += 1
            else:
                print(f"✗ {description}: '{player_name}' -> Not found")
                failed += 1
        except Exception as e:
            print(f"✗ {description}: '{player_name}' -> Error: {e}")
            failed += 1
    
    print(f"\nPlayer Linking Results: {passed} passed, {failed} failed")
    return failed == 0


def test_fuzzy_matching(linker: EntityLinker):
    """Test fuzzy matching for typos."""
    print("\n" + "="*50)
    print("Testing Fuzzy Matching")
    print("="*50)
    
    test_cases = [
        ("Laker", "Typo: 'Lakers' -> 'Laker'"),
        ("Warrior", "Typo: 'Warriors' -> 'Warrior'"),
        ("Steph Curry", "Variant: 'Stephen' -> 'Steph'"),
        ("Lebron James", "Variant: 'LeBron' -> 'Lebron'"),
    ]
    
    passed = 0
    failed = 0
    
    for entity_name, description in test_cases:
        try:
            # Try team first
            team_id = linker.link_team(entity_name)
            if team_id:
                team_info = linker.api_service.get_team_by_id(team_id)
                print(f"✓ {description}: '{entity_name}' -> Team ID {team_id} ({team_info['full_name']})")
                passed += 1
                continue
            
            # Try player
            player_id = linker.link_player(entity_name)
            if player_id:
                player_info = linker.api_service.get_player_by_id(player_id)
                full_name = f"{player_info['first_name']} {player_info['last_name']}"
                print(f"✓ {description}: '{entity_name}' -> Player ID {player_id} ({full_name})")
                passed += 1
            else:
                print(f"✗ {description}: '{entity_name}' -> Not found")
                failed += 1
        except Exception as e:
            print(f"✗ {description}: '{entity_name}' -> Error: {e}")
            failed += 1
    
    print(f"\nFuzzy Matching Results: {passed} passed, {failed} failed")
    return failed == 0


def test_generic_interface(linker: EntityLinker):
    """Test the generic link_entity interface."""
    print("\n" + "="*50)
    print("Testing Generic Interface")
    print("="*50)
    
    test_cases = [
        ("team", "Lakers", "Team via generic interface"),
        ("player", "Stephen Curry", "Player via generic interface"),
        ("team", "Warriors", "Team via generic interface"),
        ("player", "LeBron James", "Player via generic interface"),
    ]
    
    passed = 0
    failed = 0
    
    for entity_type, entity_name, description in test_cases:
        try:
            entity_id = linker.link_entity(entity_type, entity_name)
            if entity_id:
                print(f"✓ {description}: {entity_type} '{entity_name}' -> ID {entity_id}")
                passed += 1
            else:
                print(f"✗ {description}: {entity_type} '{entity_name}' -> Not found")
                failed += 1
        except Exception as e:
            print(f"✗ {description}: {entity_type} '{entity_name}' -> Error: {e}")
            failed += 1
    
    print(f"\nGeneric Interface Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests."""
    print("="*50)
    print("Entity Linker Test Suite")
    print("="*50)
    
    # Initialize services
    try:
        api_service = NBAApiService()
        linker = EntityLinker(api_service)
        print("\n✓ Entity linker initialized successfully")
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        return
    
    # Run tests
    results = []
    
    results.append(("Team Linking", test_team_linking(linker)))
    results.append(("Player Linking", test_player_linking(linker)))
    results.append(("Fuzzy Matching", test_fuzzy_matching(linker)))
    results.append(("Generic Interface", test_generic_interface(linker)))
    
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

