try:
    from .api_service import NBAApiService
    from .entity_linker import EntityLinker
    from .api_router import APIRouter
except ImportError:
    # Allow running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api_service import NBAApiService
    from entity_linker import EntityLinker
    from api_router import APIRouter


def test_team_info():
    """Test team_info intent handling."""
    print("="*50)
    print("Testing Team Info Intent")
    print("="*50)
    
    try:
        api_service = NBAApiService()
        entity_linker = EntityLinker(api_service)
        router = APIRouter(api_service, entity_linker)
        
        test_cases = [
            {
                "slots": {"input": "Lakers", "attribute": "conference"},
                "description": "Get Lakers conference"
            },
            {
                "slots": {"input": "Warriors", "attribute": "division"},
                "description": "Get Warriors division"
            },
            {
                "slots": {"input": "Celtics", "attribute": "city"},
                "description": "Get Celtics city"
            },
            {
                "slots": {"input": "Heat", "attribute": "abbreviation"},
                "description": "Get Heat abbreviation"
            },
            {
                "slots": {"input": "Lakers", "attribute": "full_name"},
                "description": "Get Lakers full name"
            }
        ]
        
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            try:
                result = router.route("team_info", test_case["slots"])
                if result["success"]:
                    print(f"[OK] {test_case['description']}: {result['data']['value']}")
                    passed += 1
                else:
                    print(f"[FAIL] {test_case['description']}: {result.get('error', 'Unknown error')}")
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {test_case['description']}: Exception - {e}")
                failed += 1
        
        print(f"\nTeam Info Results: {passed} passed, {failed} failed")
        return passed, failed
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize router: {e}")
        return 0, 1


def test_player_info():
    """Test player_info intent handling."""
    print("\n" + "="*50)
    print("Testing Player Info Intent")
    print("="*50)
    
    try:
        api_service = NBAApiService()
        entity_linker = EntityLinker(api_service)
        router = APIRouter(api_service, entity_linker)
        
        test_cases = [
            {
                "slots": {"input": "Stephen Curry", "attribute": "position"},
                "description": "Get Stephen Curry position"
            },
            {
                "slots": {"input": "LeBron James", "attribute": "height"},
                "description": "Get LeBron James height"
            },
            {
                "slots": {"input": "Joel Embiid", "attribute": "weight"},
                "description": "Get Joel Embiid weight"
            },
            {
                "slots": {"input": "Luka Doncic", "attribute": "jersey_number"},
                "description": "Get Luka Doncic jersey number"
            },
            {
                "slots": {"input": "Kevin Durant", "attribute": "country"},
                "description": "Get Kevin Durant country"
            }
        ]
        
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            try:
                result = router.route("player_info", test_case["slots"])
                if result["success"]:
                    print(f"[OK] {test_case['description']}: {result['data']['value']}")
                    passed += 1
                else:
                    print(f"[FAIL] {test_case['description']}: {result.get('error', 'Unknown error')}")
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {test_case['description']}: Exception - {e}")
                failed += 1
        
        print(f"\nPlayer Info Results: {passed} passed, {failed} failed")
        return passed, failed
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize router: {e}")
        return 0, 1


def test_game_lookup():
    """Test game_lookup intent handling."""
    print("\n" + "="*50)
    print("Testing Game Lookup Intent")
    print("="*50)
    
    try:
        api_service = NBAApiService()
        entity_linker = EntityLinker(api_service)
        router = APIRouter(api_service, entity_linker)
        
        test_cases = [
            {
                "slots": {"date": "2023-11-24"},
                "description": "Get games on 2023-11-24"
            }
        ]
        
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            try:
                result = router.route("game_lookup", test_case["slots"])
                if result["success"]:
                    if "games" in result["data"]:
                        print(f"[OK] {test_case['description']}: Found {result['data']['count']} games")
                    else:
                        print(f"[OK] {test_case['description']}: Game found")
                    passed += 1
                else:
                    print(f"[FAIL] {test_case['description']}: {result.get('error', 'Unknown error')}")
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {test_case['description']}: Exception - {e}")
                failed += 1
        
        print(f"\nGame Lookup Results: {passed} passed, {failed} failed")
        return passed, failed
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize router: {e}")
        return 0, 1


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    try:
        api_service = NBAApiService()
        entity_linker = EntityLinker(api_service)
        router = APIRouter(api_service, entity_linker)
        
        test_cases = [
            {
                "intent": "team_info",
                "slots": {"input": "InvalidTeam", "attribute": "conference"},
                "description": "Invalid team name"
            },
            {
                "intent": "player_info",
                "slots": {"input": "InvalidPlayer", "attribute": "position"},
                "description": "Invalid player name"
            },
            {
                "intent": "team_info",
                "slots": {"input": "Lakers"},  # Missing attribute
                "description": "Missing attribute slot"
            },
            {
                "intent": "unknown_intent",
                "slots": {},
                "description": "Unknown intent"
            }
        ]
        
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            try:
                result = router.route(test_case["intent"], test_case["slots"])
                if not result["success"]:
                    print(f"[OK] {test_case['description']}: Correctly handled error")
                    passed += 1
                else:
                    print(f"[FAIL] {test_case['description']}: Should have failed but succeeded")
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {test_case['description']}: Exception - {e}")
                failed += 1
        
        print(f"\nError Handling Results: {passed} passed, {failed} failed")
        return passed, failed
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize router: {e}")
        return 0, 1


def main():
    """Run all tests."""
    print("="*50)
    print("API Router Test Suite")
    print("="*50)
    
    results = []
    
    # Test team_info
    passed, failed = test_team_info()
    results.append(("Team Info", passed, failed))
    
    # Test player_info
    passed, failed = test_player_info()
    results.append(("Player Info", passed, failed))
    
    # Test game_lookup
    passed, failed = test_game_lookup()
    results.append(("Game Lookup", passed, failed))
    
    # Test error handling
    passed, failed = test_error_handling()
    results.append(("Error Handling", passed, failed))
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    total_passed = 0
    total_failed = 0
    
    for test_name, passed, failed in results:
        status = "[OK] PASSED" if failed == 0 else "[FAIL] FAILED"
        print(f"{test_name}: {status} ({passed} passed, {failed} failed)")
        total_passed += passed
        total_failed += failed
    
    print("\n" + "="*50)
    if total_failed == 0:
        print("[OK] All tests passed!")
    else:
        print(f"[FAIL] Some tests failed ({total_passed} passed, {total_failed} failed)")
    print("="*50)


if __name__ == "__main__":
    main()

