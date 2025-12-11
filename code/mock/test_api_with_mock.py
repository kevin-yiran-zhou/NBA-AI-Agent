import sys
import os
# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from end_to_end import EndToEndAgent
from API.api_service import NBAApiService
from API.entity_linker import EntityLinker
from API.api_router import APIRouter
from API.response_formatter import ResponseFormatter


def test_api_with_mock():
    """Test API integration using mock predictor."""
    print("="*60)
    print("API Integration Test (Using Mock Predictor)")
    print("="*60)
    print("\nThis test uses a mock predictor instead of a trained model.")
    print("This allows testing the API integration without training.\n")
    
    try:
        # Initialize services
        print("[1/4] Initializing API Service...")
        api_service = NBAApiService()
        
        print("[2/4] Initializing Entity Linker...")
        entity_linker = EntityLinker(api_service)
        
        print("[3/4] Initializing API Router...")
        api_router = APIRouter(api_service, entity_linker)
        
        print("[4/4] Initializing Response Formatter...")
        response_formatter = ResponseFormatter()
        
        print("\n[Creating] Creating agent with mock predictor...")
        agent = EndToEndAgent.with_mock_predictor(
            api_service=api_service,
            entity_linker=entity_linker,
            api_router=api_router,
            response_formatter=response_formatter
        )
        print("✓ Agent created successfully!\n")
        
        # Test queries
        test_queries = [
            "Which conference are the Lakers in?",
            "What position does Stephen Curry play?",
            "How tall is LeBron James?",
            "What's the abbreviation for the Warriors?",
            "Which division are the Celtics in?",
            "What's Joel Embiid's weight?",
            "Tell me the city of the Heat",
            "What college did Luka Doncic attend?",
        ]
        
        print("="*60)
        print("Testing API Integration")
        print("="*60)
        
        passed = 0
        failed = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Test {i}] Query: {query}")
            print("-" * 60)
            
            try:
                result = agent.process_query(query)
                
                print(f"Intent: {result['intent']}")
                print(f"Slots: {result['slots']}")
                
                if result['api_result']['success']:
                    print(f"✓ API Call: Success")
                    print(f"Response: {result['formatted_response']}")
                    passed += 1
                else:
                    print(f"✗ API Call: Failed")
                    print(f"Error: {result['api_result'].get('error', 'Unknown error')}")
                    failed += 1
                
            except Exception as e:
                print(f"✗ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        print("\n" + "="*60)
        print("Interactive Mode")
        print("="*60)
        print("Enter queries to test the API integration (type 'exit' to quit):\n")
        
        while True:
            try:
                query = input("🗣️  You: ").strip()
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                if not query:
                    continue
                
                result = agent.process_query(query)
                print(f"🤖 {result['formatted_response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"✗ Error: {e}\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_api_with_mock()

