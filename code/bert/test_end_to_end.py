"""
Test script for End-to-End Agent.
"""

import os
from end_to_end import EndToEndAgent
from api_service import NBAApiService
from entity_linker import EntityLinker
from api_router import APIRouter
from response_formatter import ResponseFormatter


def test_end_to_end():
    """Test the complete end-to-end pipeline."""
    print("="*60)
    print("End-to-End Agent Test")
    print("="*60)
    
    # Check if model exists
    model_dir = "models/bert_multi"
    if not os.path.exists(f"{model_dir}/model.pt"):
        print(f"[ERROR] Model not found at {model_dir}/model.pt")
        print("Please train the model first using train_bert.py")
        return
    
    try:
        # Initialize services
        print("\n[1/4] Initializing API Service...")
        api_service = NBAApiService()
        
        print("[2/4] Initializing Entity Linker...")
        entity_linker = EntityLinker(api_service)
        
        print("[3/4] Initializing API Router...")
        api_router = APIRouter(api_service, entity_linker)
        
        print("[4/4] Initializing Response Formatter...")
        response_formatter = ResponseFormatter()
        
        print("\n[Loading] Loading BERT model...")
        agent = EndToEndAgent.from_model_dir(
            model_dir=model_dir,
            api_service=api_service,
            entity_linker=entity_linker,
            api_router=api_router,
            response_formatter=response_formatter
        )
        print("✓ Model loaded successfully!\n")
        
        # Test queries
        test_queries = [
            "Which conference are the Lakers in?",
            "What position does Stephen Curry play?",
            "How tall is LeBron James?",
            "What's the abbreviation for the Warriors?",
            "Which division are the Celtics in?",
            "What's Joel Embiid's weight?",
        ]
        
        print("="*60)
        print("Testing End-to-End Pipeline")
        print("="*60)
        
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
                else:
                    print(f"✗ API Call: Failed")
                    print(f"Error: {result['api_result'].get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"✗ Error processing query: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("Interactive Mode")
        print("="*60)
        print("Enter queries to test the agent (type 'exit' to quit):\n")
        
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
    test_end_to_end()

