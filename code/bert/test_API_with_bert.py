"""
Test script for End-to-End Agent with BERT model.
Tests the complete pipeline: BERT prediction -> Entity Linking -> API Call -> Response Formatting
"""
import os
import sys
# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from end_to_end import EndToEndAgent
from API.api_service import NBAApiService
from API.entity_linker import EntityLinker
from API.api_router import APIRouter
from API.response_formatter import ResponseFormatter


def test_API_with_bert():
    """Test the complete end-to-end pipeline."""
    print("="*60)
    print("End-to-End Agent Test with BERT")
    print("="*60)
    
    # Set up paths - script is in code/bert/, models are at project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_root, "models", "bert_multi")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, "model.pt")):
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
        
        print("="*60)
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
    test_API_with_bert()

