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
from bert.bert import BertPredictor
import time


def test_API_with_bert():
    """Test the complete end-to-end pipeline."""
    print("="*60)
    print("End-to-End Agent Test with BERT")
    print("="*60)
    
    # Set up paths
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
        # Also load BertPredictor for timing
        bert_predictor = BertPredictor(model_dir=model_dir)
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
                
                model_start = time.perf_counter()
                bert_result = bert_predictor.predict(query, extract_entity=True, preprocess=True)
                model_time_ms = bert_result['bert_ms'] + bert_result.get('spacy_ms', 0.0)
                api_start = time.perf_counter()
                result = agent.process_query(query)
                api_time_ms = (time.perf_counter() - api_start) * 1000.0
                
                print(f"🤖 {result['formatted_response']}")
                print(f"⏱️  Model: {model_time_ms:.2f}ms | API: {api_time_ms:.2f}ms\n")
                
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

