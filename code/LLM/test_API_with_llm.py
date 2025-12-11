import os
import sys
# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from llm_predictor import LLMPredictor
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_predictor import LLMPredictor
from API.api_service import NBAApiService
from API.entity_linker import EntityLinker
from API.api_router import APIRouter
from API.response_formatter import ResponseFormatter


class LLMEndToEndAgent:
    """End-to-end agent using LLM predictor."""
    
    def __init__(self, llm_predictor: LLMPredictor,
                 api_service: NBAApiService = None,
                 entity_linker: EntityLinker = None,
                 api_router: APIRouter = None,
                 response_formatter: ResponseFormatter = None):
        self.llm_predictor = llm_predictor
        
        # Initialize services
        if api_service is None:
            api_service = NBAApiService()
        if entity_linker is None:
            entity_linker = EntityLinker(api_service)
        if api_router is None:
            api_router = APIRouter(api_service, entity_linker)
        if response_formatter is None:
            response_formatter = ResponseFormatter()
        
        self.api_service = api_service
        self.entity_linker = entity_linker
        self.api_router = api_router
        self.response_formatter = response_formatter
    
    def process_query(self, user_query: str):
        """Process query through end-to-end pipeline."""
        import time
        
        # Step 1: LLM prediction
        llm_result = self.llm_predictor.predict(user_query)
        llm_time_ms = llm_result['llm_ms']
        
        intent = llm_result['intent']
        slots = {
            "input": llm_result['input'],
            "attribute": llm_result['attr']
        }
        
        # Step 2: Entity Linking
        linked_slots = slots.copy()
        
        # Step 3: API call (with timing)
        api_start = time.perf_counter()
        api_result = self.api_router.route(intent, linked_slots)
        api_time_ms = (time.perf_counter() - api_start) * 1000.0
        
        # Step 4: Format response
        formatted_response = self.response_formatter.format(intent, api_result, slots)
        
        return {
            "intent": intent,
            "slots": slots,
            "linked_slots": linked_slots,
            "api_result": api_result,
            "formatted_response": formatted_response,
            "llm_result": llm_result,
            "llm_time_ms": llm_time_ms,
            "api_time_ms": api_time_ms
        }


def test_API_with_llm():
    """Test the complete end-to-end pipeline with LLM."""
    print("="*60)
    print("End-to-End Agent Test with LLM")
    print("="*60)
    
    try:
        # Initialize services
        print("\n[1/5] Initializing API Service...")
        api_service = NBAApiService()
        
        print("[2/5] Initializing Entity Linker...")
        entity_linker = EntityLinker(api_service)
        
        print("[3/5] Initializing API Router...")
        api_router = APIRouter(api_service, entity_linker)
        
        print("[4/5] Initializing Response Formatter...")
        response_formatter = ResponseFormatter()
        
        print("[5/5] Loading LLM model...")
        llm_predictor = LLMPredictor(model_name="Qwen/Qwen3-4B-Instruct-2507-FP8")
        
        # Create agent
        agent = LLMEndToEndAgent(
            llm_predictor=llm_predictor,
            api_service=api_service,
            entity_linker=entity_linker,
            api_router=api_router,
            response_formatter=response_formatter
        )
        print("✓ LLM model loaded successfully!\n")
        
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
                print(f"🤖 {result['formatted_response']}")
                print(f"⏱️  Model: {result['llm_time_ms']:.2f}ms | API: {result['api_time_ms']:.2f}ms\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"✗ Error: {e}\n")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_API_with_llm()

