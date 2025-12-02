"""
Test LLM model on test set with detailed output.
"""
import json
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
try:
    from llm_predictor import LLMPredictor
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_predictor import LLMPredictor

def load_data(path):
    """Load data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    texts = [ex["text"] for ex in data]
    intents = [ex["intent"] for ex in data]
    attributes = [ex["slots"]["attribute"] for ex in data]
    entities = [ex["slots"].get("entity", "Unknown") for ex in data]
    return texts, intents, attributes, entities

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "dataset")
    models_dir = os.path.join(project_root, "models", "bert_multi")
    
    test_path = os.path.join(dataset_dir, "test_with_names.json")
    if not os.path.exists(test_path):
        print("❌ Error: test_with_names.json not found!")
        exit(1)
    
    test_texts, test_intents, test_attrs, test_entities = load_data(test_path)
    
    intent_encoder = LabelEncoder()
    attr_encoder = LabelEncoder()
    intent_encoder.classes_ = np.load(f"{models_dir}/intent_encoder.npy", allow_pickle=True)
    attr_encoder.classes_ = np.load(f"{models_dir}/attr_encoder.npy", allow_pickle=True)
    
    print(f"📦 Loading LLM model...")
    predictor = LLMPredictor(model_name="Qwen/Qwen3-4B-Instruct-2507-FP8")
    
    print(f"\n🔮 Testing on {len(test_texts)} examples...\n")
    pred_intents = []
    pred_attrs = []
    pred_entities = []
    total_llm_time = 0.0
    
    for i, text in enumerate(test_texts):
        result = predictor.predict(text)
        pred_intents.append(result['intent'])
        pred_attrs.append(result['attr'])
        pred_entities.append(result['input'])
        total_llm_time += result['llm_ms']
        
        # Check correctness
        intent_correct = result['intent'] == test_intents[i]
        attr_correct = result['attr'] == test_attrs[i]
        entity_correct = result['input'].lower() == test_entities[i].lower()
        correct = "✅" if (intent_correct and attr_correct and entity_correct) else "❌"
        
        # Print details
        print(f"{'='*80}")
        print(f"Example {i+1}/{len(test_texts)}")
        print(f"Input text: {text}")
        print(f"LLM response: {result['raw_response']}")
        print(f"Extracted - Intent: {result['intent']}, Attribute: {result['attr']}, Name: {result['input']}")
        print(f"Expected - Intent: {test_intents[i]}, Attribute: {test_attrs[i]}, Name: {test_entities[i]}")
        print(f"Result: {correct} ({'Correct' if intent_correct and attr_correct and entity_correct else 'Wrong'})")
        print()
    
    # Timing summary will be shown at the end
    
    # Evaluation
    true_intents_encoded = intent_encoder.transform(test_intents)
    true_attrs_encoded = attr_encoder.transform(test_attrs)
    pred_intents_encoded = intent_encoder.transform(pred_intents)
    pred_attrs_encoded = attr_encoder.transform(pred_attrs)
    
    print(f"\n{'='*80}")
    print("Intent Classification")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy_score(true_intents_encoded, pred_intents_encoded):.4f}")
    print(classification_report(true_intents_encoded, pred_intents_encoded, target_names=intent_encoder.classes_, digits=4, zero_division=0))
    
    print(f"\n{'='*80}")
    print("Attribute Classification")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy_score(true_attrs_encoded, pred_attrs_encoded):.4f}")
    print(classification_report(true_attrs_encoded, pred_attrs_encoded, target_names=attr_encoder.classes_, digits=4, zero_division=0))
    
    # Entity extraction accuracy (already computed during loop)
    entity_matches = sum(1 for pred, true in zip(pred_entities, test_entities) if pred.lower() == true.lower())
    entity_accuracy = entity_matches / len(test_entities)
    
    print(f"\n{'='*80}")
    print("Entity Extraction")
    print(f"{'='*80}")
    print(f"Accuracy: {entity_accuracy:.4f} ({entity_matches}/{len(test_entities)})")
    
    # Timing summary
    print(f"\n⏱️  Timing Summary:")
    print(f"   Total time: {total_llm_time:.2f} ms ({total_llm_time/1000:.2f} s)")
    print(f"   Average per example: {total_llm_time/len(test_texts):.2f} ms")
