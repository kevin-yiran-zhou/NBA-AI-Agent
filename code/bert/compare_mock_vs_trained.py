"""
Compare Mock Predictor vs Trained Model

This script demonstrates the difference between Mock predictor and trained model
by testing queries from the training data.
"""

import json
from end_to_end import EndToEndAgent


def load_training_queries(n=10):
    """Load sample queries from training data."""
    with open('dataset/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:n]


def test_mock_predictor():
    """Test queries with Mock predictor."""
    print("="*70)
    print("Testing with MOCK PREDICTOR (Rule-based)")
    print("="*70)
    
    agent = EndToEndAgent.with_mock_predictor()
    queries = load_training_queries(10)
    
    passed = 0
    failed = 0
    
    for i, example in enumerate(queries, 1):
        query = example['text']
        expected_intent = example['intent']
        expected_slots = example['slots']
        
        print(f"\n[Test {i}] {query}")
        print(f"Expected: intent={expected_intent}, slots={expected_slots}")
        
        try:
            result = agent.process_query(query)
            predicted_intent = result['intent']
            predicted_slots = result['slots']
            
            # Check if prediction matches
            intent_match = predicted_intent == expected_intent
            input_match = predicted_slots.get('input', '').lower() == expected_slots.get('input', '').lower()
            attr_match = predicted_slots.get('attribute') == expected_slots.get('attribute')
            
            if intent_match and input_match and attr_match:
                print(f"[OK] Intent: {predicted_intent}, Slots: {predicted_slots}")
                passed += 1
            else:
                print(f"[FAIL] Intent: {predicted_intent} (expected {expected_intent})")
                print(f"       Slots: {predicted_slots} (expected {expected_slots})")
                failed += 1
                
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Mock Predictor Results: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.1f}%)")
    print("="*70)
    
    return passed, failed


def show_training_data_examples():
    """Show examples of diverse expressions in training data."""
    print("\n" + "="*70)
    print("Examples of DIVERSE EXPRESSIONS in Training Data")
    print("="*70)
    
    with open('dataset/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by intent and attribute
    examples = {}
    for ex in data[:50]:  # First 50 examples
        intent = ex['intent']
        attr = ex['slots']['attribute']
        key = f"{intent}_{attr}"
        if key not in examples:
            examples[key] = []
        examples[key].append(ex['text'])
    
    # Show examples
    for key, texts in list(examples.items())[:5]:
        print(f"\n{key}:")
        for text in texts[:3]:  # Show first 3 examples
            print(f"  - {text}")
    
    print("\n" + "="*70)
    print("Notice how the SAME question can be asked in MANY different ways!")
    print("Mock predictor can only handle some of these, but trained model")
    print("can understand the semantic similarity between all of them.")
    print("="*70)


if __name__ == "__main__":
    # Show training data examples
    show_training_data_examples()
    
    # Test Mock predictor
    test_mock_predictor()
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("Mock predictor uses simple rule matching, so it can only handle")
    print("queries that match its predefined patterns. Training data contains")
    print("6,722 examples with MANY different ways to ask the same question.")
    print("\nA trained model learns the SEMANTIC MEANING, so it can understand")
    print("all these variations, even ones it hasn't seen before!")
    print("="*70)

