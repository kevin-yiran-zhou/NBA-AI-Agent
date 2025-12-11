import json
import os
import sys
import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
try:
    from bert import BertPredictor
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bert import BertPredictor

class Tee:
    """Write to both file and stdout."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

def load_data(path):
    """Load data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    texts = [ex["text"] for ex in data]
    intents = [ex["intent"] for ex in data]
    attributes = [ex["slots"]["attribute"] for ex in data]
    entities = [ex["slots"].get("entity", "Unknown") for ex in data]
    return texts, intents, attributes, entities

def print_summary_report(y_true, y_pred, target_names, task_name):
    """Print a summary classification report."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    
    print(f"\n{'='*60}")
    print(f"{task_name} - Summary")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nPer-class metrics:")
    for label in target_names:
        if label in report:
            metrics = report[label]
            print(f"  {label:20s} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    print(f"\nMacro avg - Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted avg - Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1: {report['weighted avg']['f1-score']:.4f}")

if __name__ == "__main__":
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "dataset")
    models_dir = os.path.join(project_root, "models", "bert_multi")
    
    # Set up output file
    output_file = os.path.join(script_dir, "test_bert_output.txt")
    tee = Tee(output_file)
    sys.stdout = tee
    
    try:
        # Load test data
        test_path = os.path.join(dataset_dir, "test_with_names.json")
        if not os.path.exists(test_path):
            print("❌ Error: test_with_names.json not found!")
            exit(1)
    
        test_texts, test_intents, test_attrs, test_entities = load_data(test_path)
        
        # Load encoders (same as used in training)
        intent_encoder = LabelEncoder()
        attr_encoder = LabelEncoder()
        intent_encoder.classes_ = np.load(f"{models_dir}/intent_encoder.npy", allow_pickle=True)
        attr_encoder.classes_ = np.load(f"{models_dir}/attr_encoder.npy", allow_pickle=True)
        
        # Load model
        print(f"📦 Loading model from {models_dir}...")
        predictor = BertPredictor(model_dir=models_dir)
        
        # Predict on test set
        print(f"\n🔮 Testing on {len(test_texts)} examples...\n")
        pred_intents = []
        pred_attrs = []
        pred_entities = []
        total_bert_time = 0.0
        total_spacy_time = 0.0
        
        for i, text in enumerate(test_texts):
            result = predictor.predict(text, extract_entity=True, preprocess=False)
            pred_intents.append(result['intent'])
            pred_attrs.append(result['attr'])
            pred_entities.append(result['input'])
            total_bert_time += result.get('bert_ms', 0.0)
            total_spacy_time += result.get('spacy_ms', 0.0)
            
            # Check correctness
            intent_correct = result['intent'] == test_intents[i]
            attr_correct = result['attr'] == test_attrs[i]
            entity_correct = result['input'].lower() == test_entities[i].lower()
            correct = "✅" if (intent_correct and attr_correct and entity_correct) else "❌"
            
            # Print details
            print(f"{'='*80}")
            print(f"Example {i+1}/{len(test_texts)}")
            print(f"Input text: {text}")
            print(f"Extracted - Intent: {result['intent']}, Attribute: {result['attr']}, Name: {result['input']}")
            print(f"Expected - Intent: {test_intents[i]}, Attribute: {test_attrs[i]}, Name: {test_entities[i]}")
            print(f"Result: {correct} ({'Correct' if intent_correct and attr_correct and entity_correct else 'Wrong'})")
            print()
        
        # Convert to encoded labels for evaluation
        true_intents_encoded = intent_encoder.transform(test_intents)
        true_attrs_encoded = attr_encoder.transform(test_attrs)
        pred_intents_encoded = intent_encoder.transform(pred_intents)
        pred_attrs_encoded = attr_encoder.transform(pred_attrs)
        
        # Print results
        print_summary_report(true_intents_encoded, pred_intents_encoded, intent_encoder.classes_, "Intent Classification")
        print_summary_report(true_attrs_encoded, pred_attrs_encoded, attr_encoder.classes_, "Attribute Classification")
        
        # Detailed reports
        print(f"\n{'='*60}")
        print("Intent Classification - Detailed Report")
        print(f"{'='*60}")
        print(classification_report(true_intents_encoded, pred_intents_encoded, target_names=intent_encoder.classes_, digits=4, zero_division=0))
        
        print(f"\n{'='*60}")
        print("Attribute Classification - Detailed Report")
        print(f"{'='*60}")
        print(classification_report(true_attrs_encoded, pred_attrs_encoded, target_names=attr_encoder.classes_, digits=4, zero_division=0))
        
        # Entity extraction accuracy
        entity_matches = sum(1 for pred, true in zip(pred_entities, test_entities) if pred.lower() == true.lower())
        entity_accuracy = entity_matches / len(test_entities)
        
        print(f"\n{'='*60}")
        print("Entity Extraction (spaCy)")
        print(f"{'='*60}")
        print(f"Accuracy: {entity_accuracy:.4f} ({entity_matches}/{len(test_entities)})")
        
        # Timing summary
        total_time = total_bert_time + total_spacy_time
        print(f"\n⏱️  Timing Summary:")
        print(f"   Total BERT time: {total_bert_time:.2f} ms ({total_bert_time/1000:.2f} s)")
        print(f"   Total spaCy time: {total_spacy_time:.2f} ms ({total_spacy_time/1000:.2f} s)")
        print(f"   Total time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"   Average per example: {total_time/len(test_texts):.2f} ms")
    
    finally:
        # Restore stdout and close file
        sys.stdout = tee.stdout
        tee.close()
        print(f"\n✅ Output saved to: {output_file}")

