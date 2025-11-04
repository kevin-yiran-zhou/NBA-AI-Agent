import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def print_summary_report(y_true, y_pred, target_names):
    """Print classification report with only accuracy, macro avg, and weighted avg."""
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        digits=4, 
        zero_division=0,
        output_dict=True
    )
    accuracy = accuracy_score(y_true, y_pred)
    print(f"              precision    recall  f1-score   support\n")
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"macro avg        {macro['precision']:.4f}    {macro['recall']:.4f}    {macro['f1-score']:.4f}    {macro['support']:.0f}")
    if 'weighted avg' in report:
        weighted = report['weighted avg']
        print(f"weighted avg     {weighted['precision']:.4f}    {weighted['recall']:.4f}    {weighted['f1-score']:.4f}    {weighted['support']:.0f}")
    print(f"\naccuracy                                    {accuracy:.4f}    {len(y_true)}")

def load_data(path):
    """Load data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    texts = [ex["text"] for ex in data]
    intents = [ex["intent"] for ex in data]
    attributes = [ex["slots"]["attribute"] for ex in data]
    inputs = [ex["slots"]["input"] for ex in data]
    return texts, intents, attributes, inputs

if __name__ == "__main__":
    # Load all splits to get all possible classes (for encoder)
    train_texts, train_intents, train_attrs, train_inputs = load_data("../dataset/train.json")
    val_texts, val_intents, val_attrs, val_inputs = load_data("../dataset/val.json")
    test_texts, test_intents, test_attrs, test_inputs = load_data("../dataset/test.json")

    # Create encoders and fit on all data
    intent_encoder = LabelEncoder()
    attr_encoder = LabelEncoder()
    intent_encoder.fit(train_intents + val_intents + test_intents)
    attr_encoder.fit(train_attrs + val_attrs + test_attrs)

    # Transform test labels
    y_intent_test = intent_encoder.transform(test_intents)
    y_attr_test = attr_encoder.transform(test_attrs)
    
    # Get all unique input values for random selection
    all_inputs = list(set(train_inputs + val_inputs + test_inputs))
    num_inputs = len(all_inputs)

    # Get number of classes
    num_intents = len(intent_encoder.classes_)
    num_attrs = len(attr_encoder.classes_)

    print("="*60)
    print("Random Baseline on Test Set")
    print("="*60)
    print(f"Test samples: {len(test_texts)}")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Make random predictions
    preds_intent = []
    preds_attr = []
    preds_input = []
    
    for _ in test_texts:
        pred_intent = np.random.randint(0, num_intents)
        pred_attr = np.random.randint(0, num_attrs)
        pred_input = all_inputs[np.random.randint(0, num_inputs)]
        preds_intent.append(pred_intent)
        preds_attr.append(pred_attr)
        preds_input.append(pred_input)
    
    preds_intent = np.array(preds_intent)
    preds_attr = np.array(preds_attr)
    
    # Calculate input extraction accuracy (exact match)
    input_correct = sum(1 for pred, true in zip(preds_input, test_inputs) if pred == true)
    input_accuracy = input_correct / len(test_inputs)

    # Evaluate
    print("Intent Classification (Random Baseline):")
    print("="*60)
    print_summary_report(y_intent_test, preds_intent, intent_encoder.classes_)
    print()

    print("Attribute Classification (Random Baseline):")
    print("="*60)
    print_summary_report(y_attr_test, preds_attr, attr_encoder.classes_)
    print()

    print("Input Extraction (Random Baseline):")
    print("="*60)
    print(f"Accuracy (exact match): {input_accuracy:.4f} ({input_correct}/{len(test_inputs)})")
    print()