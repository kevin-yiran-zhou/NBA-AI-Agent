from bert import BertPredictor

# Initialize predictor
predictor = BertPredictor()

# Interactive loop
while True:
    text = input("🗣️ You: ").strip()
    if text.lower() in ["exit", "quit"]:
        break
    
    result = predictor.predict(text)
    print(f"🤖 Intent: {result['intent']} | Attribute: {result['attr']} | Input: {result['input']}")
    print(f"⏱️ BERT: {result['bert_ms']:.2f} ms | spaCy: {result['spacy_ms']:.2f} ms\n")