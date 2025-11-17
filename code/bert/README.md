# NBA-AI-Agent

An AI agent that understands natural-language NBA questions and fetches answers from the Ball Don’t Lie API.

## 📋 Overview

The system has two major parts:

1. **Training pipeline** – fine-tunes a BERT model for intent classification and slot filling.
2. **API integration** – turns the model output into NBA API calls and formats the responses.

## 🏗️ Project Structure

```
NBA-AI-Agent/
├── Training
│   ├── train_bert.py          # Model training
│   ├── test_bert.py           # Inference-only sanity checks
│   └── dataset/               # Dataset builders and JSON files
│
├── API Integration
│   ├── api_service.py         # Wrapper for Ball Don’t Lie API
│   ├── entity_linker.py       # Name → ID entity linking
│   ├── api_router.py          # Maps intents/slots to API calls
│   ├── response_formatter.py  # Natural-language responses
│   ├── end_to_end.py          # Full pipeline orchestration
│   └── mock_predictor.py      # Rule-based predictor for testing
│
└── Tests
    ├── test_api_service.py
    ├── test_entity_linker.py
    ├── test_api_router.py
    ├── test_end_to_end.py     # Requires a trained model
    └── test_api_with_mock.py  # Uses the mock predictor
```

See `PROJECT_STRUCTURE.md` for a more detailed breakdown.

## 🚀 Getting Started

### Prerequisites

1. Install dependencies:
```bash
pip install torch transformers scikit-learn balldontlie
```

2. Configure your API key:
   - Create `API_KEY.txt`
   - Paste your Ball Don’t Lie API key inside

### Option 1: Full pipeline with a trained model

```python
from end_to_end import EndToEndAgent

agent = EndToEndAgent.from_model_dir("models/bert_multi")
result = agent.process_query("Which conference are the Lakers in?")
print(result["formatted_response"])
```

> Requires running `train_bert.py` beforehand.

### Option 2: API testing with the mock predictor

```python
from end_to_end import EndToEndAgent

agent = EndToEndAgent.with_mock_predictor()
result = agent.process_query("Which conference are the Lakers in?")
print(result["formatted_response"])
```

Or run the convenience script:
```bash
python test_api_with_mock.py
```