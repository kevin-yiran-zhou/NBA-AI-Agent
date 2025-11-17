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

## 📚 Documentation

- `PROJECT_STRUCTURE.md` – explains the separation between training and API layers.
- `API_TESTING_GUIDE.md` – shows how to validate the API stack in isolation.
- `API_TASKS.md` – task checklist for API development.

## 🔄 Development Flow

### Phase 1: Parallel work

**Training pipeline**
```bash
python dataset/build_dataset_team.py
python dataset/build_dataset_player.py
python dataset/combine_dataset.py
python train_bert.py
```

**API stack**
```bash
python test_api_service.py
python test_entity_linker.py
python test_api_router.py
python test_api_with_mock.py
```

### Phase 2: Integration tests

After training finishes:
```bash
python test_end_to_end.py
```

## 🎯 Capabilities

- **Team info**: conference, division, abbreviation, city, etc.
- **Player info**: position, height, weight, college, etc.
- **Game lookup**: by date or by game ID.

Processing pipeline:

```
User query
  ↓
Intent/slot prediction
  ↓
Entity linking (names → IDs)
  ↓
API routing + call
  ↓
Response formatting
  ↓
Final answer
```

## 📊 Current Status

### ✅ Completed
- API modules
- Mock predictor for API-only tests
- Automated test scripts
- Documentation

### ⏳ In progress
- Model training and tuning
- End-to-end validation with the trained model

## 💡 Notes

1. The API stack can be tested without a trained model via the mock predictor.
2. The mock predictor is only for development convenience; deploy with the trained model for reliable semantic coverage.
3. Clear boundaries between training and API layers make the codebase easier to extend.

## 📝 License

Refer to `LICENSE`.

## 🤝 Contributing

Issues and pull requests are welcome!