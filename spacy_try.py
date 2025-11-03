import spacy
nlp = spacy.load("en_core_web_trf")  # transformer version, very accurate

def extract_entity_spacy(text, intent):
    doc = nlp(text)
    if "player" in intent:
        ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    elif "team" in intent:
        ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
    else:
        ents = [ent.text for ent in doc.ents]
    return ents[0] if ents else "Unknown"

print(extract_entity_spacy("What is full name of the Bucks?", "team_info"))