import json, random

templates = {
    "position": [
        "What position does {player} play?",
        "Which position is {player} known for?",
        "Tell me {player}'s playing position.",
        "I want to know {player}'s position.",
        "Which position does {player} play?"
    ],
    "height": [
        "How tall is {player}?",
        "What's {player}'s height?",
        "Can you tell me how tall {player} is?",
        "Tell me {player}'s height in feet.",
        "I want to know {player}'s height."
    ],
    "weight": [
        "How much does {player} weigh?",
        "What's {player}'s weight?",
        "Tell me {player}'s weight in pounds.",
        "Can you tell me how heavy {player} is?",
        "I want to know {player}'s weight."
    ],
    "jersey_number": [
        "What number does {player} wear?",
        "Which jersey number does {player} use?",
        "Tell me {player}'s jersey number.",
        "I want to know {player}'s jersey number.",
        "Which jersey number does {player} wear?"
    ],
    "college": [
        "Which college did {player} attend?",
        "Where did {player} play college basketball?",
        "Tell me {player}'s college.",
        "I want to know {player}'s college.",
        "Which college did {player} play?"
    ],
    "country": [
        "What country is {player} from?",
        "Where is {player} from?",
        "Which country does {player} represent?",
        "I want to know {player}'s country.",
        "Which country does {player} represent?"
    ],
    "draft_year": [
        "When was {player} drafted?",
        "In what year was {player} drafted into the NBA?",
        "Tell me {player}'s draft year.",
        "I want to know {player}'s draft year.",
        "Which year was {player} drafted?"
    ],
    "draft_round": [
        "What round was {player} drafted in?",
        "Which draft round was {player} selected in?",
        "Tell me {player}'s draft round.",
        "I want to know {player}'s draft round.",
        "Which draft round was {player} selected in?"
    ],
    "draft_number": [
        "What pick number was {player} in the draft?",
        "At what pick was {player} selected?",
        "Tell me {player}'s draft pick number.",
        "I want to know {player}'s draft pick number.",
        "Which pick was {player} selected in?"
    ],
    "team": [
        "Which team does {player} play for?",
        "Who does {player} play for right now?",
        "Tell me {player}'s current NBA team.",
        "I want to know {player}'s current NBA team.",
        "Which team does {player} play for?",
        "Which team did {player} play for?",
        "Tell me {player}'s previous NBA team.",
        "Which team did {player} last play for?",
        "Tell me {player}'s last NBA team.",
        "Which team did {player} last play for?",
    ]
}

players = [
    "Stephen Curry", "LeBron James", "Giannis Antetokounmpo", "Luka Doncic",
    "Kevin Durant", "Nikola Jokic", "Jayson Tatum", "Devin Booker",
    "Jimmy Butler", "Joel Embiid", "Kyrie Irving", "Anthony Davis",
    "Damian Lillard", "Ja Morant", "Trae Young", "Donovan Mitchell",
    "Shai Gilgeous-Alexander", "Zion Williamson", "Jaylen Brown", "Jalen Brunson"
]

dataset = []
for attr, tmpl_list in templates.items():
    for player in players:
        for t in random.sample(tmpl_list, 3):
            dataset.append({
                "text": t.format(player=player),
                "intent": "player_info",
                "slots": {"player_name": player, "attribute": attr}
            })

with open("player_info.json", "w") as f:
    json.dump(dataset, f, indent=2)
