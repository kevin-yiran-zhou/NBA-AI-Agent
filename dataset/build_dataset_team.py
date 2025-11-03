import json, random

templates = {
    "conference": [
        "Which conference are the {team} in?",
        "Are the {team} in the Western or Eastern Conference?",
        "What conference do the {team} play in?",
        "Tell me which conference the {team} belong to.",
        "I want to know the conference for the {team}."
    ],
    "division": [
        "Which division are the {team} in?",
        "What division do the {team} belong to?",
        "Tell me the division of the {team}.",
        "In what division do the {team} compete?",
        "I'd like to know which division the {team} are part of."
    ],
    "city": [
        "Which city are the {team} based in?",
        "Where are the {team} from?",
        "What city do the {team} represent?",
        "Tell me which city the {team} are located in.",
        "I want to know the home city of the {team}."
    ],
    "full_name": [
        "What is the full name of the {team}?",
        "Tell me the full name of the {team}.",
        "What's the complete team name for the {team}?",
        "Give me the full official name of the {team}.",
        "How do you say the full team name of the {team}?"
    ],
    "abbreviation": [
        "What is the abbreviation for the {team}?",
        "What's the team code for the {team}?",
        "Give me the short abbreviation of the {team}.",
        "Tell me the abbreviation used for the {team}.",
        "I want to know the three-letter code for the {team}."
    ]
}

teams = [
    "Warriors", "Lakers", "Celtics", "Bucks", "Heat",
    "Suns", "Nuggets", "Knicks", "76ers", "Mavericks",
    "Bulls", "Raptors", "Hawks", "Grizzlies", "Kings",
    "Cavaliers", "Pacers", "Pelicans", "Magic", "Spurs",
    "Thunder", "Jazz", "Timberwolves", "Pistons", "Wizards",
    "Nets", "Hornets", "Trail Blazers", "Clippers", "Rockets"
]

dataset = []
for attr, tmpl_list in templates.items():
    for team in teams:
        for t in random.sample(tmpl_list, 3):
            dataset.append({
                "text": t.format(team=team),
                "intent": "team_info",
                "slots": {"input": team, "attribute": attr}
            })

with open("team.json", "w") as f:
    json.dump(dataset, f, indent=2)
