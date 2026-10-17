import random
import pandas as pd

data_templates = {
    "pothole_road_crack": [
        "There is a {size} pothole on the {place}",
        "Road has {count} potholes causing {problem}",
        "A deep crack in the road near {place} is dangerous",
        "The street surface is broken and has large holes",
        "Vehicles are slowing down due to road damage at {place}"
    ],

    "damaged_road_sign": [
        "The {sign} sign near {place} is damaged",
        "Road sign is bent and not visible to drivers",
        "Traffic sign at {place} is broken",
        "The direction board is unreadable due to damage",
        "Signboard near {place} has fallen down"
    ],

    "garbage": [
        "Garbage has not been collected near {place}",
        "Overflowing trash bins causing bad smell",
        "Waste dumped on roadside near {place}",
        "Piles of garbage attracting dogs and insects",
        "Uncollected waste blocking the footpath"
    ],

    "graffiti": [
        "Graffiti painted on wall near {place}",
        "Public wall vandalized with spray paint",
        "Offensive graffiti visible on building",
        "School wall is covered with graffiti",
        "Metro pillar has abusive graffiti"
    ],

    "illegal_parking": [
        "Vehicles parked illegally near {place}",
        "Cars blocking footpath due to illegal parking",
        "No-parking zone occupied by vehicles",
        "Double parking causing traffic jam",
        "Bike parked illegally blocking entrance"
    ]
}

places = [
    "bus stop", "school", "market", "junction",
    "hospital", "metro station", "main road",
    "residential area", "apartment gate"
]

signs = [
    "speed limit",
    "no parking",
    "direction",
    "stop",
    "traffic"
]

sizes = ["huge", "large", "deep", "dangerous"]
counts = ["multiple", "several", "many"]
problems = ["traffic jam", "accidents", "vehicle damage"]

rows = []

for label, templates in data_templates.items():
    for _ in range(100):  # 100 per class
        template = random.choice(templates)
        text = template.format(
            place=random.choice(places),
            size=random.choice(sizes),
            count=random.choice(counts),
            problem=random.choice(problems),
            sign=random.choice(signs)
    )
        rows.append({"text": text, "label": label})

df = pd.DataFrame(rows)
df.to_csv("complaints_text_dataset.csv", index=False)

print("✅ Dataset generated: complaints_text_dataset.csv")
