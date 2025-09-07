import json
import os
import re

def first_two_letters_cleaned(title: str) -> str:
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r'[^\w]', '', title, flags=re.UNICODE)
    return cleaned[:2].lower()

# Load the GeoJSON file
with open("./vertices.geojson", "r", encoding="utf-8") as f:
    vertices = json.load(f)

books_map_by_first_few_letters = {}

# Group features by first two letters of the title
for feature in vertices["features"]:
    title = feature["properties"]["title"]
    first_two_letters = first_two_letters_cleaned(title)
    coords = feature["geometry"]["coordinates"]
    size = feature["properties"]["size"]

    if first_two_letters not in books_map_by_first_few_letters:
        books_map_by_first_few_letters[first_two_letters] = []

    books_map_by_first_few_letters[first_two_letters].append(
        [title.lower(), coords[0], coords[1], size]
    )

# Ensure output folder exists
os.makedirs("./books_map_by_first_few_letters", exist_ok=True)

# Write each group to its own JSON file
for key, value in books_map_by_first_few_letters.items():
    value.sort(key=lambda x: x[3], reverse=True)
    # remove size from the list
    value = [x[:3] for x in value]
    with open(f"./books_map_by_first_few_letters/{key}.json", "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
