import json
import os
import re

# Load the GeoJSON file
with open("./vertices.geojson", "r", encoding="utf-8") as f:
    vertices = json.load(f)

unique_partition_ids_book_map = {}

# create map of one book for each partition id
for feature in vertices["features"]:
    if feature["properties"]["groupId"] not in unique_partition_ids_book_map:
        unique_partition_ids_book_map[feature["properties"]["groupId"]] = feature["properties"]["id"]
    else:
        pass

# find the file containing the book id by grep -r in edges folder
for partition_id, book_id in unique_partition_ids_book_map.items():
    # Use subprocess to capture the output
    import subprocess
    
    try:
        # Run grep command and capture output
        print("searching for ", book_id)
        result = subprocess.check_output(f'grep -rl \'"{book_id}"\' ./edges', shell=True, text=True).strip().replace("\n", "")
        if result:
            # Make sure the destination directory exists
            
            # Use subprocess.run instead of os.system for better error handling
            try:
                subprocess.run(["mv", result, f"./new_edges/{partition_id}.geojson"], check=True)
                print(f"Found {book_id} in {result} and moved to {partition_id}.geojson")
            except subprocess.CalledProcessError as e:
                print(f"Error moving file: {e}")
            except PermissionError:
                print(f"Permission denied when moving {result}")
        else:
            # This branch is unlikely to be reached as empty output would raise CalledProcessError
            print(f"No match found for {book_id}")
    except subprocess.CalledProcessError:
        # grep returns non-zero exit code when no matches are found
        print(f"No match found for {book_id}")

