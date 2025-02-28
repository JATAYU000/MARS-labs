import json

def remove_n_a_entries(data):
    """Recursively removes all 'N/A' keys and values from the given dictionary."""
    if isinstance(data, dict):
        return {k: remove_n_a_entries(v) for k, v in data.items() if k != "N/A" and v != "N/A"}
    elif isinstance(data, list):
        return [remove_n_a_entries(item) for item in data]
    else:
        return data

# Read JSON file as a dictionary
with open("MARS-labs/Misc/olabs_data.json", "r", encoding="utf-8") as file:
    json_text = file.read()  # Read file content as a string
    json_dict = eval(json_text)  # Convert string to dictionary (alternative to json.load())

# Clean JSON data
cleaned_dict = remove_n_a_entries(json_dict)

# Convert dictionary back to JSON and save it
with open("MARS-labs/Misc/olabs_data.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_dict, file, indent=4)  # Dump dictionary as JSON

print("Updated JSON file successfully!")
