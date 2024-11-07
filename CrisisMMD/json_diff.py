import json

def compare_json(json1, json2, path=""):
    differences = []
    
    # Check all keys in json1
    for key in json1:
        if key not in json2:
            differences.append(f"Key '{path + key}' found in first JSON but not in second JSON.")
        else:
            # If the key is found in both JSONs, compare values
            if isinstance(json1[key], dict) and isinstance(json2[key], dict):
                differences.extend(compare_json(json1[key], json2[key], path + key + "."))
            elif json1[key] != json2[key]:
                differences.append(f"Value of '{path + key}' differs. First JSON: {json1[key]}, Second JSON: {json2[key]}.")

    # Check for keys that are in json2 but not in json1
    for key in json2:
        if key not in json1:
            differences.append(f"Key '{path + key}' found in second JSON but not in first JSON.")

    return differences

# Load JSON from files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Example usage
json1 = load_json("crisismmd_datasplit_all/vocab_file_old.json")
json2 = load_json("crisismmd_datasplit_all/vocab_file_no_drop_last.json")

differences = compare_json(json1, json2)

if differences:
    print("Differences found between the two JSONs:")
    for diff in differences:
        print(diff)
else:
    print("The JSONs are identical.")
