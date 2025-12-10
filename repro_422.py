import requests
import json

print("Testing API reproduction with edge cases...")

try:
    with open("data/sample_cml_data.csv", "rb") as f:
        file_content = f.read()

    print("\n--- TEST 1: Empty Content Type ---")
    files = {"file": ("sample_cml_data.csv", file_content, "")}
    response = requests.post("http://localhost:8000/score-cml-data", files=files)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print(json.dumps(response.json(), indent=2))

    print("\n--- TEST 2: None Content Type ---")
    files = {"file": ("sample_cml_data.csv", file_content, None)}
    response = requests.post("http://localhost:8000/score-cml-data", files=files)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print(json.dumps(response.json(), indent=2))

    print("\n--- TEST 3: Invalid Name ---")
    files = {"file": ("", file_content, "text/csv")}
    response = requests.post("http://localhost:8000/score-cml-data", files=files)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print(json.dumps(response.json(), indent=2))

except Exception as e:
    print(f"Script Error: {e}")
