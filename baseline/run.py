import requests

BASE_URL = "http://localhost:7860"

def run():
    requests.get(f"{BASE_URL}/reset")

    scores = []

    for _ in range(3):
        requests.post(f"{BASE_URL}/step", json={
            "action_type": "noop",
            "payload": {}
        })

    result = requests.post(f"{BASE_URL}/grader").json()
    scores.append(result["score"])

    print("Scores:", scores)

if __name__ == "__main__":
    run()