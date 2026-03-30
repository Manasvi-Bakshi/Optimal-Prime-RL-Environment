import requests

BASE = "http://localhost:7860"

def check():
    assert requests.get(f"{BASE}/reset").status_code == 200
    assert requests.get(f"{BASE}/tasks").status_code == 200

    score = requests.post(f"{BASE}/grader").json()["score"]
    assert 0.0 <= score <= 1.0

    print("All checks passed")

if __name__ == "__main__":
    check()