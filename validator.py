import requests

BASE = "http://localhost:8000"


def check():
    session = requests.Session()

    try:
        # RESET
        res = session.post(f"{BASE}/reset", timeout=10)
        assert res.status_code == 200, f"/reset failed: {res.text}"

        # TASKS
        res = session.get(f"{BASE}/tasks", timeout=10)
        assert res.status_code == 200, f"/tasks failed: {res.text}"

        # STEP
        payload = {"action": {"priority_ratio": 0.5}}
        res = session.post(f"{BASE}/step", json=payload, timeout=10)
        assert res.status_code == 200, f"/step failed: {res.text}"

        data = res.json()
        assert "reward" in data, "step missing reward"
        assert "done" in data, "step missing done"

        # GRADER
        res = session.post(
            f"{BASE}/grader",
            json={"rewards": [-1.0, -2.0, -3.0]},
            timeout=10
        )
        assert res.status_code == 200, f"/grader failed: {res.text}"

        data = res.json()
        score = data.get("score", None)
        assert score is not None, "grader missing score"
        assert 0.0 <= score <= 1.0, "score out of bounds"

        print("All checks passed")

    finally:
        session.close()


if __name__ == "__main__":
    check()