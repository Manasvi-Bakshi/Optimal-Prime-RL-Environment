class Environment:
    def __init__(self):
        self.state = {"step": 0}

    def reset(self):
        self.state = {"step": 0}
        return self.state

    def step(self, action):
        self.state["step"] += 1
        done = self.state["step"] >= 3

        return {
            "state": self.state,
            "done": done
        }

    def get_state(self):
        return self.state