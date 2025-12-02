class AgentState:
    """
    Simple container for the household state variables extracted
    from data + transformed by the economic model.
    """

    def __init__(self, wealth, health, investment, total_gain):
        self.wealth = wealth
        self.health = health
        self.investment = investment
        self.total_gain = total_gain

    def as_dict(self):
        return {
            "wealth": self.wealth,
            "health": self.health,
            "investment": self.investment,
            "total_gain": self.total_gain
        }

    def __repr__(self):
        return f"AgentState(wealth={self.wealth:.3f}, health={self.health:.3f})"
