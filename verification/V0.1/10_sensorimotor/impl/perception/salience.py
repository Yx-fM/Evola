"""Salience — computes how 'important' each type of fact is given the current drive."""


class SalienceMap:
    def __init__(self):
        self.food_relevance = 0.0
        self.danger_relevance = 0.0
        self.novelty_relevance = 0.0


def compute_salience(drive_energy: float, drive_safety: float,
                     drive_novelty: float) -> SalienceMap:
    sm = SalienceMap()
    sm.food_relevance = min(1.0, max(0.0, drive_energy))
    sm.danger_relevance = min(1.0, max(0.0, drive_safety))
    sm.novelty_relevance = min(1.0, max(0.0, drive_novelty))
    return sm
