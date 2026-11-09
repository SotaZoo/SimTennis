"""Short competitive episodes; native tennis rules still decide every point."""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np


@dataclass
class ContestScore:
    """First to three, no advantage requirement; at most five physical points.

    This is a training format, not a regulation tennis set. Each player serves
    in turn and alternates their own service half. Ends stay fixed per contest.
    """
    first_server: int = 0
    ends: int = 0
    first_deuce: bool = True
    points: list = field(default_factory=lambda: [0, 0])
    winner: int = -1

    def __post_init__(self):
        if self.first_server not in (0, 1) or self.ends not in (0, 1):
            raise ValueError('Server and ends must identify the two players')

    @property
    def server(self):
        return (self.first_server + sum(self.points)) % 2

    def slot(self, player):
        return player ^ self.ends

    def deuce_court(self):
        return self.first_deuce == ((sum(self.points)//2) % 2 == 0)

    def point_text(self, player):
        return str(self.points[player])

    def award(self, player):
        if self.winner != -1 or player not in (0, 1):
            return False
        self.points[player] += 1
        if self.points[player] == 3:
            self.winner = player
        return True

