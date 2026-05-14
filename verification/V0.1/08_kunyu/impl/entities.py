from enum import IntEnum


class EntityType(IntEnum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    DANGER = 3
    AGENT = 4


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4
    WAIT = 5


ACTION_DELTA = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.EAT: (0, 0),
    Action.WAIT: (0, 0),
}

ACTION_NAME = {
    Action.UP: "UP",
    Action.DOWN: "DOWN",
    Action.LEFT: "LEFT",
    Action.RIGHT: "RIGHT",
    Action.EAT: "EAT",
    Action.WAIT: "WAIT",
}
