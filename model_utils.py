from enum import Enum


class GenerationMode(Enum):
    DONT_FORCE = 0
    FORCE_FREE = 1
    FORCE_STUCK = 2