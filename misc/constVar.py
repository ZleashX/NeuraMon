from poke_env.data import GenData

GEN_4_DATA = GenData.from_gen(4)
TEAM_SIZE = 6
NUM_TYPES = 20
MAX_MOVE_ID = 196 + 1
MAX_SPECIES_ID = 268 + 1
MAX_ABILITY_ID = 102 + 1
NUM_BOOST_TYPES = 7 + 1
NUM_STATUS_TYPES = 7 + 1
NUM_WEATHER_TYPES = 9 + 1
NUM_FIELD_TYPES = 13 + 1
NUM_SIDE_CONDITIONS = 24 + 1
NUM_EFFECT_TYPES = 224 + 1
MAX_ITEM_ID = 40 + 1
STAT_TO_IDX = {
    "accuracy": 0,
    "atk": 1,
    "def": 2,
    "evasion": 3,
    "spa": 4,
    "spd": 5,
    "spe": 6,
}