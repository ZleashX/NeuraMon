from poke_env.environment.move import Move, MoveCategory
from poke_env.environment.pokemon import Pokemon, PokemonType
from poke_env.environment.weather import Weather
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from misc.constVar import GEN_4_DATA

_FIRE_IMMUNITY = {"adaptability", "flashfire", "heatproof", "thickfat"}
_ICE_IMMUNITY = {"thickfat"}
_GROUND_IMMUNITY = {"levitate"}
_ELECTRIC_IMMUNITY = {"voltabsorb", "motordrive"}
_WATER_IMMUNITY = {"waterabsorb", "dryskin"}


def calculate_damage(move: Move, attacker: Pokemon, defender: Pokemon, weather: Weather, side_condition: SideCondition) -> float:
    """
    Calculate the damage dealt by a move from an attacker to a defender.
    """
    base_power = move.base_power
    
    if move.category == MoveCategory.PHYSICAL:
        attack_stat = _stat_estimation(attacker, "atk")
        defense_stat = _stat_estimation(defender, "def")
    elif move.category == MoveCategory.SPECIAL:
        attack_stat = _stat_estimation(attacker, "spa")
        defense_stat = _stat_estimation(defender, "spd")
    else:
        attack_stat = 1.0
        defense_stat = 1.0

    level = attacker.level

    if (move.category == MoveCategory.PHYSICAL and side_condition == SideCondition.REFLECT) or (move.category == MoveCategory.SPECIAL and side_condition == SideCondition.LIGHT_SCREEN):
        screen_mult = 0.5
    else:
        screen_mult = 1.0

    if attacker.status == Status.BRN and attacker.ability != "guts" and move.category == MoveCategory.PHYSICAL:
        burn_mult = 0.5
    else:
        burn_mult = 1.0

    weather_mult = _check_weather_effect(move, weather)
    stab_mult = 1.0
    if move.type in attacker.types:
        if attacker.ability == "adaptability":
            stab_mult = 2.0
        else:
            stab_mult = 1.5
    type_effectiveness = calculate_type_effectiveness(move, defender)

    damage = (((2 * level / 5 + 2) * base_power * (attack_stat / defense_stat) / 50) * 
            burn_mult * screen_mult * weather_mult + 2) * stab_mult * type_effectiveness
    
    norm_damage = max(0,min(1,damage / _hp_estimation(defender)))

    return norm_damage

def _hp_estimation(mon: Pokemon):
    """
    Estimate the HP of a Pokemon based on its base stats and IVs.
    """
    return (((2 * mon.base_stats["hp"] + 31 + (81/4)) + 100) * mon.level / 100) + 10

def _stat_estimation(mon: Pokemon, stat: str):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

def _check_weather_effect(move: Move, weather: Weather) -> float:
    """
    Check if the move is affected by the current weather.
    """
    if weather == Weather.SUNNYDAY:
        if move.type == PokemonType.FIRE:
            return 1.5
        elif move.type == PokemonType.WATER:
            return 0.5
    elif weather == Weather.RAINDANCE:
        if move.type == PokemonType.WATER:
            return 1.5
        elif move.type == PokemonType.FIRE:
            return 0.5
    else:
        return 1.0

def calculate_type_effectiveness(move: Move, defender: Pokemon) -> float:
    """
    Calculate the type effectiveness of a move against a defender.
    """
    eff = move.type.damage_multiplier(
        defender.type_1,
        defender.type_2,
        type_chart=GEN_4_DATA.type_chart
    )

    if move.type == PokemonType.FIRE and defender.ability in _FIRE_IMMUNITY:
        eff = 0
    elif move.type == PokemonType.ICE and defender.ability in _ICE_IMMUNITY:
        eff = 0
    elif move.type == PokemonType.GROUND and (defender.ability in _GROUND_IMMUNITY or defender._type_1 == PokemonType.FLYING or defender._type_2 == PokemonType.FLYING):
        eff = 0
    elif move.type == PokemonType.ELECTRIC and defender.ability in _ELECTRIC_IMMUNITY:
        eff = 0
    elif move.type == PokemonType.WATER and defender.ability in _WATER_IMMUNITY:
        eff = 0
    
    #special case for shedinja
    if defender.name == "shedinja" or defender.ability == "wonderguard":
        if eff < 2:
            eff = 0

    return eff

def estimate_matchup(mon: Pokemon, opponent: Pokemon):
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
    score -= max(
        [mon.damage_multiplier(t) for t in opponent.types if t is not None]
    )
    if mon.base_stats["spe"] > opponent.base_stats["spe"]:
        score += SPEED_TIER_COEFICIENT
    elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
        score -= SPEED_TIER_COEFICIENT

    score += mon.current_hp_fraction * HP_FRACTION_COEFICIENT
    score -= opponent.current_hp_fraction * HP_FRACTION_COEFICIENT

    return score

def calculate_switch_score(pokemon: Pokemon):
    pass
