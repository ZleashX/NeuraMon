import numpy as np
import os
import uuid
import misc.utils as utils
from gymnasium.spaces import Box
from misc.constVar import *
from tqdm import tqdm

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import MaskablePPO

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.player import Gen4EnvSinglePlayer
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration

class RLPlayer(Gen4EnvSinglePlayer):
    _pokedex = utils.load_pokedex()
    _abilities_dex = utils.load_abilities()
    _moves_dex = utils.load_moves()
    _items_dex = utils.load_items()
    def __init__(self, opponent=None, battle_format="gen4randombattle", account_configuration=None, debug=False, server_configuration=None,start_challenging=True):
        super().__init__(battle_format=battle_format, opponent=opponent, account_configuration=account_configuration, server_configuration=server_configuration, start_challenging=start_challenging)
        self.debug = debug
        self.observation_space

    def embed_battle(self, battle):

        p_last_move = np.zeros(MAX_MOVE_ID)
        o_last_move = np.zeros(MAX_MOVE_ID)
        last_battle = battle.observations[battle.turn-1]
        events = last_battle.events
        for event in events:
            if event[1] == 'move' and event[2].startswith('p1a'):
                move = event[3]
                if move:
                    last_move_id = self._moves_dex.get(move.lower().replace(" ", "").replace("-", ""))
                    if last_move_id is None:
                        raise ValueError(f"Move not found: {move}")
                    p_last_move[last_move_id] = 1
            elif event[1] == 'move' and event[2].startswith('p2a'):
                move = event[3]
                if move:
                    last_move_id = self._moves_dex.get(move.lower().replace(" ", "").replace("-", ""))
                    if last_move_id is None:
                        raise ValueError(f"Move not found: {move}")
                    o_last_move[last_move_id] = 1
        
        battle_weather = np.zeros(NUM_WEATHER_TYPES)
        if battle.weather:
            for weather in battle.weather.keys():
                battle_weather[weather.value] = 1

        battle_field = np.zeros(NUM_FIELD_TYPES)
        if battle.fields:
            for field in battle.fields.keys():
                battle_field[field.value] = 1
        
        p_moves = np.zeros((TEAM_SIZE, 4, MAX_MOVE_ID))
        p_types = np.zeros((TEAM_SIZE, NUM_TYPES))
        p_species = np.zeros((TEAM_SIZE, MAX_SPECIES_ID))
        p_abilities = np.zeros((TEAM_SIZE, MAX_ABILITY_ID))
        p_items = np.zeros((TEAM_SIZE, MAX_ITEM_ID))
        p_boosts = np.zeros(NUM_BOOST_TYPES)
        p_status = np.zeros(NUM_STATUS_TYPES)
        p_effects = np.zeros(NUM_EFFECT_TYPES)
        p_hp = np.zeros(TEAM_SIZE)
        p_sidecondition = np.zeros(NUM_SIDE_CONDITIONS)

        if battle.side_conditions:
            for condition in battle.side_conditions.keys():
                p_sidecondition[condition.value] = 1
        else:
            p_sidecondition[0] = 1

        active_pokemon = battle.active_pokemon
        if active_pokemon:
            species, ability, item, _, types, effects, status = self._process_pokemon(active_pokemon, active=True)
            p_species[0, species] = 1
            p_abilities[0, ability] = 1
            p_items[0, item] = 1
            p_types[0, types[0]] = 1
            if types[1] != 0:
                p_types[0, types[1]] = 1
            p_effects[effects] = 1
            p_status[status] = 1
            for i, move in enumerate(battle.available_moves):
                move_id = self._moves_dex.get(move._id.lower().replace(" ", "").replace("-", ""))
                if move_id is None:
                    raise ValueError(f"Move not found: {move}")
                p_moves[0, i, move_id] = 1
            p_hp[0] = active_pokemon.current_hp_fraction

            for i, boost in enumerate(active_pokemon.boosts.values()):
                p_boosts[i] = boost

        for i, mon in enumerate(battle.available_switches):
            if i >= TEAM_SIZE: 
                break
            species, ability, item, moves, types, effects, status = self._process_pokemon(mon)
            p_species[i + 1, species] = 1
            p_abilities[i + 1, ability] = 1
            p_items[i + 1, item] = 1
            p_types[i + 1, types[0]] = 1
            if types[1] != 0:
                p_types[i + 1, types[1]] = 1
            p_effects[effects] = 1
            p_status[status] = 1
            for j, move in enumerate(moves):
                p_moves[i + 1, j, move] = 1
            p_hp[i + 1] = mon.current_hp_fraction

        o_moves = np.zeros((4, MAX_MOVE_ID))
        o_types = np.zeros(NUM_TYPES)
        o_species = np.zeros(MAX_SPECIES_ID)
        o_abilities = np.zeros(MAX_ABILITY_ID)
        o_items = np.zeros(MAX_ITEM_ID)
        o_boosts = np.zeros(NUM_BOOST_TYPES)
        o_status = np.zeros(NUM_STATUS_TYPES)
        o_effects = np.zeros(NUM_EFFECT_TYPES)
        o_hp = np.zeros(TEAM_SIZE)
        o_sidecondition = np.zeros(NUM_SIDE_CONDITIONS)
        if battle.opponent_side_conditions:
            for condition in battle.opponent_side_conditions.keys():
                o_sidecondition[condition.value] = 1
        
        opponent_active_pokemon = battle.opponent_active_pokemon
        if opponent_active_pokemon:
            species, ability, item, moves, types, effects, status = self._process_pokemon(opponent_active_pokemon, active=True)
            o_species[species] = 1
            o_abilities[ability] = 1
            o_items[item] = 1
            o_types[types[0]] = 1
            if types[1] != 0:
                o_types[types[1]] = 1
            o_effects[effects] = 1
            o_status[status] = 1
            for i, move in enumerate(moves):
                o_moves[i, move] = 1
            for i, boost in enumerate(opponent_active_pokemon.boosts.values()):
                o_boosts[i] = boost

        for i, mon in enumerate(battle.opponent_team.values()):
            o_hp[i] = mon.current_hp_fraction
        
        embedding = np.concatenate([
            battle_weather,
            battle_field,
            p_last_move,
            p_moves.flatten(),
            p_types.flatten(),
            p_species.flatten(),
            p_abilities.flatten(),
            p_items.flatten(),
            p_boosts,
            p_status,
            p_effects,
            p_hp,
            p_sidecondition,
            o_last_move,
            o_moves.flatten(),
            o_types,
            o_species,
            o_abilities,
            o_items,
            o_boosts,
            o_status,
            o_effects,
            o_hp,
            o_sidecondition,
        ])

        return embedding
    
    def _process_pokemon(self, mon: Pokemon, active = False):
        mon_species, mon_ability, mon_item, mon_effects, mon_status  = 0, 0, 0, 0, 0
        mon_moves = [0] * 4
        mon_species = self._pokedex.get(mon.base_species.lower())
        if mon_species is None:
                raise ValueError(f"Species not found: {mon.base_species}")
        if mon.ability:
            mon_ability = self._abilities_dex.get(mon.ability.lower())
        if mon_ability is None:
                raise ValueError(f"Ability not found: {mon.ability}")
        if mon.item == "unknown_item":
            mon_item = 0
        elif mon.item:
            mon_item = self._items_dex.get(mon.item.lower())
        else:
            mon_item = 1
        if mon_item is None:
                raise ValueError(f"Item not found: {mon.item}")
        if mon.moves:
            for i, move in enumerate(mon.moves.values()):
                moveid = self._moves_dex.get(move._id.lower().replace(" ", "").replace("-", ""))
                if moveid is None:
                    raise ValueError(f"Move not found: {move}")
                mon_moves[i] = moveid
        if mon.type_1:
            if mon.type_2:
                mon_types = [mon.type_1.value, mon.type_2.value]
            mon_types = [mon.type_1.value, 0]
        
        if active:
            if mon.effects:
                for effect in mon.effects.keys():
                    mon_effects = effect.value
            if mon.status:
                mon_status = mon.status.value


        return mon_species, mon_ability, mon_item, mon_moves, mon_types, mon_effects, mon_status       

    def action_masks(self):
        """
        Returns a mask for the actions that are not available in the current state.
        """
        # Get the available moves and switches
        available_moves = self.current_battle.available_moves
        available_switches = self.current_battle.available_switches

        # Create a mask for moves and switches
        move_mask = np.zeros(4, dtype=bool)
        switch_mask = np.zeros(6, dtype=bool)

        move_mask[:len(available_moves)] = True
        switch_mask[:len(available_switches)] = True

        return np.concatenate([move_mask, switch_mask])
    
    def calc_reward(self, last_state: AbstractBattle, current_state: AbstractBattle) -> float:
        reward = 0
        events = current_state.observations[current_state.turn-1].events
        if len(events) == 0:
            raise ValueError("No events")
        if current_state.won:
            reward += 1
        elif current_state.lost:
            reward -= 1
        
        fainted = 0.0125
        failed = -0.005
        supereffective = 0.0025
        resisted = -0.0025
        immune = -0.005
        
        for event in events:
            if event[1] == '-resisted':
                reward += self._process_reward(resisted, event[2])
            elif event[1] == '-immune':
                reward += self._process_reward(immune, event[2])
            elif event[1] == '-supereffective':
                reward += self._process_reward(supereffective, event[2])
            elif event[1] == 'faint':
                reward += self._process_reward(fainted, event[2])
            elif event[1] == '-fail':
                reward += self._process_reward(failed, event[2])

        return reward
    
    def _process_reward(self, value, player):
        reward = 0
        if player.startswith('p2a'):
            reward -= value
        elif player.startswith('p1a'):
            reward += value
        return reward

    
    def get_additional_info(self):
        if self.debug:
            return dict({'battle': self.current_battle})
        return {}
    
    def describe_embedding(self):
        low_battle_weather = [0] * NUM_WEATHER_TYPES
        high_battle_weather = [1] * NUM_WEATHER_TYPES
        low_battle_field = [0] * NUM_FIELD_TYPES
        high_battle_field = [1] * NUM_FIELD_TYPES
        low_p_last_move = [0] * MAX_MOVE_ID
        high_p_last_move = [1] * MAX_MOVE_ID
        low_p_moves = [0] * (TEAM_SIZE * 4 * MAX_MOVE_ID)
        high_p_moves = [1] * (TEAM_SIZE * 4 * MAX_MOVE_ID)
        low_p_types = [0] * (TEAM_SIZE * NUM_TYPES)
        high_p_types = [1] * (TEAM_SIZE * NUM_TYPES)
        low_p_species = [0] * (TEAM_SIZE * MAX_SPECIES_ID)
        high_p_species = [1] * (TEAM_SIZE * MAX_SPECIES_ID)
        low_p_abilities = [0] * (TEAM_SIZE * MAX_ABILITY_ID)
        high_p_abilities = [1] * (TEAM_SIZE * MAX_ABILITY_ID)
        low_p_items = [0] * (TEAM_SIZE * MAX_ITEM_ID)
        high_p_items = [1] * (TEAM_SIZE * MAX_ITEM_ID)
        low_p_boosts = [-6] * NUM_BOOST_TYPES
        high_p_boosts = [6] * NUM_BOOST_TYPES
        low_p_status = [0] * NUM_STATUS_TYPES
        high_p_status = [1] * NUM_STATUS_TYPES
        low_p_effects = [0] * NUM_EFFECT_TYPES
        high_p_effects = [1] * NUM_EFFECT_TYPES
        low_p_hp = [0] * TEAM_SIZE
        high_p_hp = [1] * TEAM_SIZE
        low_p_sidecondition = [0] * NUM_SIDE_CONDITIONS
        high_p_sidecondition = [1] * NUM_SIDE_CONDITIONS
        low_o_last_move = [0] * MAX_MOVE_ID
        high_o_last_move = [1] * MAX_MOVE_ID
        low_o_moves = [0] * (MAX_MOVE_ID * 4)
        high_o_moves = [1] * (MAX_MOVE_ID * 4)
        low_o_types = [0] * NUM_TYPES
        high_o_types = [1] * NUM_TYPES
        low_o_species = [0] * MAX_SPECIES_ID
        high_o_species = [1] * MAX_SPECIES_ID
        low_o_abilities = [0] * MAX_ABILITY_ID
        high_o_abilities = [1] * MAX_ABILITY_ID
        low_o_items = [0] * MAX_ITEM_ID
        high_o_items = [1] * MAX_ITEM_ID
        low_o_boosts = [-6] * NUM_BOOST_TYPES
        high_o_boosts = [6] * NUM_BOOST_TYPES
        low_o_status = [0] * NUM_STATUS_TYPES
        high_o_status = [1] * NUM_STATUS_TYPES
        low_o_effects = [0] * NUM_EFFECT_TYPES
        high_o_effects = [1] * NUM_EFFECT_TYPES
        low_o_hp = [0] * TEAM_SIZE
        high_o_hp = [1] * TEAM_SIZE
        low_o_sidecondition = [0] * NUM_SIDE_CONDITIONS
        high_o_sidecondition = [1] * NUM_SIDE_CONDITIONS

        low = (
            low_battle_weather
            + low_battle_field
            + low_p_last_move
            + low_p_moves
            + low_p_types
            + low_p_species
            + low_p_abilities
            + low_p_items
            + low_p_boosts
            + low_p_status
            + low_p_effects
            + low_p_hp
            + low_p_sidecondition
            + low_o_last_move
            + low_o_moves
            + low_o_types
            + low_o_species
            + low_o_abilities
            + low_o_items
            + low_o_boosts
            + low_o_status
            + low_o_effects
            + low_o_hp
            + low_o_sidecondition
        )

        high = (
            high_battle_weather
            + high_battle_field
            + high_p_last_move
            + high_p_moves
            + high_p_types
            + high_p_species
            + high_p_abilities
            + high_p_items
            + high_p_boosts
            + high_p_status
            + high_p_effects
            + high_p_hp
            + high_p_sidecondition
            + high_o_last_move
            + high_o_moves
            + high_o_types
            + high_o_species
            + high_o_abilities
            + high_o_items
            + high_o_boosts
            + high_o_status
            + high_o_effects
            + high_o_hp
            + high_o_sidecondition
        )
        
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

class RLOpponent(Player):
    _pokedex = utils.load_pokedex()
    _abilities_dex = utils.load_abilities()
    _moves_dex = utils.load_moves()
    _items_dex = utils.load_items()
    def __init__(self, model_path, battle_format, account_configuration=None):
        super().__init__(battle_format=battle_format,account_configuration=account_configuration)
        self.model_path = model_path
        self.model = None
        self.last_mod_time = None

    def load_model(self):
        try:
            current_mod_time = os.path.getmtime(self.model_path)
        except FileNotFoundError:
            return
        
        if self.model is None or current_mod_time != self.last_mod_time:
            self.model = MaskablePPO.load(self.model_path)
            self.last_mod_time = current_mod_time

    def embed_battle(self, battle):

        p_last_move = np.zeros(MAX_MOVE_ID)
        o_last_move = np.zeros(MAX_MOVE_ID)
        last_battle = battle.observations[battle.turn-1]
        events = last_battle.events
        for event in events:
            if event[1] == 'move' and event[2].startswith('p1a'):
                move = event[3]
                if move:
                    last_move_id = self._moves_dex.get(move.lower().replace(" ", "").replace("-", ""))
                    if last_move_id is None:
                        raise ValueError(f"Move not found: {move}")
                    p_last_move[last_move_id] = 1
            elif event[1] == 'move' and event[2].startswith('p2a'):
                move = event[3]
                if move:
                    last_move_id = self._moves_dex.get(move.lower().replace(" ", "").replace("-", ""))
                    if last_move_id is None:
                        raise ValueError(f"Move not found: {move}")
                    o_last_move[last_move_id] = 1
        
        battle_weather = np.zeros(NUM_WEATHER_TYPES)
        if battle.weather:
            for weather in battle.weather.keys():
                battle_weather[weather.value] = 1

        battle_field = np.zeros(NUM_FIELD_TYPES)
        if battle.fields:
            for field in battle.fields.keys():
                battle_field[field.value] = 1
        
        p_moves = np.zeros((TEAM_SIZE, 4, MAX_MOVE_ID))
        p_types = np.zeros((TEAM_SIZE, NUM_TYPES))
        p_species = np.zeros((TEAM_SIZE, MAX_SPECIES_ID))
        p_abilities = np.zeros((TEAM_SIZE, MAX_ABILITY_ID))
        p_items = np.zeros((TEAM_SIZE, MAX_ITEM_ID))
        p_boosts = np.zeros(NUM_BOOST_TYPES)
        p_status = np.zeros(NUM_STATUS_TYPES)
        p_effects = np.zeros(NUM_EFFECT_TYPES)
        p_hp = np.zeros(TEAM_SIZE)
        p_sidecondition = np.zeros(NUM_SIDE_CONDITIONS)

        if battle.side_conditions:
            for condition in battle.side_conditions.keys():
                p_sidecondition[condition.value] = 1
        else:
            p_sidecondition[0] = 1

        active_pokemon = battle.active_pokemon
        if active_pokemon:
            species, ability, item, _, types, effects, status = self._process_pokemon(active_pokemon, active=True)
            p_species[0, species] = 1
            p_abilities[0, ability] = 1
            p_items[0, item] = 1
            p_types[0, types[0]] = 1
            if types[1] != 0:
                p_types[0, types[1]] = 1
            p_effects[effects] = 1
            p_status[status] = 1
            for i, move in enumerate(battle.available_moves):
                move_id = self._moves_dex.get(move._id.lower().replace(" ", "").replace("-", ""))
                if move_id is None:
                    raise ValueError(f"Move not found: {move}")
                p_moves[0, i, move_id] = 1
            p_hp[0] = active_pokemon.current_hp_fraction

            for i, boost in enumerate(active_pokemon.boosts.values()):
                p_boosts[i] = boost

        for i, mon in enumerate(battle.available_switches):
            if i >= TEAM_SIZE: 
                break
            species, ability, item, moves, types, effects, status = self._process_pokemon(mon)
            p_species[i + 1, species] = 1
            p_abilities[i + 1, ability] = 1
            p_items[i + 1, item] = 1
            p_types[i + 1, types[0]] = 1
            if types[1] != 0:
                p_types[i + 1, types[1]] = 1
            p_effects[effects] = 1
            p_status[status] = 1
            for j, move in enumerate(moves):
                p_moves[i + 1, j, move] = 1
            p_hp[i + 1] = mon.current_hp_fraction

        o_moves = np.zeros((4, MAX_MOVE_ID))
        o_types = np.zeros(NUM_TYPES)
        o_species = np.zeros(MAX_SPECIES_ID)
        o_abilities = np.zeros(MAX_ABILITY_ID)
        o_items = np.zeros(MAX_ITEM_ID)
        o_boosts = np.zeros(NUM_BOOST_TYPES)
        o_status = np.zeros(NUM_STATUS_TYPES)
        o_effects = np.zeros(NUM_EFFECT_TYPES)
        o_hp = np.zeros(TEAM_SIZE)
        o_sidecondition = np.zeros(NUM_SIDE_CONDITIONS)
        if battle.opponent_side_conditions:
            for condition in battle.opponent_side_conditions.keys():
                o_sidecondition[condition.value] = 1
        
        opponent_active_pokemon = battle.opponent_active_pokemon
        if opponent_active_pokemon:
            species, ability, item, moves, types, effects, status = self._process_pokemon(opponent_active_pokemon, active=True)
            o_species[species] = 1
            o_abilities[ability] = 1
            o_items[item] = 1
            o_types[types[0]] = 1
            if types[1] != 0:
                o_types[types[1]] = 1
            o_effects[effects] = 1
            o_status[status] = 1
            for i, move in enumerate(moves):
                o_moves[i, move] = 1
            for i, boost in enumerate(opponent_active_pokemon.boosts.values()):
                o_boosts[i] = boost

        for i, mon in enumerate(battle.opponent_team.values()):
            o_hp[i] = mon.current_hp_fraction
        
        embedding = np.concatenate([
            battle_weather,
            battle_field,
            p_last_move,
            p_moves.flatten(),
            p_types.flatten(),
            p_species.flatten(),
            p_abilities.flatten(),
            p_items.flatten(),
            p_boosts,
            p_status,
            p_effects,
            p_hp,
            p_sidecondition,
            o_last_move,
            o_moves.flatten(),
            o_types,
            o_species,
            o_abilities,
            o_items,
            o_boosts,
            o_status,
            o_effects,
            o_hp,
            o_sidecondition,
        ])

        return embedding

    def _process_pokemon(self, mon: Pokemon, active = False):
        mon_species, mon_ability, mon_item, mon_effects, mon_status  = 0, 0, 0, 0, 0
        mon_moves = [0] * 4
        mon_species = self._pokedex.get(mon.base_species.lower())
        if mon_species is None:
                raise ValueError(f"Species not found: {mon.base_species}")
        if mon.ability:
            mon_ability = self._abilities_dex.get(mon.ability.lower())
        if mon_ability is None:
                raise ValueError(f"Ability not found: {mon.ability}")
        if mon.item == "unknown_item":
            mon_item = 0
        elif mon.item:
            mon_item = self._items_dex.get(mon.item.lower())
        else:
            mon_item = 1
        if mon_item is None:
                raise ValueError(f"Item not found: {mon.item}")
        if mon.moves:
            for i, move in enumerate(mon.moves.values()):
                moveid = self._moves_dex.get(move._id.lower().replace(" ", "").replace("-", ""))
                if moveid is None:
                    raise ValueError(f"Move not found: {move}")
                mon_moves[i] = moveid
        if mon.type_1:
            if mon.type_2:
                mon_types = [mon.type_1.value, mon.type_2.value]
            mon_types = [mon.type_1.value, 0]
        
        if active:
            if mon.effects:
                for effect in mon.effects.keys():
                    mon_effects = effect.value
            if mon.status:
                mon_status = mon.status.value


        return mon_species, mon_ability, mon_item, mon_moves, mon_types, mon_effects, mon_status   

    def choose_move(self, battle):
        self.load_model()
        if self.model is None:
            return self.choose_random_move(battle)
        
        obs = self.embed_battle(battle)
        masks = self.action_masks(battle)
        action, _ = self.model.predict(
            obs,
            action_masks=masks,
            deterministic=True
        )
        
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)
        
    def action_masks(self, battle):
        """
        Returns a mask for the actions that are not available in the current state.
        """
        # Get the available moves and switches
        available_moves = battle.available_moves
        available_switches = battle.available_switches

        # Create a mask for moves and switches
        move_mask = np.zeros(4, dtype=bool)
        switch_mask = np.zeros(6, dtype=bool)

        move_mask[:len(available_moves)] = True
        switch_mask[:len(available_switches)] = True

        return np.concatenate([move_mask, switch_mask])
        
    
def make_env(model_path, eval=False, debug=False):
    def _init():
        env_num = str(uuid.uuid4())[:5] 
        print(f"Creating environment {env_num}")
        if eval:
            opponent = SimpleHeuristicsPlayer(battle_format="gen4randombattle", account_configuration=AccountConfiguration(f"HeurPlayer_{env_num}", None))
        else:
            opponent = RLOpponent(model_path, "gen4randombattle", AccountConfiguration(f"RLOpponent_{env_num}", None))
        env = RLPlayer(opponent=opponent, account_configuration=AccountConfiguration(f"RLPlayer_{env_num}", None), debug=debug)
        return env
    return _init

def eval_progress(locals, globals):
    if "episode_counts" in locals:
        total = 0
        for count in locals["episode_counts"]:
            total += count
        pbar.update(total - pbar.n)

modeldir = "models/ppo"
logdir = "logs"
MODEL_PATH = os.path.join(modeldir, f"complex_ppo.zip")
NUM_ENVS = 8

if not os.path.exists(modeldir):
    os.makedirs(modeldir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

def train():
    global pbar
    # Create vectorized environment
    vec_env = make_vec_env(make_env(MODEL_PATH), n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    
    if os.path.exists(MODEL_PATH):
        model = MaskablePPO.load(MODEL_PATH, env=vec_env)
    else:
        model = MaskablePPO("MlpPolicy", vec_env, verbose=1,
                    n_steps=6 * 128, gamma=0.9999, gae_lambda= 0.754, clip_range=0.0829, clip_range_vf=0.0184, ent_coef=0.0588,
                    vf_coef=0.4375, max_grad_norm=0.543, batch_size=256, n_epochs=7,
                    tensorboard_log=logdir
                    )
    num_epochs = 200
    timesteps_per_epoch = model.n_steps * NUM_ENVS * 4
    
    for epoch in range(num_epochs):
        model.learn(
            total_timesteps=timesteps_per_epoch,
            reset_num_timesteps=False,
            tb_log_name="complex_ppo",
            progress_bar=True
        )
        
        # Save checkpoint
        model.save(MODEL_PATH)
        print("Model saved...")
        vec_env.close()

        #create a validation environment
        print("Evaluating model...")
        val_env = make_vec_env(make_env(MODEL_PATH, eval=True), n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
        pbar = tqdm(total=200, desc="Evaluating episodes")
        mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=200, deterministic=True, callback=eval_progress)
        pbar.close()
        battles_won = sum(val_env.get_attr("n_won_battles"))
        battles_finished = sum(val_env.get_attr("n_finished_battles"))
        winrate = battles_won / battles_finished * 100

        model.logger.record("eval/mean_reward", mean_reward)
        model.logger.record("eval/winrate", winrate)
        model.logger.dump(model.num_timesteps)
        val_env.close()

        if model.num_timesteps >= 13_000_000:
            print("Training complete.")
            break

        # Recreate the environment to avoid memory leaks
        vec_env = make_vec_env(make_env(MODEL_PATH), n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
        model.set_env(vec_env)

    vec_env.close()
    