import numpy as np
import os
import uuid
import misc.utils as utils
from gymnasium.spaces import Box
from misc.constVar import *
from tqdm import tqdm

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import MaskablePPO

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.player import Gen4EnvSinglePlayer
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from misc.battle_utils import calculate_damage, estimate_matchup, _stat_estimation

class RLPlayer(Gen4EnvSinglePlayer):
    def __init__(self, opponent=None, battle_format="gen4randombattle", account_configuration=None, debug=False, server_configuration=None,start_challenging=True):
        super().__init__(battle_format=battle_format, opponent=opponent, account_configuration=account_configuration, server_configuration=server_configuration, start_challenging=start_challenging)
        self.debug = debug

    def embed_battle(self, battle):
        moves_dmg_multiplier = np.ones(4)
        moves_status = np.zeros((4, NUM_STATUS_TYPES))
        moves_boost = np.zeros((4, NUM_BOOST_TYPES))
        move: Move
        for i, move in enumerate(battle.available_moves):
            moves_dmg_multiplier[i] = calculate_damage(move, battle.active_pokemon, battle.opponent_active_pokemon, battle.weather, battle.side_conditions)
            if move.status:
                moves_status[i,move.status.value] = 1
            if move.boosts:
                for stat, boost in move.boosts.items():
                    moves_boost[i, STAT_TO_IDX[stat]] = boost

        p_active_status = np.zeros(NUM_STATUS_TYPES)
        p_active_boost = np.zeros(NUM_BOOST_TYPES)
        o_active_status = np.zeros(NUM_STATUS_TYPES)
        o_active_boost = np.zeros(NUM_BOOST_TYPES)
        if battle.active_pokemon.status:
            p_active_status[battle.active_pokemon.status.value] = 1
        for i, boost in enumerate(battle.active_pokemon.boosts.values()):
            p_active_boost[i] = boost
        if battle.opponent_active_pokemon.status:
            o_active_status[battle.opponent_active_pokemon.status.value] = 1
        for i, boost in enumerate(battle.opponent_active_pokemon.boosts.values()):
            o_active_boost[i] = boost

        
        p_health = np.zeros(TEAM_SIZE)
        o_health = np.zeros(TEAM_SIZE)
        p_active = np.zeros(TEAM_SIZE)
        o_active = np.zeros(TEAM_SIZE)

        for i, mon in enumerate(battle.team.values()):
            if i >= TEAM_SIZE: 
                break
            p_health[i] = mon.current_hp_fraction
            if mon.active:
                p_active[i] = 1
        for i, mon in enumerate(battle.opponent_team.values()):
            if i >= TEAM_SIZE: 
                break
            o_health[i] = mon.current_hp_fraction
            if mon.active:
                o_active[i] = 1
        
        switch_scores = np.zeros(TEAM_SIZE)
        switch_scores[0] = estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon)
        for i, mon in enumerate(battle.available_switches):
            if i >= TEAM_SIZE: 
                break
            switch_scores[i+1] = estimate_matchup(mon, battle.opponent_active_pokemon)

        speed_score = np.zeros(1)
        opponent_speed = _stat_estimation(battle.opponent_active_pokemon, "spe")
        if battle.active_pokemon.stats["spe"] is not None:
            if battle.active_pokemon.stats["spe"] > opponent_speed:
                speed_score = np.ones(1)
        
        embedding = np.concatenate([
            moves_dmg_multiplier,
            moves_status.flatten(),
            moves_boost.flatten(),
            p_active_status,
            o_active_status,
            p_active_boost,
            o_active_boost,
            p_health,
            o_health,
            p_active,
            o_active,
            switch_scores,
            speed_score,
        ])

        return embedding
        
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
        low_dmgmult = [0] * 4
        high_dmgmult = [1] * 4
        low_status = [0] * 4 * NUM_STATUS_TYPES
        high_status = [1] * 4 * NUM_STATUS_TYPES
        low_boost = [-6] * 4 * NUM_BOOST_TYPES
        high_boost = [6] * 4 * NUM_BOOST_TYPES
        low_active_status = [0] * NUM_STATUS_TYPES * 2
        high_active_status = [1] * NUM_STATUS_TYPES * 2
        low_active_boost = [-6] * NUM_BOOST_TYPES * 2
        high_active_boost = [6] * NUM_BOOST_TYPES * 2
        low_health = [0] * TEAM_SIZE * 2
        high_health = [1] * TEAM_SIZE * 2
        low_active = [0] * TEAM_SIZE * 2
        high_active = [1] * TEAM_SIZE * 2
        low_switch = [-4.5] * TEAM_SIZE
        high_switch = [4.5] * TEAM_SIZE
        low_speed = [0]
        high_speed = [1]

        low = (low_dmgmult + low_status + low_boost + low_active_status + low_active_boost + low_health + low_active + low_switch + low_speed)
        high = (high_dmgmult + high_status + high_boost + high_active_status + high_active_boost + high_health + high_active + high_switch + high_speed)
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

class RLOpponent(Player):
    def __init__(self, model_path, battle_format, account_configuration=None, server_configuration=None):
        super().__init__(battle_format=battle_format,account_configuration=account_configuration,server_configuration=server_configuration,start_timer_on_battle_start=True)
        self.model = MaskablePPO.load(model_path) if os.path.exists(model_path) else None

    def embed_battle(self, battle):
        moves_dmg_multiplier = np.ones(4)
        moves_status = np.zeros((4, NUM_STATUS_TYPES))
        moves_boost = np.zeros((4, NUM_BOOST_TYPES))
        move: Move
        for i, move in enumerate(battle.available_moves):
            moves_dmg_multiplier[i] = calculate_damage(move, battle.active_pokemon, battle.opponent_active_pokemon, battle.weather, battle.side_conditions)
            if move.status:
                moves_status[i,move.status.value] = 1
            if move.boosts:
                for stat, boost in move.boosts.items():
                    moves_boost[i, STAT_TO_IDX[stat]] = boost

        p_active_status = np.zeros(NUM_STATUS_TYPES)
        p_active_boost = np.zeros(NUM_BOOST_TYPES)
        o_active_status = np.zeros(NUM_STATUS_TYPES)
        o_active_boost = np.zeros(NUM_BOOST_TYPES)
        if battle.active_pokemon.status:
            p_active_status[battle.active_pokemon.status.value] = 1
        for i, boost in enumerate(battle.active_pokemon.boosts.values()):
            p_active_boost[i] = boost
        if battle.opponent_active_pokemon.status:
            o_active_status[battle.opponent_active_pokemon.status.value] = 1
        for i, boost in enumerate(battle.opponent_active_pokemon.boosts.values()):
            o_active_boost[i] = boost

        
        p_health = np.zeros(TEAM_SIZE)
        o_health = np.zeros(TEAM_SIZE)
        p_active = np.zeros(TEAM_SIZE)
        o_active = np.zeros(TEAM_SIZE)

        for i, mon in enumerate(battle.team.values()):
            if i >= TEAM_SIZE: 
                break
            p_health[i] = mon.current_hp_fraction
            if mon.active:
                p_active[i] = 1
        for i, mon in enumerate(battle.opponent_team.values()):
            if i >= TEAM_SIZE: 
                break
            o_health[i] = mon.current_hp_fraction
            if mon.active:
                o_active[i] = 1
        
        switch_scores = np.zeros(TEAM_SIZE)
        switch_scores[0] = estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon)
        for i, mon in enumerate(battle.available_switches):
            if i >= TEAM_SIZE: 
                break
            switch_scores[i+1] = estimate_matchup(mon, battle.opponent_active_pokemon)

        speed_score = np.zeros(1)
        opponent_speed = _stat_estimation(battle.opponent_active_pokemon, "spe")
        if battle.active_pokemon.stats["spe"] is not None:
            if battle.active_pokemon.stats["spe"] > opponent_speed:
                speed_score = np.ones(1)
        
        embedding = np.concatenate([
            moves_dmg_multiplier,
            moves_status.flatten(),
            moves_boost.flatten(),
            p_active_status,
            o_active_status,
            p_active_boost,
            o_active_boost,
            p_health,
            o_health,
            p_active,
            o_active,
            switch_scores,
            speed_score,
        ])

        return embedding

    def choose_move(self, battle):
        if self.model is None:
            return self.choose_random_move(battle)
        
        obs = self.embed_battle(battle)
        masks = self.action_masks(battle)
        action,_ = self.model.predict(
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
        env_num = str(uuid.uuid4())[:4] 
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
MODEL_PATH = os.path.join(modeldir, f"simpleV5_ppo.zip")
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
                    n_steps=NUM_ENVS * 256, gamma=0.9999, gae_lambda= 0.754, clip_range=0.0829, clip_range_vf=0.0184, ent_coef=0.0588,
                    vf_coef=0.4375, max_grad_norm=0.543, batch_size=512, n_epochs=7,
                    tensorboard_log=logdir
                    )
    num_epochs = 200
    timesteps_per_epoch = model.n_steps * NUM_ENVS * 4
    
    for epoch in range(num_epochs):
        model.learn(
            total_timesteps=timesteps_per_epoch,
            reset_num_timesteps=False,
            tb_log_name="simpleV5_ppo",
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

        if model.num_timesteps >= 5_000_000:
            print("Reached 5 million timesteps, stopping training.")
            break

        # Recreate the environment to avoid memory leaks
        vec_env = make_vec_env(make_env(MODEL_PATH), n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
        model.set_env(vec_env)

    vec_env.close()
    