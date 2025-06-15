from poke_env import cross_evaluate
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from train.simplev5_ppo import RLOpponent as PPOSimple
from train.simplev5 import RLOpponent as RPPOSimple
from train.complex import RLOpponent as RPPOComplex
from train.complex_ppo import RLOpponent as PPOComplex
from tabulate import tabulate
import asyncio
import argparse
from tqdm import tqdm

class RPPOSimple(RPPOSimple):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)
class PPOSimple(PPOSimple):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)
class RPPOComplex(RPPOComplex):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)
class PPOComplex(PPOComplex):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)
class RandomPlayer(RandomPlayer):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)
class SimpleHeuristicsPlayer(SimpleHeuristicsPlayer):
    def _init(self, *args, **kwargs):
        super()._init(*args, **kwargs)
    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        pbar.update(0.5)

async def main():
    rpposimple = RPPOSimple("models/recurrent_ppo/simpleV5.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("RPPOSimple", None))
    pposimple = PPOSimple("models/ppo/simpleV5_ppo.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("PPOSimple", None))
    rppocomplex = RPPOComplex("models/recurrent_ppo/complex.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("RPPOComplex", None))
    ppocomplex = PPOComplex("models/ppo/complex_ppo.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("PPOComplex", None))
    randomplayer = RandomPlayer(battle_format="gen4randombattle", account_configuration=AccountConfiguration("Random", None))
    shplayer = SimpleHeuristicsPlayer(battle_format="gen4randombattle", account_configuration=AccountConfiguration("Heuristic", None))
    players = [rpposimple, pposimple, rppocomplex, ppocomplex, randomplayer, shplayer]
    results = await cross_evaluate(players, n_challenges=eval_episodes)

    table = [["-"] + [p.username for p in players]]
    for p_1, result in results.items():
        table.append([p_1] + [results[p_1][p_2] for p_2 in result])

    print(tabulate(table))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-evaluate models")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()
    global pbar
    eval_episodes = args.episodes
    total_battles = (6 * (6 - 1) // 2) * eval_episodes
    pbar = tqdm(total=total_battles, desc="Evaluating episodes")
    asyncio.get_event_loop().run_until_complete(main())