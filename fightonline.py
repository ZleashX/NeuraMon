import argparse
import asyncio
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
from train.complex import RLOpponent as RPPOComplex
from train.complex_ppo import RLOpponent as PPOComplex
from train.simplev5 import RLOpponent as RPPOSimple
from train.simplev5_ppo import RLOpponent as PPOSimple

async def main():
    print("Ready to fight!")
    await player.accept_challenges(None, n_challenges=args.battles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Online")
    parser.add_argument("--model", choices=["recurrentpposimple", "recurrentppocomplex", "pposimple", "ppocomplex"], required=True, help="Player model")
    parser.add_argument("--battles", type=int, default=1, help="Number of challenge battles to send")
    args = parser.parse_args()

    if args.model == "recurrentppocomplex":
        player = RPPOComplex("models/recurrent_ppo/complex.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("Socapdi","socapdi"), server_configuration=ShowdownServerConfiguration)
    elif args.model == "recurrentpposimple":
        player = RPPOSimple("models/recurrent_ppo/simplev5.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("Socapdi","socapdi"), server_configuration=ShowdownServerConfiguration)
    elif args.model == "ppocomplex":
        player = PPOComplex("models/ppo/complex_ppo.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("Socapdi","socapdi"), server_configuration=ShowdownServerConfiguration)
    elif args.model == "pposimple":
        player = PPOSimple("models/ppo/simplev5_ppo.zip", battle_format="gen4randombattle", account_configuration=AccountConfiguration("Socapdi","socapdi"), server_configuration=ShowdownServerConfiguration)

    asyncio.get_event_loop().run_until_complete(main())

