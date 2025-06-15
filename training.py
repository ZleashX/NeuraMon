import argparse
import sys
from train.simplev5_ppo import train as simplev5_ppo_train
from train.simplev5 import train as simplev5_recurrentppo_train
from train.complex_ppo import train as complex_ppo_train
from train.complex import train as complex_recurrentppo_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model type and algorithm to train")
    parser.add_argument("--state", choices=["simple", "complex"], required=True, help="State Feature Type")
    parser.add_argument("--algo", choices=["ppo", "recurrentppo"], required=True, help="Algorithm")
    args = parser.parse_args()

    if args.state == "simple" and args.algo == "ppo":
        simplev5_ppo_train()
    elif args.state == "simple" and args.algo == "recurrentppo":
        simplev5_recurrentppo_train()
    elif args.state == "complex" and args.algo == "ppo":
        complex_ppo_train()
    elif args.state == "complex" and args.algo == "recurrentppo":
        complex_recurrentppo_train()
    else:
        print("Invalid combination of state and algo.")
        sys.exit(1)