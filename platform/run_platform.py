import argparse
import json

from fedswarm import Platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--config", type=str, default="input.txt", help="Path to input file"
    )
    args = parser.parse_args()

    # load the JSON config file from the specified path
    with open(args.config, "r") as f:
        config = json.load(f)

    # do something with the config data
    Platform(config)
