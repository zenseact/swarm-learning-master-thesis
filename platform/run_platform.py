import argparse
import json

from fedswarm import Platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to input file"
    )
    parser.add_argument(
        "--force", type=bool, default=False, help="Run the platform even if the same config has been run before"
    )
    args = parser.parse_args()

    # load the JSON config file from the specified path
    with open(args.config, "r") as f:
        config = json.load(f)

    # do something with the config data
    Platform(config, force=args.force)
