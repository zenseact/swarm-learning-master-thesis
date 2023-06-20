import argparse
import json

from fedswarm import Platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to input file"
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="Run the platform even if the same config has been run before",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name the run folder something other than the default",
    )
    args = parser.parse_args()
    
    name = None if args.name == "NULL" else args.name
    
    # load the JSON config file from the specified path
    with open(args.config, "r") as f:
        config = json.load(f)

    # do something with the config data
    Platform(config, force=args.force, name=name)
