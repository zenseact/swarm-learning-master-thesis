from copy import deepcopy
import logging
import jsonschema
import json
import ray
import os

from pathlib import Path
from jinja2 import Template
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .utils.data.data_handler import DataHandler
from .utils.training import run_centralised, run_federated, run_swarm

logger = logging.getLogger(__name__)

# get the absolute path to the directory where the current file resides
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
platform_dir = current_dir.parent.absolute()
# build the path to the config file relative to the current file's directory
templates_path = os.path.join(current_dir, "utils", "templates")

# open the files using the absolute path
with open(templates_path + "/config_schema.json") as file:
    CONFIG_SCHEMA = json.load(file)

with open(templates_path + "/config_message.md", "r") as file:
    TEMPLATE = Template(file.read())


class Platform:
    def __init__(self, config: dict, data_only: bool = False, write=True) -> None:
        try:
            # Save config
            self.config = deepcopy(config)
            # Create run and relevant directories
            self.top_log_dir = Path( "runs")
            self.create_if_not_exists(self.top_log_dir)
            self.run_id = self.create_run()
            self.run_dir = Path(self.top_log_dir, self.run_id)
            self.write = write
            self.skip = False
            
            skip, same_dir = self.check_for_same_run(config)
            if skip:
                print("Same run already exists, skipping run but creates platform")
                print("Run found in {}".format(same_dir))
                self.skip = True
                self.write = False
            else:
                print("No same run found, creating new run")
            if self.write:
                self.create_if_not_exists(self.run_dir)
                # Save config to file
                with open("{}.json".format(Path(self.run_dir, "config")), "w") as f:
                    json.dump(self.config, f, indent=4)
                logger.info("New run created: {}".format(self.run_id))
                # Set up tensorboard writer
                self.writer = SummaryWriter(self.run_dir)
                # Set up logging to file and set format
                logging.basicConfig(
                    filename="{}.log".format(Path(self.run_dir, "platform")),
                    encoding="utf-8",
                    level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s",
                )

            # Check if config is valid
            self.validate_config()

            self.parse_config()

            self.announce_configuration()

            

            # Run configuration
            self.tb_log_config(config)
            self.data = DataHandler(self.config, self.run_dir)

            # If data_only is set, stop here
            if data_only or self.skip:
                return 

            # Run training for each enabled method
            if "central" in self.methods:
                run_centralised(**self.training_args("central"))
                self.unmount_dataloaders("central")

            if "federated" in self.methods:
                run_federated(**self.training_args("federated"))
                self.unmount_dataloaders("federated")

            if "swarm" in self.methods:
                run_swarm(**self.training_args("swarm"))
                self.unmount_dataloaders("swarm")

            if "baseline" in self.methods:
                run_swarm(**self.training_args("baseline"), baseline=True)
                self.unmount_dataloaders("baseline")
            
            logger.info("END OF PLATFORM ACTIVITIES - SHUTTING DOWN")
        except Exception as e:
            logger.exception(e)
            logger.error("UNCAUGHT EXCEPTION - SHUTTING DOWN")
            raise e
        if self.write:
            self.writer.close()
            logging.shutdown()

    def training_args(self, method: str) -> list:
        # Dynamically create arguments for training functions
        return dict(
            config=self.config,
            data=getattr(self.data, method),
            log_dir=self.run_dir,
        )

    def create_if_not_exists(self, path: Path | str) -> None:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: {}".format(path))
        except FileExistsError:
            logger.debug("Directory already exists: {}".format(path))
        except Exception as e:
            logger.error("Unknown error when creating directory: {}".format(path))
            logger.exception(e)
            raise e

    def create_run(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def validate_config(self) -> None:
        logger.debug("Validating configuration")
        try:
            jsonschema.validate(instance=self.config, schema=CONFIG_SCHEMA)
            logger.info("Configuration is valid")
        except jsonschema.exceptions.ValidationError as e:
            logger.error("Invalid configuration")
            logger.exception(e)
            raise e

    def parse_config(self) -> None:
        logger.debug("Parsing configuration")
        # Get list of decentralised methods from shared decentralised config
        try:
            decentralised_methods = self.config["decentralised"]["train"]
        except KeyError:
            decentralised_methods = []

        # Get list of specified decentralised methods from config
        specific_decentralised_methods = [
            method
            for method in self.config.keys()
            if method in ["federated", "swarm", "baseline"]
        ]

        # Create an empty list for all methods
        methods = []

        # Check if centralised method is specified in config
        try:
            if "central" in self.config and self.config["central"]["train"]:
                methods.append("central")
        except KeyError:
            pass
        
        # Add all enabled methods to the list
        self.methods = methods + decentralised_methods + specific_decentralised_methods

        # Check for methods with ambiguous config
        ambigious = set(specific_decentralised_methods).intersection(
            set(decentralised_methods)
        )
        if ambigious:
            logger.warning("Ambiguous methods: {}".format(ambigious))
            logger.warning(
                "{} is provided in both decentralised and specific config".format(
                    ambigious
                )
            )
            logger.warning("Using specific config for {}".format(ambigious))

        # Remove ambiguous methods from decentralised methods
        decentralised_methods = list(set(decentralised_methods) - ambigious)

        # Expand the config
        if decentralised_methods:
            expansion_part = deepcopy(self.config["decentralised"])
            expansion_part.pop("train")

            shortcuts = []
            to_remove = []

            # Identify shortcut parameters
            for key in expansion_part.keys():
                for sub_key, value in expansion_part[key].items():
                    key_chunks = sub_key.split("_")
                    # Check if key is a shortcut
                    if key_chunks[0] in ["federated", "swarm", "baseline"]:
                        shortcuts.append(([key_chunks[0], key, key_chunks[1]], value))
                        to_remove.append((key, sub_key))

            # Remove shortcut parameters from expansion part
            for key, sub_key in to_remove:
                expansion_part[key].pop(sub_key)

            # Add enabled train parameters to expansion part
            expansion_part["train"] = True

            # Make decentralised parameters into specific methods
            for method in decentralised_methods:
                self.config[method] = deepcopy(expansion_part)

        # For undefined methods, add and set train to false
        for method in set(["central", "federated", "swarm", "baseline"]) - set(
            self.methods
        ):
            self.config[method] = {"train": False}

        # Expand shortcut parameters
        if decentralised_methods:
            for (method, top_key, sub_key), value in shortcuts:
                if method in self.methods:
                    self.config[method][top_key][sub_key] = value

        # Add baseline orchestrator for swarm simulator
        if "baseline" in self.methods:
            self.config["baseline"][
                "orchestrator"
            ] = "synchronous_fixed_rounds_edgeless"

        logger.debug("Expanded config: {}".format(self.config))

    def announce_configuration(self) -> None:
        logger.info("[CONFIG] Enabled methods: {}".format(self.methods))
        logger.info(
            "[CONFIG] Data allowance: {:.2%} of available data".format(
                self.config["data"]["ratio"]
            )
        )
        logger.info("[CONFIG] More information in tensorboard")

    def tb_log_config(self, original_config: dict) -> None:
        logger.debug("Logging configuration to tensorboard")
        config = json.dumps(original_config, indent=4)
        config_expanded = json.dumps(self.config, indent=4)
        note = "" if not "note" in self.config else self.config["note"]
        logger.debug("Reading configuration template")
        render = TEMPLATE.render(
            config=config,
            config_expanded=config_expanded,
            runtime=self.run_id,
            note=note,
        )
        try:
            if self.write:
                self.writer.add_text("configuration", render)
                logger.info("[TENSORBOARD] Configuration logged to tensorboard")
        except Exception as e:
            logger.error("[TENSORBOARD] Error when logging configuration")
            logger.exception(e)
            raise e

    def unmount_dataloaders(self, method: str) -> None:
        for set_type in ["train", "val", "test"]:
            data_object = getattr(getattr(self.data, method), set_type)
            if hasattr(data_object, "dataloader"):
                data_object.unmount_dataloader()
            elif hasattr(data_object, "dataloaders"):
                data_object.unmount_dataloaders()
            else:
                logger.warning(
                    "No dataloader to unmount for {} in {} set".format(method, set_type)
                )
                
    def check_for_same_run(self, input_config) -> None:
        # Get a list of all the directories in the runs directory
        run_dirs = os.listdir(self.top_log_dir)

        # Loop through each run directory
        for run_dir in run_dirs:
            # Construct the path to the config file for this run
            config_path = os.path.join(self.top_log_dir, run_dir, "config.json")
            # Check if the config file exists
            if os.path.exists(config_path):
                # Load the config file
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Compare the input_config to the config for this run
                if input_config == config:
                    # If there is a match, print a message and return True
                    return True, run_dir
        return False, None
