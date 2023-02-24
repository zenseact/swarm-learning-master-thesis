import logging
import jsonschema
import json
import ray 

from pathlib import Path
from jinja2 import Template
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process

from utils.data import DataHandler
from utils.training import run_centralised, run_federated, run_swarm

logger = logging.getLogger(__name__)

with open("utils/templates/config_schema.json") as f:
    CONFIG_SCHEMA = json.load(f)


class Platform:
    def __init__(self, config) -> None:
        # Save config
        self.config = config
        # Create run and relevant directories
        self.top_log_dir = "runs"
        self.create_if_not_exists(self.top_log_dir)
        self.run_id = self.create_run()
        self.run_dir = Path(self.top_log_dir, self.run_id)
        self.create_if_not_exists(self.run_dir)
        logger.info("New run created: {}".format(self.run_id))
        
        # Check if config is valid
        self.validate_config()

        # Set up tensorboard writer
        self.writer = SummaryWriter(self.run_dir)
        
        # Init ray if required
        if "federated" in self.methods or "swarm" in self.methods:
            _start_ray(config)

        # Set up logging to file and set format
        logging.basicConfig(
            filename="{}.log".format(Path(self.run_dir, "platform")),
            encoding="utf-8",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

        # Run configuration
        self.tb_log_config()
        self.data = DataHandler(config, self.run_dir)

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
            self.announce_configuration(self.config)
        except jsonschema.exceptions.ValidationError as e:
            logger.error("Invalid configuration")
            logger.exception(e)
            raise e

    def announce_configuration(self, config: dict) -> None:
        self.methods = [
            x for x in config if "train" in config[x] and config[x]["train"] == "true"
        ]
        logger.info("[CONFIG] Enabled methods: {}".format(self.methods))
        logger.info(
            "[CONFIG] Data allowance: {:.2%} of available data".format(
                config["data"]["ratio"]
            )
        )
        logger.info("[CONFIG] More information in tensorboard")

    def tb_log_config(self) -> None:
        logger.debug("Logging configuration to tensorboard")
        config = json.dumps(self.config, indent=4)
        with open("utils/templates/config_message.md", "r") as file:
            logger.debug("Reading configuration template")
            render = Template(file.read()).render(
                config=config,
                runtime=self.run_id,
            )
        try:
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
                

def _start_ray(config: dict) -> None:
    try:
        logger.info("Initialising Ray runtime")
        ray.init(**config["swarm"]["global"]["ray_init_args"])
    except Exception as e:
        logger.error("Error initialising Ray runtime: {}".format(e))
        logger.exception(e)
        raise e
