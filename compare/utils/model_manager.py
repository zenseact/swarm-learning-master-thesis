import importlib
import logging

from torch.nn import Module


def load_model(config, **kwargs) -> Module:
    name = config["model"]["name"]
    config_model_args = config["model"]["args"]

    module = importlib.import_module(f"models.{name}")
    model = module.metadata["model"]

    # create instance
    try:
        if kwargs:
            logging.info(
                "Using inline supplied arguments (not from config) to create model instance"
            )
            instance = model(**kwargs)
        else:
            logging.info("Using config arguments to create model instance")
            img_size = config["data"]["img_size"]
            instance = model(img_size=img_size, **config_model_args)
        logging.info("Loaded %s", module.metadata["name"])
        return instance
    except Exception as e:
        raise (e)
