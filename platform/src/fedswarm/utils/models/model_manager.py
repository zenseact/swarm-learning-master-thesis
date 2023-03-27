import importlib
import logging
import torch

logger = logging.getLogger(__name__)


def load_model_from_name(name, **kwargs) -> torch.nn.Module:
    module = importlib.import_module(f"fedswarm.utils.models.{name}")
    model = module.metadata["model"]
    return model(**kwargs)

def load_model(config, **kwargs) -> torch.nn.Module:
    name = config["model"]["name"]
    config_model_args = config["model"]["args"]

    module = importlib.import_module(f"fedswarm.utils.models.{name}")
    model = module.metadata["model"]
    

    # create instance
    try:
        if kwargs:
            logger.info(
                "Using inline supplied arguments (not from config) to create model instance"
            )
            instance = model(**kwargs)
        else:
            logger.info("Using config arguments to create model instance")
            img_size = config["data"]["img_size"]
            instance = model(img_size=img_size, **config_model_args)
        logger.info("Loaded %s", module.metadata["name"])
        return instance
    except Exception as e:
        logger.error("Error loading model: %s", e)
        logger.exception(e)
        raise (e)

def load_model_from_checkpoint(model_path, config):
    # Define the device to be used (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    model_params = torch.load(model_path)
    
    # Load the model name from config
    model_name = config["model"]["name"]
    model_args = config["model"]["args"]

    # Load the model
    model = load_model_from_name(model_name, **model_args)
    model.load_state_dict(model_params)

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the specified device
    model.to(device)
    

    return model
