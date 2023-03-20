from collections import OrderedDict
from copy import deepcopy
from typing import List


def average(models: List["TemporalModel"] = None) -> OrderedDict:
    # Average factor k, and a deepcopy of the first model
    k = 1 / len(models)
    new_model = deepcopy(models[0].get_parameters())

    # Sum the model weights
    if len(models) > 1:
        for target_model in models[1:]:
            target_parameters = target_model.get_parameters()
            for key in new_model.keys():
                new_model[key] += target_parameters[key]

        # Multiply by k to get the average
        for key in new_model.keys():
            new_model[key] *= k

    return new_model
