from datetime import datetime
import importlib 
import logging


from utils.training.swarm import SimulationRunner

from utils.training.topologies import fully_connected_centralised, edgeless_graph

logger = logging.getLogger(__name__)


def synchronous_fixed_rounds(simulator: SimulationRunner, topology_generator: callable):
    """Synchronous training with fixed number of rounds."""
        
    for i in range(0, simulator.config["swarm"]["global"]["global_rounds"]):
        logger.debug("Starting new round: {}".format(i + 1))
        start_time = datetime.now()
        
        simulator.train()
        simulator.set_topology(topology_generator)
        simulator.propagate_topology()
        simulator.start_communication()
        
        # Calculate the training duration
        end_time = datetime.now()
        soft_duration = str(end_time - start_time).split(".")[0]
        logger.debug("Round finished after {}".format(soft_duration))


# Fully connected
def synchronous_fixed_rounds_fc(simulator: SimulationRunner):
    synchronous_fixed_rounds(simulator, fully_connected_centralised)

# Edgeless
def synchronous_fixed_rounds_edgeless(simulator: SimulationRunner):
    synchronous_fixed_rounds(simulator, edgeless_graph)

