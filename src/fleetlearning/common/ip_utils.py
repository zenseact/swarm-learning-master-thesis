"""Configuration of IP addresses."""
import copy
import secrets

from fleetlearning.common.static_params import global_configs
import yaml


def update_ip_maps(id_bit_length: int = 10) -> dict:
    """Create YAML file with ip-client map.

    Returns:
        dict: ip-client mapping
    """
    nodes = list(global_configs.DEVICE_DICT.keys())
    n_clients = global_configs.NUM_CLIENTS
    n_nodes = len(nodes)

    client_nr_to_node_map = {}
    client_id_to_node_map = {}
    node_num_mapped_clients = dict.fromkeys(nodes, 0)
    client_nr_to_client_id = {}

    for client_number in range(1, n_clients + 1):
        node_id = client_number % n_nodes if client_number % n_nodes != 0 else n_nodes
        client_id = secrets.token_hex(id_bit_length)

        client_nr_to_node_map[client_number] = nodes[node_id - 1]
        client_id_to_node_map[client_id] = nodes[node_id - 1]
        node_num_mapped_clients[nodes[node_id - 1]] += 1
        client_nr_to_client_id[client_number] = client_id

    final_map = {
        "client_nr_to_client_id": client_nr_to_client_id,
        "client_nr_to_node_map": client_nr_to_node_map,
        "client_id_to_node_map": client_id_to_node_map,
        "node_num_mapped_clients": node_num_mapped_clients,
    }

    with open("src/fleetlearning/client_maps.yaml", "w") as file:
        yaml.dump(final_map, file, default_flow_style=False, sort_keys=False)

    return final_map


def update_docker_compose(
    input_file: str = "src/fleetlearning/docker-compose-template.yml",
    output_file: str = "src/fleetlearning/docker-compose.yml",
) -> None:
    """Automatically update the docker-compose file.

    Use the configurations from global_configs and create a docker-compose file based
    on a template.

    Args:
        input_file (str, optional):
            Defaults to "src/fleetlearning/docker-compose-template.yml".
        output_file (str, optional): Defaults to "src/fleetlearning/docker-compose.yml".
    """

    class quoted(str):
        pass

    def quoted_presenter(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

    yaml.add_representer(quoted, quoted_presenter)

    with open(input_file, "r") as file:
        compose_file_template = yaml.safe_load(file)

    new_compose_file = {}
    nr_clients = global_configs.NUM_CLIENTS
    client_depends_on = ["az-server"]
    agx_ips = global_configs.DEVICE_DICT.keys()
    final_map = update_ip_maps()

    # write version and services main part
    new_compose_file["version"] = compose_file_template["version"]
    new_compose_file["services"] = {}

    # write server part
    server_environment = compose_file_template["services"]["az-server"]["environment"]
    server_node = copy.deepcopy(compose_file_template["services"]["az-server"])
    server_node.update(environment=[quoted(*server_environment)])
    new_compose_file["services"]["az-server"] = server_node

    # write agx schedulers
    for agx_nr, agx_ip in enumerate(agx_ips):
        agx_name = f"agx-scheduler-{agx_nr+1}"
        client_depends_on.append(agx_name)
        environment_name = f"constraint:node=={agx_ip}"

        nr_replicas = 0
        if final_map["node_num_mapped_clients"][agx_ip] > 0:
            nr_replicas = 1

        agx_node = copy.deepcopy(compose_file_template["services"]["agx-scheduler"])
        agx_node.update(environment=[quoted(environment_name)])
        agx_node["deploy"]["replicas"] = nr_replicas
        new_compose_file["services"][agx_name] = agx_node

    # write client parts
    client_node = copy.deepcopy(compose_file_template["services"]["vv-client"])
    client_node["deploy"]["replicas"] = nr_clients
    client_node["depends_on"] = client_depends_on
    client_environment = compose_file_template["services"]["vv-client"]["environment"]
    client_node.update(environment=[quoted(*client_environment)])
    new_compose_file["services"]["vv-client"] = client_node

    with open(output_file, "w") as file:
        yaml.dump(new_compose_file, file, default_flow_style=False, sort_keys=False)
