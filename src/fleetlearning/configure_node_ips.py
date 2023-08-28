"""Configuration of ips to AGXs."""

from fleetlearning.common.ip_utils import update_docker_compose, update_ip_maps


def main() -> None:
    """Update parameters from global config."""
    update_ip_maps()
    update_docker_compose()


if __name__ == "__main__":
    main()
