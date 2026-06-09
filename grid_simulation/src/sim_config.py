import yaml

from typing import List


class GridSimConfig:
    """
    Class that handles loading the grid simulator config from yaml file.
    """

    def __init__(self, config_yaml: str) -> None:
        """
        Creates an instance of the GridSimConfig class. 

        Args:
            config_yaml: The path to the config file to load.
        """
        data: dict
        with open(config_yaml, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")

        all_grids = data["grids"] 
        self.grid_config: List[dict] = [item for _, item in all_grids.items()]
