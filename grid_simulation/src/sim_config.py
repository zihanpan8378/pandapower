import yaml

from collections import defaultdict
from grid_regions import GridRegion
from typing import List
from pandapower import pandapowerNet


class DataCenterConfig:
    """
    Class that represents the configuration for a data center 
    """

    def __init__(
        self, 
        region: GridRegion, 
        load_share: float, 
    ) -> None:
        """
        Initializes the DataCenterConfig object.
        
        Args:
            region: The grid region the data center is located in
            load_share: The share of the total load that the data center is responsible for
        """
        self.region = region
        self.load_share = load_share


class GridConfig:
    """
    Class that represents the configuration for a grid
    """

    def __init__(
        self, 
        enable_weather_variation: bool, 
        onsite_overprovision_factor: float,
        pp_grid: pandapowerNet,
        renewable_share: float
    ) -> None:
        """
        Initializes the GridConfig object. 

        Args:
            enable_weather_variation: A boolean that indicates whether to enable weather variation in the simulation. 
            onsite_overprovision_factor: A float that indicates the overprovision factor for onsite generation capacity compared to the data center load.
            renewable_share: A float that indicates the percentage of renewable energy in the grid.
        """
        self.grid_regions: defaultdict[
            GridRegion, List[DataCenterConfig]
        ]                                           = defaultdict(list)
        self.enable_weather_variation: bool         = enable_weather_variation
        self.onsite_overprovision_factor: float     = onsite_overprovision_factor
        self.pp_grid: pandapowerNet                 = pp_grid
        self.renewable_share: float                 = renewable_share

        
    def add_grid_region(
        self, 
        region: GridRegion, 
        data_center_config: DataCenterConfig
    ) -> None:
        """
        Adds a grid region and its corresponding data center configuration to the grid config.

        Args:
            region: The grid region to add
            data_center_config: The data center configuration corresponding to the grid region
        """
        self.grid_regions[region].append(data_center_config)

    

        
