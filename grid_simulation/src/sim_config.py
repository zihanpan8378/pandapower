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
        load_share: float, 
        onsite_overprovision_factor: float
    ) -> None:
        """
        Initializes the DataCenterConfig object.
        
        Args:
            region: The grid region the data center is located in
            load_share: The share of the total load that the data center is responsible for
            onsite_overprovision_factor: A float that indicates the overprovision factor for 
                                         onsite generation capacity compared to the data center load.
        """
        self.load_share                     = load_share
        self.onsite_overprovision_factor    = onsite_overprovision_factor


class GridConfig:
    """
    Class that represents the configuration for a grid
    """

    def __init__(
        self, 
        pp_grid: pandapowerNet,
        renewable_share: float,
        region: GridRegion
    ) -> None:
        """
        Initializes the GridConfig object. 

        Args:
            enable_weather_variation: A boolean that indicates whether to enable weather variation in the simulation. 
            onsite_overprovision_factor: A float that indicates the overprovision factor for onsite generation capacity compared to the data center load.
            renewable_share: A float that indicates the percentage of renewable energy in the grid.
            region: The grid region the configuration is for.
        """
        self.pp_grid: pandapowerNet                 = pp_grid
        self.renewable_share: float                 = renewable_share
        self.region: GridRegion                     = region
        self.data_centers: List[DataCenterConfig]   = []        

    
    def add_data_center(self, data_center_config: DataCenterConfig) -> int:
        """
        Adds a data center configuration to the grid configuration.

        Args:
            data_center_config: The configuration for the data center to be added

        Returns:
            The load id for the added data center.
        """
        self.data_centers.append(data_center_config)
        return len(self.data_centers) - 1


        
