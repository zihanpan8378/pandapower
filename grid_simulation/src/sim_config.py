from grid_regions import GridRegion
from typing import List
from pandapower import pandapowerNet
from generation_types import GenerationType


class DataCenterConfig:
    """
    Class that represents the configuration for a data center 
    """

    def __init__(
        self, 
        load_share: float, 
        onsite_overprovision_factor: float,
        onsite_generation_source: GenerationType
    ) -> None:
        """
        Initializes the DataCenterConfig object.
        
        Args:
            region: The grid region the data center is located in
            load_share: The share of the total load that the data center is responsible for
            onsite_overprovision_factor: A float that indicates the overprovision factor for 
                                         onsite generation capacity compared to the data center load.
            onsite_generation_source: The type of generation source to use for onsite generation.
        """
        self.load_share                     = load_share
        self.onsite_overprovision_factor    = onsite_overprovision_factor
        self.onsite_generation_source       = onsite_generation_source


class GridConfig:
    """
    Class that represents the configuration for a grid
    """

    def __init__(
        self, 
        pp_grid: pandapowerNet,
        region: GridRegion,
        energy_profile: dict[GenerationType, float]
    ) -> None:
        """
        Initializes the GridConfig object. 

        Args:
            enable_weather_variation: A boolean that indicates whether to enable weather variation in the simulation. 
            onsite_overprovision_factor: A float that indicates the overprovision factor for onsite generation capacity compared to the data center load.
            renewable_share: A float that indicates the percentage of renewable energy in the grid.
            region: The grid region the configuration is for.
            energy_profile: A dictionary mapping generation types to their respective shares in the grid.
        """
        self.pp_grid: pandapowerNet                 = pp_grid
        self.region: GridRegion                     = region
        self.data_centers: List[DataCenterConfig]   = []        
        self.energy_profile: dict[GenerationType, float] = energy_profile

    
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


        
