import pandapower.networks as pn
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

from grid_custom import DatacenterGrid

from typing import List
from pandapower import pandapowerNet

from generation_types import GenerationType
from grid_regions import GridRegion

from sim_config import GridConfig, DataCenterConfig
from grid_observer import GridObserver


_SHIFT_PROPORTION = 0.1


class GridSimulation:
    """
    Class that represents running a mock grid simulation. Requires initial load
    for data centers, can be set as an absolute value or percentage 
    of gross load demand.
    """

    def __init__(
        self, 
        grid_configs: List[GridConfig],
        shifting_threshold: float,
        enable_weather_variation: bool
    ) -> None:
        """
        Constructs an instance of GridSimulation.
        
        Args:
            grid_configs: A list of grid configuration objects.
        """
        self._grid_configs: List[GridConfig] = grid_configs
        self._weather_index: int             = 0
        self._num_grids: int                 = len(grid_configs)
        self._shifting_threshold: float      = shifting_threshold
        self._enable_weather_variation: bool = enable_weather_variation
        self._grids: List[DatacenterGrid]    = []
        self._base_loads: List[float]        = []
 
        # Create all DataCenter grids defined in the simulation configuration
        for grid_config in self._grid_configs:

            grid = DatacenterGrid(
                grid_region = grid_config.region,
                net         = grid_config.pp_grid 
            )
            self._grids.append(grid)
            self._base_loads.append(grid.get_gross_load_demand())

            # Load ids are based on index of data center in grid config
            load_ids = [ idx for idx in range(len(grid_config.data_centers)) ]
            max_power_shares = [ dc_config.load_share for dc_config in grid_config.data_centers ]
            onsite_source_types = [ GenerationType.WIND for _ in grid_config.data_centers ]
            self.__setup_grid(
                grid                = grid,
                grid_config         = grid_config,
                load_ids            = load_ids,
                max_power_shares    = max_power_shares,
                onsite_source_types = onsite_source_types,
                renewable_share     = grid_config.renewable_share
            )
            

        # Create the observer to record data during simnulation
        grid_names = [ grid_config.region.name for grid_config in self._grid_configs ]
        self.observer = GridObserver(
            num_grids  = self._num_grids,
            grid_names = grid_names
        )
        

    def __setup_grid(
        self, 
        grid: DatacenterGrid, 
        grid_config: GridConfig,
        load_ids: List[int], 
        max_power_shares: List[float], 
        onsite_source_types: List[GenerationType],
        renewable_share: float 
    ) -> None:
        """
        Sets up data center loads with onsite generation sources for a given grid.

        The onsite generation sources are used before electricity from other sources
        in the grid are used.
        
        Args:
            grid:                  The data center grid to set up
            load_ids:              The list of load ids for the data centers
            max_powers:            The list of maximum power demands for the data centers, should be in the same order as load_ids
            onsite_source_types:   The list of generation types for the onsite sources, should be in the same order as load_ids
            renewable_share:       The percentage of renewable energy in the grid
        """
        assert len(load_ids) == len(max_power_shares) == len(onsite_source_types)
        base_grid_demand = self._base_loads[-1]

        for load_id, max_power_share, gen_type, dc_config in zip(
            load_ids, 
            max_power_shares, 
            onsite_source_types, 
            grid_config.data_centers
        ):
            self.__setup_data_center_with_backup_source(
                grid=grid,
                load_id=load_id,
                max_power=base_grid_demand * max_power_share,
                onsite_source_type=gen_type,
                onsite_overprovision_factor=dc_config.onsite_overprovision_factor
            )
        
        # grid.assign_grid_gen_percentage_renewable(
        #     renewable_share      = renewable_share,
        #     assign_random_source = False
        # ) 
        grid.simplify_grid()
            

    def __setup_data_center_with_backup_source(
        self,
        grid: DatacenterGrid,
        load_id: int,
        max_power: float,
        onsite_source_type: GenerationType,
        onsite_overprovision_factor: float
    ) -> None:
        """
        Sets up a data center load with an onsite generation source.

        The onsite generation source is used before electricity from other sources
        in the grid are used.
        
        Args:
            grid:                The data center grid to set up the data center on
            load_id:            The load id of the data center to set up
            max_power:          The maximum power demand of the data center
            onsite_source_type: The type of the onsite generation source to create
            onsite_overprovision_factor: The factor by which to overprovision the onsite generation
        """
        grid.assign_new_data_center(
            load_id     = load_id,
            max_power   = max_power
        )

        # Ensures that onsite generation supplies all of dc's idle demand
        grid.create_source_next_to_dc(
            load_id     = load_id,
            max_p_mw    = max_power * onsite_overprovision_factor,
            gen_type    = onsite_source_type
        )


    def step(self) -> None:
        """ 
        Runs the simulation flow, records data including carbon intensity and 
        emissions.
        """
        carbon_intensities = np.zeros(shape=self._num_grids)

        # Power flow required for each grid to update grid metrics
        for idx, grid in enumerate(self._grids):
            grid.run_flow(weather_index=self._weather_index, weather_variation=self._enable_weather_variation)
            print(f"[step]   Grid {idx} carbon intensity: {grid.get_grid_carbon_intensity():.2f} gCO2eq/kWh, carbon emission rate: {grid.get_grid_carbon_emission_rate():.2f} gCO2eq/s")
            self.observer.record_metrics(
                grid_idx=idx, 
                carbon_intensity=grid.get_grid_carbon_intensity(), 
                carbon_emission_rate=grid.get_grid_carbon_emission_rate()
            )
            carbon_intensities[idx] = grid.get_grid_carbon_intensity()

        self._weather_index += 1 


    def get_total_generation(self, grid_idx: int) -> float:
        """
        Gets the total generation in MW for a given grid.

        Args:
            grid_idx: The index of the grid to get total generation for
            
        Returns:
            The total generation in MW for the specified grid
        """
        target_grid = self._grids[grid_idx]
        return target_grid.get_total_generation_mw()

        
    def get_load_demand(self, grid_idx: int, load_id: int) -> float:
        """
        Gets the current load demand for a specific data center in the simulation.

        Args:
            grid_idx: The index of the grid containing the data center
            load_id: The ID of the data center load to get the demand for

        Returns:
            The current load demand for the specified data center
        """
        target_grid = self._grids[grid_idx]
        return target_grid.get_data_center_active_power([load_id])[0]    
            
        # # We shift only if there is a noticeable difference in carbon intensity
        # should_shift = not np.allclose(
        #     carbon_intensities, 
        #     carbon_intensities[0], 
        #     atol=self._shifting_threshold
        # )
        # if not should_shift:
        #     return

        # # Shift data center load based on diffference in carbon intensity
        # lowest_ci_grid_idx = np.argmin(carbon_intensities)
        # target_grid = self._grids[lowest_ci_grid_idx]
        # num_target_dcs = len(target_grid.get_data_center_load_ids())
        # assert num_target_dcs > 0, "Target grid for load shifting has no data centers to shift load to."

        # for idx, grid in enumerate(self._grids):
        #     if idx == lowest_ci_grid_idx:
        #         continue
        #     self.__shift_load(
        #         from_grid = grid,
        #         to_grid   = target_grid
        #     )

        
    def update_dc_loads(
        self, 
        grid_idx: int,
        load_ids: List[int],
        new_loads: List[float]
    ) -> None:
        """
        Updates the load for a specific data center in the simulation.

        Args:
            grid_idx: The index of the grid containing the data center
            load_ids: The IDs of the loads to update
            new_loads: The new load values to set
        """
        target_grid = self._grids[grid_idx]
        target_grid.set_data_center_active_power(
            load_ids = load_ids,
            loads    = new_loads
        )
        
        
    def update_dc_load_share(
        self,
        grid_idx: int,
        load_ids: List[int],
        new_load_shares: List[float]
    ) -> None:
        """
        Updates the load share for a specific data center in the simulation.

        Args:
            grid_idx: The index of the grid containing the data center
            load_ids: The IDs of the loads to update
            new_load_shares: The new load share values to set
        """
        target_grid = self._grids[grid_idx]
        new_loads = [self._base_loads[grid_idx] * share for share in new_load_shares]
        target_grid.set_data_center_active_power(
            load_ids = load_ids,
            loads    = new_loads
        )

        
    def shift_load(
        self, 
        from_grid: DatacenterGrid, 
        to_grid: DatacenterGrid, 
    ) -> None:
        """
        Shifts load for a specific data center from one grid to another.

        Args:
            from_grid:     The grid to shift load from
            to_grid:       The grid to shift load to
        """
        from_dc_ids = from_grid.get_data_center_load_ids()
        from_dc_loads = from_grid.get_data_center_active_power(from_dc_ids)
        shift_amounts = [ load * _SHIFT_PROPORTION for load in from_dc_loads ]
        new_from_dc_loads = [ og - shift for og, shift in zip(from_dc_loads, shift_amounts) ]
        from_grid.set_data_center_active_power(
            load_ids = from_dc_ids,
            loads    = new_from_dc_loads
        )
        total_shift_amount = sum(shift_amounts)
        to_dc_ids = to_grid.get_data_center_load_ids()
        to_dc_loads = to_grid.get_data_center_active_power(to_dc_ids)
        new_to_dc_loads = [ og + total_shift_amount / len(to_dc_ids) for og in to_dc_loads ]
        to_grid.set_data_center_active_power(
            load_ids = to_dc_ids, 
            loads    = new_to_dc_loads
        ) 



# Time period: 2020-08-01 ~ 2020-08-31
# 
# Todo:
#   1. Add background non-dc load and some generators (maybe)
#   2. Add interface to connect to the workload simulator 
#       - input for datacenter power load
#       - output for carbon emission rate (system average and datacenter specific) (and other grid metrics if needed ...)
#   3. Add flutuation of renewable generation (maybe based on weather data)
#       - weather data: /mnt/grid-cloud-migration-estimates_copy/data/weather (four regions CA_ON, US_CAL_CISO, US_MIDA_PJM, US_TEX_ERCO)
#   4. Try on a larger grid (e.g. IEEE 300-Bus System pandapower.networks.power_system_test_cases.case300)