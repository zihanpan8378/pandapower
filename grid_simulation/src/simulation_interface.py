import pandapower.networks as pn
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

from grid_custom import DatacenterGrid
from constants import OVERPROVISION_FACTOR

from typing import List
from pandapower import pandapowerNet

from generation_types import GenerationType
from grid_regions import GridRegion

from sim_config import GridConfig, DataCenterConfig


_SHIFT_PROPORTION = 0.1


class GridObserver:
    """
    Class that represents an observer of the grid simulation, can be used to record
    and plot various grid metrics such as carbon intensity and emissions. 
    """
    
    def __init__(self, num_grids: int, grid_names: List[str]) -> None:
        """
        Constructs an instance of GridObserver.

        Args:
            num_grids:  The number of grids in the simulation, used to initialize data structures for recording metrics.
            grid_names: A list of names for each grid.
        """
        self._ci_data: List[List[float]] = [
            [] for _ in range(num_grids)
        ]
        self._ce_data: List[List[float]] = [
            [] for _ in range(num_grids)
        ]
        self._grid_names = grid_names
        

    def record_metrics(self, grid_idx: int, carbon_intensity: float, carbon_emission_rate: float) -> None:
        """
        Records the carbon intensity and carbon emission rate for a given grid at a specific time step.

        Args:
            grid_idx:               The index of the grid to record metrics for
            carbon_intensity:       The carbon intensity value to record
            carbon_emission_rate:   The carbon emission rate value to record
        """
        self._ci_data[grid_idx].append(carbon_intensity)
        self._ce_data[grid_idx].append(carbon_emission_rate)


    def plot_ci_data(self, save_path: str = "carbon_intensity.png") -> None:
        """
        Plots the carbon intensity data recorded for each grid over time. 

        Args:
            save_path: The path to save the generated plot image.
        """

        plt.figure(figsize=(10, 6))
        num_grids = len(self._ci_data)

        for idx in range(num_grids):
            grid_ci = self._ci_data[idx]
            plt.plot(range(len(grid_ci)), grid_ci,
                    label=self._grid_names[idx], linewidth=2)

        plt.title("Regional Grid Carbon Intensity Over Time")
        plt.xlabel("Time Step (Weather Index)")
        plt.ylabel("Carbon Intensity")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        

class GridSimulation:
    """
    Class that represents running a mock grid simulation. Requires initial load
    for data centers, can be set as an absolute value or percentage 
    of gross load demand.
    """

    def __init__(
        self, 
        grid_config: GridConfig,
        enable_shifting: bool,
        shifting_threshold: float
    ) -> None:
        """
        Constructs an instance of GridSimulation.
        
        Args:
            grid_config: The grid configuration object.
        """
        self._grid_config: GridConfig       = grid_config
        self._weather_index: int            = 0
        self._num_grids: int                = len(self._grid_config.grid_regions)
        self._enable_shifting: bool         = enable_shifting
        self._shifting_threshold: float     = shifting_threshold
        self._grids: List[DatacenterGrid]   = []

        # Create all DataCenter grids defined in the simulation configuration
        for grid_region, dc_configs in self._grid_config.grid_regions.items(): 

            grid = DatacenterGrid(
                grid_region = grid_region,
                net         = self._grid_config.pp_grid 
            )
            self._grids.append(grid)

            load_ids = [ idx for idx in range(len(dc_configs)) ]
            idle_powers = [ dc_config.load_share for dc_config in dc_configs ]
            onsite_source_types = [ GenerationType.SOLAR for _ in dc_configs ]
            self.__setup_grid(
                grid                = grid,
                load_ids            = load_ids,
                idle_powers         = idle_powers,
                onsite_source_types = onsite_source_types,
                renewable_share     = self._grid_config.renewable_share
            )

        # Create the observer to record data during simnulation
        grid_names = [ region.name for region in self._grid_config.grid_regions.keys() ]
        self._observer = GridObserver(
            num_grids  = self._num_grids,
            grid_names = grid_names
        )
        

    def __setup_grid(
        self, 
        grid: DatacenterGrid, 
        load_ids: List[int], 
        idle_powers: List[float], 
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
            idle_powers:           The list of idle power demands for the data centers, should be in the same order as load_ids
            onsite_source_types:   The list of generation types for the onsite sources, should be in the same order as load_ids
            renewable_share:       The percentage of renewable energy in the grid
        """
        assert len(load_ids) == len(idle_powers) == len(onsite_source_types)

        for load_id, idle_power, gen_type in zip(load_ids, idle_powers, onsite_source_types):
            self.__setup_data_center_with_backup_source(
                grid=grid,
                load_id=load_id,
                idle_power=idle_power,
                onsite_source_type=gen_type
            )
        
        grid.assign_grid_gen_percentage_renewable(
            renewable_share      = renewable_share,
            assign_random_source = False
        ) 
            

    def __setup_data_center_with_backup_source(
        self,
        grid: DatacenterGrid,
        load_id: int,
        idle_power: float,
        onsite_source_type: GenerationType
    ) -> None:
        """
        Sets up a data center load with an onsite generation source.

        The onsite generation source is used before electricity from other sources
        in the grid are used.
        
        Args:
            load_id:            The load id of the data center to set up
            idle_power:         The idle power demand of the data center
            onsite_source_type: The type of the onsite generation source to create
        """
        assign_dc_success: bool = grid.assign_new_data_center(
            load_id     = load_id,
            idle_power  = idle_power
        )
        assert assign_dc_success

        # Ensures that onsite generation supplies all of dc's idle demand
        assign_renewable_success: bool = grid.create_source_next_to_dc(
            load_id     = load_id,
            max_p_mw    = idle_power * OVERPROVISION_FACTOR,
            gen_type    = onsite_source_type
        )
        assert assign_renewable_success
        

    def step(self) -> None:
        """ 
        Runs the simulation flow, records data including carbon intensity and 
        emissions.
        """
        carbon_intensities = np.zeros(shape=self._num_grids)

        # Power flow required for each grid to update grid metrics
        for idx, grid in enumerate(self._grids):
            
            grid.run_flow(weather_index=self._weather_index, weather_variation=self._grid_config.enable_weather_variation)
            print(f"[step]   Grid {idx} carbon intensity: {grid.get_grid_carbon_intensity():.2f} gCO2eq/kWh, carbon emission rate: {grid.get_grid_carbon_emission_rate():.2f} gCO2eq/s")
            self._observer.record_metrics(
                grid_idx=idx, 
                carbon_intensity=grid.get_grid_carbon_intensity(), 
                carbon_emission_rate=grid.get_grid_carbon_emission_rate()
            )
            carbon_intensities[idx] = grid.get_grid_carbon_intensity()

        if not self._grid_config.enable_weather_variation:
            return

        # We shift only if there is a noticeable difference in carbon intensity
        should_shift = not np.allclose(
            carbon_intensities, 
            carbon_intensities[0], 
            atol=self._shifting_threshold
        )
        if not should_shift:
            return

        # Shift data center load based on diffference in carbon intensity
        lowest_ci_grid_idx = np.argmin(carbon_intensities)
        target_grid = self._grids[lowest_ci_grid_idx]
        target_region_dc_ids: List[int] = target_grid.get_data_center_load_ids()
        target_region_dc_loads: List[float] = target_grid.get_data_center_active_power(
            load_ids = target_region_dc_ids
        )
        total_power_shift = 0.0

        for idx, grid in enumerate(self._grids):
            if idx == lowest_ci_grid_idx:
                continue

            # TODO: Right now, shifting done by shifting a percentage of original workload
            # Try shifting amount proprotional to difference in carbon intensity
            # diff_ci = carbon_intensities[idx] - carbon_intensities[lowest_ci_grid_idx]
            dc_ids = grid.get_data_center_load_ids()
            dc_loads = grid.get_data_center_active_power(dc_ids) 

            shift_amounts = [ load * _SHIFT_PROPORTION for load in dc_loads ]
            new_dc_loads = [ og - shift for og, shift in zip(dc_loads, shift_amounts) ]
            grid.set_data_center_active_power(
                load_ids = dc_ids, 
                loads    = new_dc_loads
            )
            total_power_shift += sum(shift_amounts)

        num_target_dcs = len(target_region_dc_ids)
        assert num_target_dcs > 0, "Target grid has no data centers to shift load to!"
        
        new_target_dc_power = [ (load + total_power_shift / num_target_dcs) for load in target_region_dc_loads]
        target_grid.set_data_center_active_power(
            load_ids = target_region_dc_ids,
            loads    = new_target_dc_power
        )

        
    def __shift_load(
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


    def generate_plots(self) -> None:
        """
        Generates and saves all relevant plots for the simulation results.
        """
        self._observer.plot_ci_data(save_path="carbon_intensity.png")


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