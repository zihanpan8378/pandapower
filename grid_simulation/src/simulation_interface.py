import pandapower.networks as pn
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

from grid_custom import DatacenterGrid
from sim_config import GridSimConfig
from constants import RENEWABLE_SOURCE_TYPES, OVERPROVISION_FACTOR

from typing import List
from pandapower import pandapowerNet


_SHIFT_PROPORTION = 0.3


class GridObserver:
    """
    Class that represents an observer of the grid simulation, can be used to record
    and plot various grid metrics such as carbon intensity and emissions. 
    """
    
    def __init__(self, num_grids: int, grid_names: List[str]) -> None:
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
            grid_idx: The index of the grid to record metrics for
            carbon_intensity: The carbon intensity value to record
            carbon_emission_rate: The carbon emission rate value to record
        """
        self._ci_data[grid_idx].append(carbon_intensity)
        self._ce_data[grid_idx].append(carbon_emission_rate)


    def plot_ci_data(self, save_path: str = "carbon_intensity.png") -> None:
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

    def __init__(self, sim_config: str, raw_grids: List[pandapowerNet]) -> None:
        """
        Constructs an instance of GridSimulation.
        
        Args:
            sim_config: The path to the simulation configuration yaml file.
            raw_grids: The list of pandapower networks to use as the underlying grids for the simulation. 
                       Should be in the same order as the grids defined in the sim_config yaml file.
        """
        config = GridSimConfig(config_yaml = sim_config)
        self._grids: List[DatacenterGrid] = []
        grid_names: List[str] = []
        num_grids = len(config.grid_config)
        
        assert num_grids == len(raw_grids)

        # Create all DataCenter grids defined in the simualtion configuration
        for idx, grid_cfg in enumerate(config.grid_config):
            region = grid_cfg["region"]
            print(f"[init] Setting up grid {idx} (region: {region})")
            net = raw_grids[idx]
            self._grids.append(DatacenterGrid(net=net, grid_region=region))
            self._grids[-1].simplify_grid()
            grid_names.append(region)
            gross_load_demand = self._grids[-1].get_gross_load_demand()
            print(f"[init]   Gross load demand: {gross_load_demand:.2f} MW")

            # For each grid, initialize data centers and configure their power demand
            load_ids = list(self._grids[-1]._net.load.index)
            num_dcs = len(grid_cfg["dc_init_power_shares"])
            print(f"[init]   Assigning {num_dcs} data center(s)")
            for i, power_percentage in enumerate(grid_cfg["dc_init_power_shares"]):
                assign_dc_success: bool = self._grids[-1].assign_new_data_center(
                    load_id=load_ids[i],
                    idle_power=gross_load_demand * power_percentage
                )
                assert assign_dc_success

            # For each grid, initialize renewable sources (if any)
            # Gross load demand now includes data center loads
            gross_load_demand = self._grids[-1].get_gross_load_demand()
            num_renewable_sources = grid_cfg["num_renewable_sources"]
            print(f"[init]   Assigning {num_renewable_sources} renewable source(s)")

            # TODO: Maybe add more complicated energy profile for each renewable sources
            gen_ids = list(self._grids[-1]._net.gen.index)
            for i in range(num_renewable_sources):
                # Randomly choose a generation type
                gen_type = random.choice(RENEWABLE_SOURCE_TYPES)
                assign_renewable_success: bool = self._grids[-1].assign_renewable_source(
                    gen_id=gen_ids[i],
                    max_p_mw=gross_load_demand * OVERPROVISION_FACTOR / num_renewable_sources,
                    gen_type=gen_type
                )
                assert assign_renewable_success
        
        # Create the observer to record data during simnulation
        self._observer = GridObserver(
            num_grids  = num_grids,
            grid_names = grid_names
        )
        

    def step(self, shifting_enabled: bool, weather_index: int) -> None:
        """ 
        Runs the simulation flow, records data including carbon intensity and 
        emissions.
        """
        carbon_intensities = np.zeros(shape=len(self._grids))
        # Power flow required for each grid to update grid metrics
        for idx, grid in enumerate(self._grids):
            grid.run_flow(weather_index=weather_index)
            self._observer.record_metrics(
                grid_idx=idx, 
                carbon_intensity=grid.get_grid_carbon_intensity(), 
                carbon_emission_rate=grid.get_grid_carbon_emission_rate()
            )
            carbon_intensities[idx] = grid.get_grid_carbon_intensity()

        if not shifting_enabled:
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

            # TODO: Right now, shifting done by shifting a percntage of original workload
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

    def generate_plots(self) -> None:
        """
        Generates and saves all relevant plots for the simulation results.
        """
        self._observer.plot_ci_data(save_path="carbon_intensity.png")


    # def plot_ci(self) -> None:
    #     """
    #     Plots the average carbon intensity over time for each data center grid.
    #     """
    #     # Safety check in case plot is called before run
    #     if not self._ci_data:
    #         print("No data to plot. Please call run() first.")
    #         return

    #     plt.figure(figsize=(10, 6))
    #     num_grids = len(self._ci_data)

    #     # Extract and plot the data for each grid separately
    #     for i in range(num_grids):
    #         grid_ci = self._ci_data[i]
    #         plt.plot(
    #             range(len(grid_ci)), 
    #             grid_ci, 
    #             label=f"Grid {i} (Datacenter)", 
    #             linewidth=2
    #         )

    #     # Formatting the chart for readability
    #     plt.title("Regional Grid Carbon Intensity Over Time")
    #     plt.xlabel("Time Step (Weather Index)")
    #     plt.ylabel("Carbon Intensity")
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(loc="upper right")
    #     plt.tight_layout()
        
    #     # Display the plot
    #     plt.show()


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