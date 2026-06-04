import os
import numpy as np
import pandapower as pp
import math
import pandas as pd

from pandapower import pandapowerNet
from typing import List, Dict

from run_bialek import calculate_source_to_load_contributions, gross_gen_demand
from run_bialek import _DEMAND_SOURCES, _GENERATION_SOURCES, _BRANCH_SPECS


# Define paths for weather data per region
weather_data_base_path = "/mnt/grid-cloud-migration-estimates_copy/data/weather"
_WEATHER_DATA_PATHS: Dict[str, str] = {
    "CA_ON": os.path.join(weather_data_base_path, "processed", "CA-ON.csv"),
    "US_CAL_CISO": os.path.join(weather_data_base_path, "processed", "US-CAL-CISO.csv"),
    "US_MIDA_PJM": os.path.join(weather_data_base_path, "processed", "US-MID-PJM.csv"),
    "US_TEX_ERCO": os.path.join(weather_data_base_path, "processed", "US-TEX-ERCO.csv"),
}

# TODO: change energy costs, other constants to be per type based
_CARBON_INTENSITY: float = 95.0
_NONRENEWABLE_BASE_COST: float = 100_000_000.0
_RENEWABLE_BASE_COST: float = 100.0


_ALLOWED_GRID_REGIONS: List[str] = [
    "CA_ON",
    "US_CAL_CISO",
    "US_MIDA_PJM",
    "US_TEX_ERCO"
]

_RENEWABLE_SOURCE_TYPES: List[str] = [
    "solar",
    "wind"
]


# class DataCenter:
#     """
#     Class that represents a data center load. 

#     Attributes:
#         _idle_power: The idle power of the data center
#         _max_power: The max power of the data center
#         _capacity: The capacity of the data center
#         _id: The id of the data center
#     """

#     def __init__(
#         self, 
#         idle_power: float, 
#         max_power: float, 
#         capacity: float,
#         id: int
#     ) -> None:
#         """
#         Constructs an instance of a data center.

#         Args:
#             idle_power: the idle power
#             max_power: the max power
#             capacity: the capacity
#             id: the id
#         """
#         self._idle_power = idle_power
#         self._max_power = max_power
#         self._capacity = capacity
#         self._current_load: float = 0.0
#         self._id = id


#     def get_active_power_demand_MW(self) -> float:
#         """
#         Returns the active power demand of this data center based on current load.

#         Returns:
#             The active power demand in MW
#         """
#         utilization = self._current_load / self._capacity
#         return self._idle_power + (self._max_power - self._idle_power) * utilization


#     def set_load(self, new_load: float) -> None:
#         """
#         Sets the load of this data center.

#         Args:
#             new_load: The new load
#         """
#         self._current_load = new_load


#     def get_id(self) -> int:
#         """
#         Gets the data center's id. 

#         Returns:
#             The id of the data center
#         """
#         return self._id


class ProcessGrid:
    """
    Class that contains helper methods to modify the grid for simulation.
    """

    def sync_generator_limits(self, net: pandapowerNet, renewable_sources: List[int]) -> None:
        """
        Updates the generation limits for non-renewable grid generation components.

        Args:
            net: The pandapower network to modify
            renewable_sources: The list of generator ids that are renewable sources
        """
        for elem, _ in _GENERATION_SOURCES:

            # Right now, only generators can be set as renewable sources
            # We handle the "gen" case separately
            if elem == "gen" and len(net[elem]) > 0:
                non_ren_mask = ~net.gen.index.isin(renewable_sources)
                net.gen.loc[non_ren_mask, "max_p_mw"] = 99999.0

            # Set the max power of the generation source to be virtually uncapped
            elif len(net[elem]) > 0:
                net[elem]["max_p_mw"] = 99999.0


    def sync_generator_costs(self, net: pandapowerNet, renewable_sources: List[int]) -> None:
        """
        Updates costs for all elements in poly_cost. 
        
        'Gen' elements marked as renewable will be set to have a cheap electricity prices.
        Other elements will have expensive electricity prices. This is so pandapower dc optimal power 
        flow will dispatch the renewable energy sources before the non-renewable ones.

        Args:
            net: The pandapower network to modify 
            renewable_sources: The list of 'gen' element ids that represent renewable sources
        """
        # An element is only renewable if it is a 'gen' and its ID is in the renewable list
        is_gen = net.poly_cost["et"] == "gen"
        is_renewable_gen = is_gen & net.poly_cost["element"].isin(renewable_sources)
        
        # Apply costs: Renewable if it matches the mask above, non-renewable for everything else
        net.poly_cost["cp0_eur"] = np.where(
            is_renewable_gen,
            _RENEWABLE_BASE_COST,
            _NONRENEWABLE_BASE_COST
        )
        net.poly_cost["cp2_eur_per_mw2"] = np.where(
            is_renewable_gen,
            _RENEWABLE_BASE_COST,
            _NONRENEWABLE_BASE_COST
        )


    def modify_non_datacenter_load(self, net: pandapowerNet, data_centers: List[int]) -> None:
        """
        Updates the active power of non data center loads to be 0.
        
        Args:
            net: The pandapower network to modify
            data_centers: The list of load ids that are data centers
        """
        non_data_center_mask = ~net.load.index.isin(data_centers) 
        net.load.loc[non_data_center_mask, "p_mw"] = 0.0


    def unconstrain_tranmission(self, net: pandapowerNet) -> None:
        """
        Helper function to update the transmission components of the grid 
        to have (virtually) limitless capacity.

        Args:
            net: The pandapower network to update
        """
        net.line['max_i_ka'] = 99999.0

        # 2. Unconstrain transformers (Limit is in MVA)
        if len(net.trafo) > 0:
            net.trafo['sn_mva'] = 99999.0

        # 3. Unconstrain impedance branches (Limit is in MVA)
        if len(net.impedance) > 0:
            net.impedance['sn_mva'] = 99999.0

        # 4. Unconstrain DC lines (Limit is in MW)
        if len(net.dcline) > 0:
            net.dcline['max_p_mw'] = 99999.0

        net.line['max_loading_percent'] = 99999.0
        if len(net.trafo) > 0:
            net.trafo['max_loading_percent'] = 99999.0 


class DatacenterGrid:
    """
    Class that represents a electric grid, containing data centers and renewable energy sources.

    Attributes:
        _net: The underlying pandapower network
        _data_centers: Maps pandapower load id to DataCenter instance
        _renewable_sources: List of pandapower gen id that are renewable sources
    """

    def __init__(self, net: pandapowerNet, grid_region: str) -> None:
        """
        Creates an instance of the DataCenterGrid class.

        Args:
            net: The underlying pandapower network
            grid_region: The region that the grid is in, used for determining
                         energy generation at a given time

        Raises:
            ValueError: If the grid_region is not an allowed region
        """

        if grid_region not in _ALLOWED_GRID_REGIONS:
            raise ValueError(
                f"Invalid Grid Region: {grid_region} is not in {_ALLOWED_GRID_REGIONS}"
            )

        self._net = net
        self._data_centers: List[int] = []       # List of load indexes marked as data centers
        
        self._renewable_sources: List[int] = []  # List of gen indexes marked as renewable
        self._renewable_source_types: List[str] = []
        self._renewable_source_max_pmw: List[float] = []
        self._weather_data = pd.read_csv(_WEATHER_DATA_PATHS[grid_region])

        # We take the maximum wind speed to support proportional power scaling
        # based on wind speed at given time
        # We don't need to do this for cloud cover, which is already in interval[0, 100]
        self._max_wind_speed = self._weather_data.loc[:, "windspeed_10m"].max()
        self.process_grid = ProcessGrid()


    def assign_new_data_center(
        self, 
        load_id: int,
        idle_power: float
    ) -> bool:
        """
        Marks an existing load as a data center. 

        Args:
            load_id: the id of the load to mark as a data center
            idle_power: the idle power of the data center

        Returns:
            True if the data center was marked succesfully, false otherwise
        """

        if (load_id not in self._net.load.index.values
            or load_id in self._data_centers):
            return False

        self._data_centers.append(load_id)
        self._net.load.loc[load_id, 'p_mw'] = idle_power
        return True


    def set_data_center_active_power(
        self, 
        load_ids: List[int], 
        loads: List[float]
    ) -> bool:
        """
        Sets the active power of the data center loads.
        """
        # Ensure that all loads being assigned are data centers
        if not all(load_id in self._data_centers for load_id in load_ids):
            return False

        # Ensure data consistency
        if len(load_ids) != len(loads):
            return False

        for load_id, load in zip(load_ids, loads):
            self._net.load.loc[load_id, "p_mw"] = load

        return True

    
    def get_grid_carbon_intensity(self) -> float:
        """
        Calculates the grid's average carbon intensity (e.g., lbs CO2 / MWh).

        Returns:
            float: The grid's average carbon intensity.
        """
        total_generation_mw = 0.0
        total_carbon_emissions = 0.0

        for elem, res in _GENERATION_SOURCES:
            # 1. Extract the power output for all generators of this type
            p_mw = self._net[res]["p_mw"]
            
            # 2. Add to the absolute total generation pool
            total_generation_mw += p_mw.sum()

            # 3. Calculate emissions based on actual MW generated, not count
            if elem == "gen":
                # Mask out the renewables
                non_renewable_mask = ~self._net[res].index.isin(self._renewable_sources)
                
                # Extract MW only for non-renewables and calculate emissions
                dirty_mw = p_mw.loc[non_renewable_mask]
                total_carbon_emissions += (dirty_mw * _CARBON_INTENSITY).sum()
            else:
                # Assuming all non-'gen' sources (like ext_grid) in your framework 
                # are treated as emitting sources based on your original else block.
                total_carbon_emissions += (p_mw * _CARBON_INTENSITY).sum()

        # 4. Avoid division by zero if the grid is completely dead
        if total_generation_mw == 0:
            return 0.0

        # 5. Calculate and return the average intensity
        average_intensity = total_carbon_emissions / total_generation_mw
        
        return float(average_intensity)
        
        
    def get_data_center_carbon_intensity(self, load_index: int) -> float:
        """
        Calculates the carbon intensity (emissions per MW consumed) 
        allocated to a specific data center load via Bialek's tracing.
        
        Args:
            load_index: The index of the data center load in net.load.

        Returns:
            The carbon intensity attributable to this load.
        """
        # 1. Get the Pandapower bus ID for the target load
        bus_pp_id = self._net.load.loc[load_index, "bus"]
        
        # 2. Get the positional index rapidly using C-optimized .get_loc()
        bus_df_id = self._net.bus.index.get_loc(bus_pp_id)

        # 3. Create the mask for non-renewable buses
        renewable_buses_pp_ids = self._net.gen.loc[self._renewable_sources, "bus"].values
        is_renewable_mask = self._net.bus.index.isin(renewable_buses_pp_ids)
        non_renewable_mask = ~is_renewable_mask

        # 4. Retrieve the Bialek tracing distribution matrix
        E_distribution_mw, _, _ = calculate_source_to_load_contributions(self._net)
        
        # 5. Calculate total emissions rate from dirty sources
        non_renewable_contributions_mw = E_distribution_mw[non_renewable_mask, bus_df_id]
        total_carbon_emissions = (non_renewable_contributions_mw * _CARBON_INTENSITY).sum()

        # 6. Calculate total power delivered to this bus from ALL sources (the denominator)
        total_power_delivered_mw = E_distribution_mw[:, bus_df_id].sum()

        # 7. Handle edge case where the data center is disconnected/offline
        if total_power_delivered_mw == 0:
            return 0.0

        # 8. Calculate intensity
        carbon_intensity = total_carbon_emissions / total_power_delivered_mw

        return float(carbon_intensity)   
    

    def get_grid_carbon_emissions(self) -> float:
        """
        Calculates the grid's total carbon emissions.

        Returns:
            The grid's carbon total carbon emissions
        """
        carbon_emissions = 0.0

        # Handle generators separately, which consist of renewable and non-renewable sources
        non_renewable_mask = ~self._net["res_gen"].index.isin(self._renewable_sources)
        non_renewable_p_mw = self._net["res_gen"].loc[non_renewable_mask, "p_mw"]
        carbon_emissions += (non_renewable_p_mw * _CARBON_INTENSITY).clip(lower=0).sum()

        for elem, res in _GENERATION_SOURCES:

            # Skip generators since we accounting for them already
            if elem == "gen":
                continue

            # Accumulate the carbon intensity rates, since non-renewable
            elem_carbon_intensity_rates = self._net[res]["p_mw"] * _CARBON_INTENSITY
            carbon_emissions += elem_carbon_intensity_rates.clip(lower=0).clip(lower=0).sum()

        return carbon_emissions


    def get_data_center_carbon_emissions(self, load_index: int) -> float:
        """
        Calculates the total carbon emissions
        allocated to a specific data center load via Bialek's tracing.
        
        Args:
            load_index: The index of the data center load in net.load.

        Returns:
            The carbon emissions attributable to this load.
        """
        bus_pp_id = self._net.load.loc[load_index, "bus"]
        bus_pp_to_df = {
            bus_id: idx for idx, bus_id in enumerate(self._net.bus["bus_id"])
        }
        bus_df_id = bus_pp_to_df[bus_pp_id]

        renewable_buses_pp_ids = self._net.gen.loc[self._renewable_sources, "bus"].values

        is_renewable_mask = self._net.bus["bus_id"].isin(renewable_buses_pp_ids)
        non_renewable_mask = ~is_renewable_mask

        E_distribution_mw, _, _ =  calculate_source_to_load_contributions(self._net)
        non_renewable_contributions_mw = E_distribution_mw[non_renewable_mask, bus_df_id]

        carbon_emissions_rate = (non_renewable_contributions_mw * _CARBON_INTENSITY).sum()

        return float(carbon_emissions_rate)


    def assign_renewable_source(
        self, 
        gen_id: int, 
        gen_type: str,
        max_p_mw: float
    ) -> bool:
        """
        Mark a 'gen' element as a renewable source.

        Args:
            gen_id: The id of the generator to mark
            gen_type: The type of renewable source
            max_p_mw: The maximum power generation of the source
                      This is different from the pandapower maximum p_mw
                      which will change depending on weather

        Returns:
            Whether the generator was marked succesfully
        """
        # Only allow generators to be renewable sources
        # No duplicates
        if (gen_id not in self._net.gen.index.values
            or gen_id in self._renewable_sources):
            return False

        # Ensure renewable source is defined correctly
        if gen_type not in _RENEWABLE_SOURCE_TYPES:
            return False

        self._renewable_sources.append(gen_id)
        self._renewable_source_types.append(gen_type)
        self._renewable_source_max_pmw.append(max_p_mw)
        return True

    
    def simplify_grid(self) -> None:
        """
        Modifies the grid.
        
        This function uncaps tranmission line limits, sets costs for electricity generation, and
        and more.
        """

        # self.process_grid.modify_non_datacenter_load(
        #     net=self._net, data_centers=self._data_centers
        # )
        self.process_grid.sync_generator_costs(
            net=self._net, renewable_sources=self._renewable_sources 
        )
        self.process_grid.sync_generator_limits(
            net=self._net, renewable_sources=self._renewable_sources
        )
        self.process_grid.unconstrain_tranmission(
            net=self._net
        )


    def run_flow(self, weather_index: int) -> None:
        """
        Runs a power flow analysis for this grid.
        """
        current_weather = self._weather_data.iloc[weather_index]
        cloudcover = current_weather["cloudcover"]
        windspeed = current_weather["windspeed_10m"]

        for idx, load_id in enumerate(self._renewable_sources):
            gen_type = self._renewable_source_types[idx]
            gen_max_pmw = self._renewable_source_max_pmw[idx]
            actual_max_pmw: float            

            if gen_type == "solar":
                actual_max_pmw = gen_max_pmw * (1 - cloudcover / 100)

            elif gen_type == "wind":
                actual_max_pmw = gen_max_pmw * windspeed / self._max_wind_speed

            self._net.gen.loc[load_id, "max_p_mw"] = actual_max_pmw

        pp.rundcopp(self._net)

    
        

