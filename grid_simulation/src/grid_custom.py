import os
import numpy as np
import pandapower as pp
import math
import pandas as pd

from pandapower import pandapowerNet
from typing import List, Dict

from run_bialek import (
    calculate_source_to_load_contributions, 
    gross_gen_demand,
    GENERATION_SOURCES
)
from constants import (
    RENEWABLE_SOURCE_TYPES, 
    CARBON_INTENSITIES,
    ALLOWED_GRID_REGIONS,
    WEATHER_DATA_PATHS
)
from process_grid import ProcessGrid

_DEFAULT_ENERGY_SOURCE: str = "coal"


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

        if grid_region not in ALLOWED_GRID_REGIONS:
            raise ValueError(
                f"Invalid Grid Region: {grid_region} is not in {ALLOWED_GRID_REGIONS}"
            )
        
        self._net = net
        self._weather_data = pd.read_csv(WEATHER_DATA_PATHS[grid_region])

        # We take the maximum wind speed to support proportional power scaling
        # based on wind speed at given time
        # We don't need to do this for cloud cover, which is already in interval[0, 100]
        self._max_wind_speed = self._weather_data.loc[:, "windspeed_10m"].max()
        self.process_grid = ProcessGrid()

        # We label energy sources and data center loads inside pandapower net
        self._dc_label: str = self.__initialize_dc_label()
        self._source_label: str = self.__initialize_energy_source_labels()
        self._power_limit_label: str = self.__initialize_energy_max_power()

        self.__assert_at_most_one_source_per_bus() 
        

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

        # Data center must be assigned to an existing load 
        if load_id not in self._net.load.index.values:
            return False

        self._net.load.loc[load_id, self._dc_label] = True
        self._net.load.loc[load_id, 'p_mw'] = idle_power
        return True


    def get_data_center_load_ids(self) -> List[int]:
        """
        Retrieves the load ids of data centers.
        """
        load_idices = self._net.load.index[self._net.load[self._dc_label] == True].tolist()
        return load_idices


    def set_data_center_active_power(
        self, 
        load_ids: List[int], 
        loads: List[float]
    ) -> bool:
        """
        Sets the active power of the data center loads.
        """
        # We can only set load for existing data centers
        if not all(load_id in self._net.load.index.values 
                   and self._net.load.loc[load_id, self._dc_label] 
                   for load_id in load_ids):
            return False

        # Ensure data consistency
        if len(load_ids) != len(loads):
            return False

        for load_id, load in zip(load_ids, loads):
            self._net.load.loc[load_id, "p_mw"] = load

        return True


    def get_data_center_active_power(
        self,
        load_ids: List[int]
    ) -> List[float]:
        """
        Gets the current loads of the data centers.
        """
        # We can only get load for existing data centers
        assert all(load_id in self._net.load.index.values 
            and self._net.load.loc[load_id, self._dc_label] 
            for load_id in load_ids)

        return self._net.load.loc[load_ids, "p_mw"].tolist()
    

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
        if (gen_id not in self._net.gen.index.values
            or gen_type not in RENEWABLE_SOURCE_TYPES):
            return False

        self._net.gen.loc[gen_id, self._source_label] = gen_type
        self._net.gen.loc[gen_id, self._power_limit_label] = max_p_mw
        return True

    
    def get_grid_carbon_intensity(self) -> float:
        """
        Calculates the grid's average carbon intensity.

        Returns:
            float: The grid's average carbon intensity.
        """
        total_generation_mw = self.__get_total_generation_mw()
        total_carbon_emission_rate = 0.0

        for elem, res in GENERATION_SOURCES:
            p_mw = self._net[res]["p_mw"]

            # We account for each energy source separately since they have different carbon intensities
            for source_type, carbon_intensity in CARBON_INTENSITIES.items():
                source_mask = self._net[elem][self._source_label] == source_type
                if source_mask.any():
                    source_p_mw = p_mw[source_mask].clip(lower=0)
                    total_carbon_emission_rate += (source_p_mw * carbon_intensity).sum()

        # Avoid division by zero if the grid is completely dead
        if total_generation_mw == 0:
            return 0.0

        average_intensity = total_carbon_emission_rate / total_generation_mw
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
        # We need the dataframe index of the load's connected bus to index into the E distribution matrix
        target_bus_pp_id = self._net.load.loc[load_index, "bus"]
        target_bus_df_id = self._net.bus.index.get_loc(target_bus_pp_id)

        # Total power delivered to this load (denominator for carbon intensity calculation)
        total_power_delivered_mw = self._net["res_load"].loc[load_index, "p_mw"]
        carbon_emissions_rate = 0.0

        E_distribution_mw, _, _ = calculate_source_to_load_contributions(self._net)

        for elem, _ in GENERATION_SOURCES:
            connected_bus_pp = self._net[elem]["bus"]
            connected_bus_df = connected_bus_pp.map(lambda bus_id: self._net.bus.index.get_loc(bus_id)) 
            carbon_intensities = self._net[elem][self._source_label].map(lambda source_type: CARBON_INTENSITIES.get(source_type, 0.0))
            carbon_contributions = E_distribution_mw[connected_bus_df.values, target_bus_df_id].clip(lower=0) * carbon_intensities.values
            carbon_emissions_rate += carbon_contributions.sum()

        assert math.isclose(
            total_power_delivered_mw,
            E_distribution_mw[:, target_bus_df_id].sum(), 
            rel_tol=1e-3
        ), "Total power delivered to load does not match total contributions from sources according to E distribution"

        # Avoid division by zero if the load is not consuming any power
        if total_power_delivered_mw == 0:
            return 0.0

        carbon_intensity = carbon_emissions_rate / total_power_delivered_mw
        return float(carbon_intensity)   
    

    def get_grid_carbon_emission_rate(self) -> float:
        """
        Calculates the grid's total carbon emissions.

        Returns:
            The grid's carbon total carbon emissions
        """
        grid_carbon_intensity = self.get_grid_carbon_intensity()
        total_generation_mw = self.__get_total_generation_mw()
        total_carbon_emissions_rate = grid_carbon_intensity * total_generation_mw
        return float(total_carbon_emissions_rate)


    def get_data_center_carbon_emissions(self, load_index: int) -> float:
        """
        Calculates the total carbon emissions
        allocated to a specific data center load via Bialek's tracing.
        
        Args:
            load_index: The index of the data center load in net.load.

        Returns:
            The carbon emissions attributable to this load.
        """
        load_carbon_intensity = self.get_data_center_carbon_intensity(load_index)
        load_power_mw = self._net["res_load"].loc[load_index, "p_mw"]
        total_carbon_emissions = load_carbon_intensity * load_power_mw
        return float(total_carbon_emissions)

    
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
            net=self._net, source_label=self._source_label
        )
        self.process_grid.sync_generator_limits(
            net=self._net, source_label=self._source_label
        )
        self.process_grid.unconstrain_tranmission(
            net=self._net
        )


    def run_flow(self, weather_index: int) -> None:
        """
        Runs a power flow analysis for this grid.
        """
        # Cloud cover and windspeed data are used to update renewable energy generation
        current_weather = self._weather_data.iloc[weather_index]
        cloudcover = current_weather["cloudcover"]
        windspeed = current_weather["windspeed_10m"]
    
        solar_mask = self._net.gen[self._source_label] == "solar"
        wind_mask = self._net.gen[self._source_label] == "wind"
        solar_power_capacity = self._net.gen.loc[solar_mask, self._power_limit_label]
        wind_power_capacity = self._net.gen.loc[wind_mask, self._power_limit_label]

        # Solar power is inversely proportional to cloud cover
        # Wind power is proportional to wind speed
        self._net.gen.loc[solar_mask, "max_p_mw"] = solar_power_capacity * (1 - cloudcover / 100)
        self._net.gen.loc[wind_mask, "max_p_mw"] = wind_power_capacity * windspeed / self._max_wind_speed

        pp.rundcopp(self._net)


    def get_gross_load_demand(self) -> float:
        """
        Calculates the gross demand of this grid. 

        Returns:
            The gross power demand.
        """
        if len(self._net["load"]) == 0:
            return 0.0
        # Gross demand consists of the demand of all loads
        return self._net["load"]["p_mw"].sum()
    
        
    def __initialize_dc_label(self) -> str:
        """
        Initializes the data center label for each load.

        Returns:
            The label name for data center loads.
        """
        dc_label = "is_data_center"
        for load_id in self._net.load.index.values:
            self._net.load.loc[load_id, dc_label] = False
        return dc_label


    def __initialize_energy_source_labels(self) -> str:
        """
        Initializes the energy source label for each generator.

        Returns:
            The label name for energy source generators.
        """
        source_label = "source_type"
        # Set all energy generation sources to default type (e.g., coal) initially.
        # Assign the column directly to handle empty tables where the row loop would never run.
        for elem, _ in GENERATION_SOURCES:
            self._net[elem][source_label] = _DEFAULT_ENERGY_SOURCE
        return source_label

    
    def __initialize_energy_max_power(self) -> str:
        """
        Initializes the maximum power for each generator to be the same as the pandapower max_p_mw.

        This is used for tracking the original maximum power of renewable sources, which will be 
        modified based on weather.
        """
        power_limit_label: str = "power_limit"
        for gen_id in self._net.gen.index.values:
            self._net.gen.loc[gen_id, power_limit_label] = self._net.gen.loc[gen_id, "max_p_mw"]
        return power_limit_label

    
    def __get_total_generation_mw(self) -> float:
        """
        Helper function to calculate total generation in the grid.

        Returns:
            Total generation in MW.
        """
        total_generation_mw: float = 0.0
        for _, res in GENERATION_SOURCES:
            p_mw = self._net[res]["p_mw"]
            total_generation_mw += p_mw.sum()
        return total_generation_mw
    

    def __assert_at_most_one_source_per_bus(self) -> None:
        """Raises if any bus has more than one generation source across all element types."""
        bus_counts = {}
        for elem, _ in GENERATION_SOURCES:
            for bus_id in self._net[elem]["bus"].values:
                bus_counts[bus_id] = bus_counts.get(bus_id, 0) + 1

        offending = {bus: n for bus, n in bus_counts.items() if n > 1}
        assert not offending, f"Buses with more than one generation source: {offending}"