import os
import numpy as np
import pandapower as pp
import math
import random
import pandas as pd

from pandapower import pandapowerNet
from typing import List, Dict
from datetime import datetime

from run_bialek import (
    calculate_source_to_load_contributions, 
    PP_GENERATION_SOURCES
)
from constants import (
    CARBON_INTENSITIES,
    WEATHER_DATA_PATHS,
    ONSITE_GENERATION_COST_PER_MW,
    ONSITE_GENERATION_BASE_COST,
    ONSITE_GENERATION_COST_PER_MW2
)
from process_grid import ProcessGrid
from generation_types import GenerationType, GenerationTypeCost
from grid_regions import GridRegion



class DatacenterGrid:
    """
    Class that represents a electric grid, containing data centers and renewable energy sources.

    Attributes:
        _net: The underlying pandapower network
        _data_centers: Maps pandapower load id to DataCenter instance
        _renewable_sources: List of pandapower gen id that are renewable sources
    """

    def __init__(self, net: pandapowerNet, grid_region: GridRegion) -> None:
        """
        Creates an instance of the DataCenterGrid class.

        Args:
            net: The underlying pandapower network
            grid_region: The region that the grid is in, used for determining
                         energy generation at a given time

        Raises:
            ValueError: If the grid_region is not an allowed region
        """

        self._net: pandapowerNet    = net
        self._region: GridRegion    = grid_region
        self._weather_data          = pd.read_csv(WEATHER_DATA_PATHS[grid_region])
        self._weather_data.set_index("datetime_utc", inplace=True)

        # We take the maximum wind speed to support proportional power scaling
        # based on wind speed at given time
        # We don't need to do this for cloud cover, which is already in interval[0, 100]
        self._max_wind_speed    = self._weather_data["windspeed_10m"].max()
        self.process_grid       = ProcessGrid()

        # We label energy sources and data center loads inside pandapower net
        self._power_limit_label: str    = self.__initialize_energy_max_power()
        self._dc_label: str             = self.__initialize_dc_label()
        self._source_label: str         = self.__initialize_energy_source_labels()
        self._onsite_gen_label: str     = self.__initialize_onsite_gen_label()

        self.__assert_at_most_one_source_per_bus()
        # NOTE: consumption permissions depend on each generator's source type, which
        # is only assigned during grid setup. They are therefore applied in
        # simplify_grid(), once every generator has a source type, rather than here.


    def assign_new_data_center(
        self, 
        load_id: int,
        max_power: float
    ) -> None:
        """
        Marks an existing load as a data center. 

        Args:
            load_id: the id of the load to mark as a data center
            max_power: the maximum power consumption of the data center, used for setting up backup generation capacity
        """

        # Data center must be assigned to an existing load 
        if load_id not in self._net.load.index.values:
            raise ValueError(f"Load id {load_id} does not exist in the grid, cannot assign data center")

        self._net.load.loc[load_id, self._dc_label] = True
        self._net.load.loc[load_id, self._power_limit_label] = max_power

        
    def get_max_power_for_dcs(
        self,
        load_ids: list[int],
    ) -> list[float]:
        """
        Gets the maximum power limits for a list of data center loads.

        Args:
            load_ids: the ids of the loads to get the power limits for

        Returns:
            A list of maximum power limits for the data center loads
        """
        # All load ids must be valid data center loads
        if not all(load_id in self._net.load.index.values 
                   and self._net.load.loc[load_id, self._dc_label] 
                   for load_id in load_ids):
            raise ValueError(f"All load ids must be valid data center loads to get power limits")

        return self._net.load.loc[load_ids, self._power_limit_label].tolist()


    def get_data_center_load_ids(self) -> List[int]:
        """
        Retrieves the load ids of data centers.
        """
        load_indicies = self._net.load.index[self._net.load[self._dc_label] == True].tolist()
        return load_indicies


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
    

    def assign_generation_source_type(
        self, 
        gen_id: int, 
        gen_type: GenerationType,
        max_p_mw: float
    ) -> None:
        """
        Mark a 'gen' element as a renewable source.

        Args:
            gen_id: The id of the generator to mark
            gen_type: The type of renewable source
            max_p_mw: The maximum power generation of the source
                      This is different from the pandapower maximum p_mw
                      which will change depending on weather
        """
        # Casting GenerationType to str gives the label name (i.e. coal, solar)
        self._net.gen.loc[gen_id, self._source_label] = gen_type.value
        self._net.gen.loc[gen_id, self._power_limit_label] = max_p_mw


    def create_generation_source(
        self,
        target_bus_id: int,
        gen_type: GenerationType,
        max_p_mw: float
    ):
        """
        Creates a new generator in the grid of a specified type at a specified bus.

        Args:
            target_bus_id: The id of the bus to connect the generator to
            gen_type: The type of renewable source
            max_p_mw: The maximum power generation of the source
                      This is different from the pandapower maximum p_mw
                      which will change depending on weather
        """
        voltage_level: float = self._net.bus.loc[target_bus_id, "vn_kv"] # type: ignore
        new_bus = pp.create_bus(
            net     = self._net, 
            vn_kv   = voltage_level
        )
        pp.create_line_from_parameters(
            net             = self._net,
            from_bus        = new_bus,
            to_bus          = target_bus_id, # type: ignore
            length_km       = 0.1,
            r_ohm_per_km    = 0.1,
            x_ohm_per_km    = 0.1,
            c_nf_per_km     = 0,
            max_i_ka        = np.inf
        )
        new_gen_id = pp.create_gen(
            net      = self._net,
            bus      = new_bus,
            p_mw     = 0.0,
            max_p_mw = max_p_mw,
            min_p_mw = -np.inf
        )
        self._net.gen.loc[new_gen_id, self._onsite_gen_label] = False
        pp.create_poly_cost(
            net             = self._net,
            element         = new_gen_id,
            et              = "gen",
            cp0_eur         = GenerationTypeCost(gen_type).base_cost,
            cp1_eur_per_mw  = GenerationTypeCost(gen_type).linear_cost,
            cp2_eur_per_mw2 = GenerationTypeCost(gen_type).quadratic_cost
        )
        self.assign_generation_source_type(
            gen_id      = new_gen_id, # type: ignore
            gen_type    = gen_type,
            max_p_mw    = max_p_mw
        )

    
    def assign_grid_generation_percentage(
        self, 
        gen_percentage: float,
        source_type: GenerationType
    ) -> None:
        """
        Assigns a percentage of the generators in the grid to be renewable sources.

        Uses a greedy approximation approach where we assign renewable sources to the largest generators in the grid
        until we reach the desired percentage of renewable generation capacity.

        Args:
            gen_percentage: The percentage of generators to assign as renewable sources
            source_type: The type of renewable source to assign
        """
        # Only consider in-service grid generators that have not yet been assigned a
        # source type. Filtering on the source label (rather than the pandapower
        # "type" column, which is None for every generator) excludes the onsite
        # generator, the backup-coal generators created below, and generators already
        # assigned in a previous pass - so none of them get reassigned or duplicated.
        valid_gens = self._net.gen[
            (self._net.gen["in_service"] == True)
            & (self._net.gen[self._source_label].isna())
            & (self._net.gen[self._onsite_gen_label] != True)
        ].copy()
        total_p_mw = valid_gens["max_p_mw"].sum()

        # The amount of generation we want from the source 'source_type' in this grid
        target_p_mw = total_p_mw * gen_percentage

        # We use max power instead of active power during the grid simulation
        sorted_gens = valid_gens.sort_values(by="max_p_mw", ascending=False) 
        selected_generator_indices = []
        accumulated_renewable_p_mw = 0.0
    
        for idx, row in sorted_gens.iterrows():
            gen_p_mw = row["max_p_mw"]

            # Keep on adding generators as renewable sources until we reach 
            # over the target source generation percentage for the grid
            if accumulated_renewable_p_mw < target_p_mw:
                selected_generator_indices.append(idx) 
                accumulated_renewable_p_mw += gen_p_mw
            else:
                break

                
        for gen_id in selected_generator_indices:
            # We keep the maximum power of the renewable source the same
            current_max_p_mw: float = self._net.gen.loc[gen_id, "max_p_mw"] # type: ignore
            self.assign_generation_source_type(
                gen_id      = gen_id, 
                gen_type    = source_type, 
                max_p_mw    = current_max_p_mw
            )


    def create_backup_gen_for_renewables(self) -> None:
        """
        Creates backup coal generators for every renewable generator in the grid.

        This is to ensure that the grid can still meet load even when renewable generation is low due to weather variation.
        The backup generators are connected to the same bus as the renewable generator, but with a different line, so that they can be distinguished in power flow results and don't interfere with bialek's tracing algorithm.
        The backup generators have the same maximum power as the renewable generator they back up, to ensure they can fully back up the renewable generation if needed.
        """
        renewable_mask = self._net.gen[self._source_label].isin(
            [GenerationType.SOLAR.value, GenerationType.WIND.value]
            )
        offsite_mask = self._net.gen[self._onsite_gen_label] != True
        renewable_gens = self._net.gen.loc[renewable_mask & offsite_mask]
        
        for _, gen_row in renewable_gens.iterrows():
            gen_connected_bus: int = gen_row["bus"]
            max_p_mw: float        = gen_row["max_p_mw"]
            self.create_generation_source(
                target_bus_id   = gen_connected_bus, 
                gen_type        = GenerationType.COAL,
                max_p_mw        = max_p_mw
            )


    def fill_missing_generation_source_types(self, source_type: GenerationType) -> None:
        """
        Fills all generation sources in the grid with the specified source type. 

        This is used to set a default source type for all generators before selectively assigning some as renewable sources.

        Args:
            source_type: The type of generation source to assign to all generators
        """
        for elem, _ in PP_GENERATION_SOURCES:
            missing_source_mask = self._net[elem][self._source_label].isna()
            self._net[elem].loc[missing_source_mask, self._source_label] = source_type.value


    def create_source_next_to_dc(
        self, 
        load_id: int, 
        gen_type: GenerationType,
        max_p_mw: float,
    ) -> None:
        """
        Creates a new renewable source next to a data center load.

        Args:
            load_id:    The id of the data center load
            gen_type:   The type of renewable source
            max_p_mw:   The maximum power generation of the source
        """
        # Find the bus to which the load is connected
        target_bus = self._net.load.loc[load_id, "bus"]

        # Generator must be connected via a different line rather than the same bus 
        # to avoid loss of information when running bialek's tracing algorithm
        voltage_level = self._net.bus.loc[target_bus, "vn_kv"] # type: ignore
        new_bus_id = pp.create_bus(
            net     = self._net, 
            name    = f"bus_for_gen_{load_id}", 
            vn_kv   = voltage_level
        )
        
        # We set aribtray constants for the line since it doesn't affect DC optimal power flow results
        pp.create_line_from_parameters(
            net             = self._net,
            from_bus        = new_bus_id,
            to_bus          = target_bus, # type: ignore
            length_km       = 0.1,
            r_ohm_per_km    = 0.1,
            x_ohm_per_km    = 0.1,
            c_nf_per_km     = 0,
            max_i_ka        = np.inf
        )

        # We overprovision the onsite generator capacity to support more than the dc's idle load
        # This is to ensure the dc starts off using onsite generation
        new_gen_id = pp.create_gen(
            net      = self._net,
            bus      = new_bus_id,
            p_mw     = 0.0,
            max_p_mw = max_p_mw,
            min_p_mw = -np.inf
        )
        self._net.gen.loc[new_gen_id, self._onsite_gen_label] = True

        # Poly cost required to dispatch the generator in optimal power flow
        pp.create_poly_cost(
            net             = self._net,
            element         = new_gen_id,
            et              = "gen",
            cp0_eur         = ONSITE_GENERATION_BASE_COST,
            cp1_eur_per_mw  = ONSITE_GENERATION_COST_PER_MW,
            cp2_eur_per_mw2 = ONSITE_GENERATION_COST_PER_MW2
        )

        # Mark the generator as a renewable source
        self.assign_generation_source_type(
            gen_id = new_gen_id,    # type: ignore
            gen_type = gen_type,
            max_p_mw = max_p_mw
        ) 


    def get_grid_carbon_intensity(self) -> float:
        """
        Calculates the grid's average carbon intensity.

        Returns:
            float: The grid's average carbon intensity.
        """
        total_generation_mw = self.get_total_generation_mw()
        total_carbon_emission_rate = 0.0

        for elem, res in PP_GENERATION_SOURCES:
            p_mw = self._net[res]["p_mw"]

            # We account for each energy source separately since they have different carbon intensities
            for source_type, carbon_intensity in CARBON_INTENSITIES.items():
                source_mask = self._net[elem][self._source_label] == source_type.value
                if source_mask.any():
                    source_p_mw = p_mw[source_mask].clip(lower=0)
                    total_carbon_emission_rate += (source_p_mw * carbon_intensity).sum()

        # Avoid division by zero if the grid is completely dead
        if total_generation_mw == 0:
            return 0.0

        average_intensity_kwh = total_carbon_emission_rate / total_generation_mw
        return float(average_intensity_kwh)
        
        
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

        for elem, _ in PP_GENERATION_SOURCES:
            connected_bus_pp = self._net[elem]["bus"]
            connected_bus_df = connected_bus_pp.map(lambda bus_id: self._net.bus.index.get_loc(bus_id)) 
            carbon_intensities = self._net[elem][self._source_label].map(lambda source_type: CARBON_INTENSITIES.get(GenerationType(source_type), 0.0))
            carbon_contributions = np.clip(E_distribution_mw[connected_bus_df.values, target_bus_df_id], 0, None) * carbon_intensities.values
            carbon_emissions_rate += carbon_contributions.sum()

        assert math.isclose(
            total_power_delivered_mw,
            E_distribution_mw[:, target_bus_df_id].sum(), 
            rel_tol=1e-3
        ), "Total power delivered to load does not match total contributions from sources according to E distribution"

        assert total_power_delivered_mw != 0, "Total power delivered to load is zero"

        # TODO: Refactor
        carbon_intensity = carbon_emissions_rate / total_power_delivered_mw * 1000
        return float(carbon_intensity)   
    

    def get_grid_carbon_emission_rate(self) -> float:
        """
        Calculates the grid's total carbon emissions.

        Returns:
            The grid's carbon total carbon emissions
        """
        grid_carbon_intensity = self.get_grid_carbon_intensity()
        total_generation_mw = self.get_total_generation_mw()
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
        #     net=self._net, dc_label=self._dc_label
        # )
        self.process_grid.sync_generator_costs(
            net                 = self._net, 
            source_label        = self._source_label,
            onsite_gen_label    = self._onsite_gen_label
        )
        # self.process_grid.sync_generator_limits(
        #     net=self._net, source_label=self._source_label
        # )
        self.process_grid.unconstrain_tranmission(
            net=self._net
        )
        # Apply consumption permissions now that every generator has a source type:
        # coal generators may consume power to balance load (the ext_grid is capped at
        # 0), while renewables (incl. the onsite generator) may not.
        self.__set_generation_source_consumption_allowed()


    def run_flow(self, date: datetime, weather_variation: bool) -> None:
        """
        Runs a power flow analysis for this grid.

        Args:
            date: The datetime to run the power flow for, used for determining weather conditions
            weather_variation: Whether to modify renewable energy generation based on weather conditions for this power flow run
        """
        # Cloud cover and windspeed data are used to update renewable energy generation
        current_weather = self._weather_data.loc[date.isoformat() + "Z"]
        cloudcover = current_weather["cloudcover"]
        windspeed = current_weather["windspeed_10m"]
    
        # TODO: Change to use dynamic dispatch for calculating power, based on source
        solar_mask = self._net.gen[self._source_label] == GenerationType.SOLAR.value
        wind_mask = self._net.gen[self._source_label] == GenerationType.WIND.value
        coal_mask = self._net.gen[self._source_label] == GenerationType.COAL.value
        onsite_gen_mask = self._net.gen[self._onsite_gen_label] == True

        # # Only offsite (grid) renewables are weather-dependent. The onsite generator
        # # is a firm, dedicated backup and runs at its full nameplate regardless of
        # # weather. The read and write masks must match, otherwise pandas index
        # # alignment writes NaN into the generators that are read out but not written.
        # solar_offsite_mask = solar_mask & ~onsite_gen_mask
        # wind_offsite_mask = wind_mask & ~onsite_gen_mask

        solar_power_capacity = self._net.gen.loc[solar_mask, self._power_limit_label]
        wind_power_capacity = self._net.gen.loc[wind_mask, self._power_limit_label]

        # Solar power is inversely proportional to cloud cover
        # Wind power is proportional to wind speed
        if weather_variation:
            self._net.gen.loc[solar_mask, "max_p_mw"] = solar_power_capacity * (1 - cloudcover / 100)
            self._net.gen.loc[wind_mask, "max_p_mw"] = wind_power_capacity * windspeed / self._max_wind_speed
        else:
            self._net.gen.loc[solar_mask, "max_p_mw"] = solar_power_capacity
            self._net.gen.loc[wind_mask, "max_p_mw"] = wind_power_capacity

        # Coal and onsite generation always run at their full nameplate capacity.
        self._net.gen.loc[coal_mask, "max_p_mw"] = self._net.gen.loc[coal_mask, self._power_limit_label]
        self._net.gen.loc[onsite_gen_mask, "max_p_mw"] = self._net.gen.loc[onsite_gen_mask, self._power_limit_label]

        self._net.gen["p_mw"] = 0.0

        # pp.rundcopp(self._net, verbose=True)
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


    def get_total_generation_mw(self) -> float:
        """
        Helper function to calculate total generation in the grid.

        Returns:
            Total generation in MW.
        """
        total_generation_mw: float = 0.0
        for _, res in PP_GENERATION_SOURCES:
            p_mw = self._net[res]["p_mw"].clip(lower=0)
            total_generation_mw += p_mw.sum()
        return total_generation_mw

    
    def get_grid_dispatch_price(self) -> float:
        """
        Gets the current grid dispatch price for this grid.

        Returns:
            float: The current grid dispatch price for this grid.
        """
        return self._net.res_cost.sum()
        
   
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
        # Set all energy generation sources to unknown type initially.
        # Assign the column directly to handle empty tables where the row loop would never run.
        for elem, _ in PP_GENERATION_SOURCES:
            self._net[elem][source_label] = None
        return source_label
    
    
    def __initialize_onsite_gen_label(self) -> str:
        """
        Initializes the onsite generator label for each generator.

        Returns:
            str: _description_
        """
        onsite_gen_label = "is_onsite_gen"
        for gen_id in self._net.gen.index.values:
            self._net.gen.loc[gen_id, onsite_gen_label] = False
        return onsite_gen_label

    
    def __initialize_energy_max_power(self) -> str:
        """
        Initializes the maximum power for each generator to be the same as the pandapower max_p_mw.

        This is used for tracking the original maximum power of renewable sources, which will be 
        modified based on weather.
        """
        power_limit_label: str = "power_limit"
        for gen_id in self._net.gen.index.values:
            prev_max_p_mw = self._net.gen.loc[gen_id, "max_p_mw"]
            self._net.gen.loc[gen_id, power_limit_label] = prev_max_p_mw
        return power_limit_label
    

    def __assert_at_most_one_source_per_bus(self) -> None:
        """Raises if any bus has more than one generation source across all element types."""
        bus_counts = {}
        for elem, _ in PP_GENERATION_SOURCES:
            for bus_id in self._net[elem]["bus"].values:
                bus_counts[bus_id] = bus_counts.get(bus_id, 0) + 1

        offending = {bus: n for bus, n in bus_counts.items() if n > 1}
        assert not offending, f"Buses with more than one generation source: {offending}"


    def __set_generation_source_consumption_allowed(self) -> None:
        """
        Sets whether generation sources are allowed to consume power (i.e. have negative p_mw) based on their type.

        Renewable energy sources should not be allowed to consume power, while non-renewable sources should be allowed to consume power to allow for load balancing.
        """
        for elem, _ in PP_GENERATION_SOURCES:
            source_types = self._net[elem][self._source_label]

            for source_type in GenerationType:
                source_mask = (source_types == source_type.value)
                if source_mask.any():
                    if source_type in [GenerationType.SOLAR, GenerationType.WIND]:
                        # Renewable sources should not be allowed to consume power
                        self._net[elem].loc[source_mask, "min_p_mw"] = 0.0
                    else:
                        # Non-renewable sources should be allowed to consume power for load balancing
                        self._net[elem].loc[source_mask, "min_p_mw"] = -np.inf
                        
        # Extended grid and only consume excess power 
        self._net.ext_grid["max_p_mw"] = 0.0