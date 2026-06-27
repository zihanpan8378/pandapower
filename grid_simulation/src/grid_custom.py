import os
import numpy as np
import pandapower as pp
import math
import random
import pandas as pd

from pandapower import pandapowerNet
from datetime import datetime
from collections import defaultdict

from run_bialek import (
    calculate_source_to_load_contributions, 
    PP_GENERATION_SOURCES
)
from constants import (
    WEATHER_DATA_PATHS,
    ONSITE_GENERATION_COST_PER_MW,
    ONSITE_GENERATION_BASE_COST,
    ONSITE_GENERATION_COST_PER_MW2,
    BACKUP_GENERATION_COST_PER_MW,
    BACKUP_GENERATION_BASE_COST,
    BACKUP_GENERATION_COST_PER_MW2,
    CustomGridGenElem
)

from process_grid import ProcessGrid
from generation_types import (
    GenerationType, 
    SOURCE_CARBON_INTENSITIES,
    BASE_GENERATION_COST,
    LINEAR_GENERATION_COST_PER_MWh,
    QUADRATIC_GENERATION_COST_PER_MWh2,
    OIL_DOLLARS_PER_BARREL_BY_DAY,
    PETROLEUM_BARREL_PER_MWH,
)
from grid_regions import GridRegion
from grid_diagnostic import GridDiagnostic



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
        self._diagnostics        = GridDiagnostic(self._net) 

        self.__set_sgen_controllable()
        self.__setup_ext_grid_controllable()

        # Must be the total generation of gen + sgen before additional ones are added
        self._num_generators: float = len(self._net.gen) + len(self._net.sgen)
        self._base_generation_mw: float = self._net.gen["max_p_mw"].sum() + self._net.sgen["max_p_mw"].sum()

        # We label energy sources and data center loads inside pandapower net
        self._power_limit_label: str    = self.__initialize_energy_max_power()
        self._dc_label: str             = self.__initialize_dc_label()
        self._source_label: str         = self.__initialize_energy_source_labels()
        self._onsite_gen_label: str     = self.__initialize_onsite_gen_label()
        self._is_backup_gen_label: str  = self.__initialize_backup_gen_label()
        self._net.ext_grid[self._source_label] = GenerationType.COAL.value

        self.__assert_at_most_one_source_per_bus()
        # NOTE: consumption permissions depend on each generator's source type, which
        # is only assigned during grid setup. They are therefore applied in
        # simplify_grid(), once every generator has a source type, rather than here.


    def assign_new_datacenter(
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

        
    def get_max_power_for_datacenters(
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


    def get_datacenter_load_ids(self) -> list[int]:
        """
        Retrieves the load ids of data centers.
        
        Returns:
            A list of load ids corresponding to data centers in the grid.
        """
        load_indicies = self._net.load.index[self._net.load[self._dc_label] == True].tolist()
        return load_indicies


    def set_datacenter_active_power(
        self, 
        load_ids: list[int], 
        loads: list[float]
    ) -> None:
        """
        Sets the active power of the data center loads.
        """
        assert len(load_ids) == len(loads), "Length of load ids and loads must be the same"

        # Ensure that load ids are correct and correspond to data centr
        assert all(load_id in self._net.load.index.values 
                   and self._net.load.loc[load_id, self._dc_label]
                   for load_id in load_ids), "All load ids must be valid data center loads to set active power"

        for load_id, load in zip(load_ids, loads):
            self._net.load.loc[load_id, "p_mw"] = load


    def get_datacenter_active_power(
        self,
        load_ids: list[int]
    ) -> list[float]:
        """
        Gets the current loads of the data centers.
        
        Args:
            load_ids: The ids of the data center loads to get the active power for
            
        Returns:
            A list of active power values for the specified data center loads
        """
        # We can only get load for existing data centers
        assert all(load_id in self._net.load.index.values 
            and self._net.load.loc[load_id, self._dc_label] 
            for load_id in load_ids)

        return self._net.load.loc[load_ids, "p_mw"].tolist()
    

    def __classify_generation_characteristics(
        self,
        gen_type: CustomGridGenElem,
        id: int, 
        energy_source: GenerationType,
        max_p_mw: float,
        is_onsite: bool,
        is_backup: bool
    ) -> None:
        """
        Mark a custom grid generation element as a renewable source.

        Args:
            type: The type of generation element (GEN or SGEN)
            id: The id of the generation element
            gen_type: The type of generator (GEN or SGEN)
            energy_source: The type of energy source (e.g. solar, wind, coal)
            max_p_mw: The maximum power generation of the source
                      This is different from the pandapower maximum p_mw
                      which will change depending on weather
        """
        # Casting GenerationType to str gives the label name (i.e. coal, solar)
        self._net[gen_type.value].loc[id, self._source_label]           = energy_source.value
        self._net[gen_type.value].loc[id, self._power_limit_label]      = max_p_mw
        self._net[gen_type.value].loc[id, self._onsite_gen_label]       = is_onsite
        self._net[gen_type.value].loc[id, self._is_backup_gen_label]    = is_backup


    def create_generation_source(
        self,
        target_bus_id: int,
        energy_source: GenerationType,
        max_p_mw: float,
        is_onsite: bool = False,
        is_backup: bool = False
    ):
        """
        Creates a new generator in the grid of a specified type at a specified bus.

        Creates a new bus, line, and generator to connect the source to the target bus.

        Args:
            target_bus_id: The id of the bus to connect the generator to
            energy_source: The type of energy source
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
        # The actual cost for this generation element is updated in the 
        pp.create_poly_cost(
            net             = self._net,
            element         = new_gen_id,
            et              = "gen",
            cp0_eur         = 0.0,
            cp1_eur_per_mw  = 0.0,
            cp2_eur_per_mw2 = 0.0
        )
        self.__classify_generation_characteristics(
            gen_type        = CustomGridGenElem.GEN,
            id              = int(new_gen_id),
            energy_source   = energy_source,
            max_p_mw        = max_p_mw,
            is_onsite       = is_onsite,
            is_backup       = is_backup
        )
        self.__update_generation_cost(
            gen_type        = CustomGridGenElem.GEN,
            id              = int(new_gen_id),
            energy_source   = energy_source,
            is_onsite       = is_onsite,
            is_backup       = is_backup
        )

    
    def assign_grid_generation_profile(
        self,
        energy_profile: dict[GenerationType, float]
    ) -> None:
        """
        Assigns source types to the grid's generators to match a target energy profile.

        Rather than filling one source at a time (which makes the largest generators
        overshoot small targets and can starve later, small-share sources), this
        considers every source jointly. It walks the generators largest-first and
        assigns each to whichever source is currently furthest below its target
        capacity (largest absolute deficit). This is the standard greedy for multiway
        number partitioning: large sources outbid small ones for the big generators,
        while small sources keep a positive deficit and pick up the small leftover
        generators - so every source with a positive share gets some capacity as long
        as enough generators exist.

        Generators left unassigned (e.g. when every share is already met) keep their
        None source type and are later defaulted by fill_missing_generation_source_types.

        Args:
            energy_profile: Maps each generation type to its target share of the grid's
                            base generation capacity. Shares must sum to at most 1; any
                            remainder is left for the default-fill pass. A profile that
                            sums to more than 1 is infeasible (the targets exceed total
                            capacity) and is rejected, because the largest sources would
                            otherwise consume every generator and starve the small-share
                            sources entirely.

        Raises:
            ValueError: If the requested shares sum to more than 1.
        """
        # Reject over-subscribed profiles up front. If shares sum to > 1 the targets
        # exceed the grid's capacity, the never-satisfiable large sources monopolise
        # every generator under the most-deficient rule, and small sources silently get
        # nothing. Normalise the profile (divide each share by the total) so it sums to
        # at most 1 before calling this.
        total_share = sum(share for share in energy_profile.values() if share > 0)
        if total_share > 1.0 + 1e-9:
            raise ValueError(
                f"Energy profile shares sum to {total_share:.4f}, which exceeds 1.0 and "
                f"cannot be matched by the grid's generation capacity. Normalise the "
                f"profile so the shares sum to at most 1."
            )

        # Only consider in-service grid generators that have not yet been assigned a
        # source type. Filtering on the source label (rather than the pandapower
        # "type" column, which is None for every generator) excludes the onsite
        # generator, the backup-coal generators created below, and generators already
        # assigned - so none of them get reassigned or duplicated.
        gen_power_map: dict[tuple[CustomGridGenElem, int], float] = {}
        valid_gens = self._net.gen[
            (self._net.gen["in_service"] == True)
            & (self._net.gen[self._source_label].isna())
            & (self._net.gen[self._onsite_gen_label] != True)
            & (self._net.gen[self._is_backup_gen_label] != True)
        ].copy()
        valid_sgens = self._net.sgen[
            (self._net.sgen["in_service"] == True)
            & (self._net.sgen[self._source_label].isna())
        ].copy()

        for gen in valid_gens.index:
            gen_power_map[(CustomGridGenElem.GEN, gen)] = valid_gens.loc[gen, "max_p_mw"] # type: ignore
        for sgen in valid_sgens.index:
            gen_power_map[(CustomGridGenElem.SGEN, sgen)] = valid_sgens.loc[sgen, "max_p_mw"] # type: ignore

        # The target capacity (MW) we want from each source in this grid.
        targets: dict[GenerationType, float] = {
            source: self._base_generation_mw * share
            for source, share in energy_profile.items()
            if share > 0
        }
        accumulated: dict[GenerationType, float] = {source: 0.0 for source in targets}

        # Walk generators largest-first; each is assigned to the most-deficient source.
        sorted_gens: list[tuple[tuple[CustomGridGenElem, int], float]] = sorted(
            gen_power_map.items(), key=lambda x: x[1], reverse=True)

        for gen_info, max_pw in sorted_gens:
            # Only sources still below their target are eligible, so a source that has
            # met its share stops absorbing generators (the remainder falls through to
            # the default-fill pass). Once every source is satisfied, we are done.
            candidates = [s for s in targets if accumulated[s] < targets[s]]
            if not candidates:
                break

            source_type = max(candidates, key=lambda s: targets[s] - accumulated[s])
            accumulated[source_type] += max_pw

            elem_type, id = gen_info
            self.__classify_generation_characteristics(
                id              = id,
                gen_type        = elem_type,
                energy_source   = source_type,
                max_p_mw        = max_pw,
                is_onsite       = False,
                is_backup       = False
            )
            self.__update_generation_cost(
                id              = id,
                gen_type        = elem_type,
                energy_source   = source_type,
                is_onsite       = False,
                is_backup       = False
            )

        # Any remaining unassigned generators are filled with the largest-share source
        self.fill_missing_generation_source_types(
            source_type = max(targets, key=lambda s: targets[s])
        )


    def create_backup_gen_for_renewables(self) -> None:
        """
        Creates backup coal generators for every renewable generator in the grid.

        This is to ensure that the grid can still meet load even when renewable generation is low due to weather variation.
        The backup generators are connected to the same bus as the renewable generator, but with a different line, so that they can be distinguished in power flow results and don't interfere with bialek's tracing algorithm.
        The backup generators have the same maximum power as the renewable generator they back up, to ensure they can fully back up the renewable generation if needed.
        """

        # Loop through defined generation elements to find renewable sources
        for elem in CustomGridGenElem:
            renewable_mask = self._net[elem.value][self._source_label].isin(
                [GenerationType.SOLAR.value, GenerationType.WIND.value]
                )
            offsite_mask = self._net[elem.value][self._onsite_gen_label] != True
            renewable_gens = self._net[elem.value].loc[renewable_mask & offsite_mask]
            
            for _, gen_row in renewable_gens.iterrows():
                gen_connected_bus: int = gen_row["bus"]
                max_p_mw: float        = gen_row["max_p_mw"]
                energy_source_type: GenerationType
                
                if gen_row[self._source_label] == GenerationType.SOLAR.value:
                    energy_source_type = GenerationType.SOLAR_NON_VARYING
                elif gen_row[self._source_label] == GenerationType.WIND.value:
                    energy_source_type = GenerationType.WIND_NON_VARYING
                else:
                    raise ValueError(f"Unexpected energy source type {gen_row[self._source_label]} for renewable generator {gen_row.name}")

                self.create_generation_source(
                    target_bus_id   = gen_connected_bus, 
                    energy_source   = energy_source_type,
                    max_p_mw        = max_p_mw,
                    is_onsite       = False,
                    is_backup       = True
                )


    def fill_missing_generation_source_types(self, source_type: GenerationType) -> None:
        """
        Fills all generation sources in the grid with the specified source type. 

        This is used to set a default source type for all generators before selectively assigning some as renewable sources.

        Args:
            source_type: The type of generation source to assign to all generators
        """
        for elem in CustomGridGenElem:
            missing_source_mask = self._net[elem.value][self._source_label].isna()
            ids = self._net[elem.value].loc[missing_source_mask].index
            for id in ids:
                max_p_mw = self._net[elem.value].loc[id, "max_p_mw"]
                self.__classify_generation_characteristics(
                    id              = id,
                    gen_type        = elem,
                    energy_source   = source_type,
                    max_p_mw        = max_p_mw,
                    is_onsite       = False,
                    is_backup       = False
                )
                # Pre-existing grid generators may not have a poly_cost row, in which
                # case __update_generation_cost would silently no-op. Create one (with
                # placeholder zeros, overwritten below) so the cost actually gets set.
                has_poly_cost = (
                    (self._net.poly_cost["et"] == elem.value)
                    & (self._net.poly_cost["element"] == id)
                ).any()
                if not has_poly_cost:
                    pp.create_poly_cost(
                        net             = self._net,
                        element         = id,
                        et              = elem.value,
                        cp0_eur         = 0.0,
                        cp1_eur_per_mw  = 0.0,
                        cp2_eur_per_mw2 = 0.0
                    )
                self.__update_generation_cost(
                    id              = id,
                    gen_type        = elem,
                    energy_source   = source_type,
                    is_onsite       = False,
                    is_backup       = False
                )



    def create_source_next_to_dc(
        self, 
        load_id: int, 
        energy_source: GenerationType,
        max_p_mw: float,
    ) -> None:
        """
        Creates a new renewable source next to a data center load.

        Args:
            load_id:    The id of the data center load
            energy_source:   The type of renewable source
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
        self.__classify_generation_characteristics(
            id              = int(new_gen_id),
            gen_type        = CustomGridGenElem.GEN,
            energy_source   = energy_source,
            max_p_mw        = max_p_mw,
            is_onsite       = True,
            is_backup       = False
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
            for source_type, carbon_intensity in SOURCE_CARBON_INTENSITIES.items():
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
        carbon_emissions_rate_mw = 0.0

        E_distribution_mw, _, _ = calculate_source_to_load_contributions(self._net)

        for elem, _ in PP_GENERATION_SOURCES:
            connected_bus_pp = self._net[elem]["bus"]
            connected_bus_df = connected_bus_pp.map(lambda bus_id: self._net.bus.index.get_loc(bus_id)) 
            carbon_intensities = self._net[elem][self._source_label].map(
                lambda source_type: SOURCE_CARBON_INTENSITIES.get(GenerationType(source_type), 0.0)
                )
            carbon_contributions = np.clip(E_distribution_mw[connected_bus_df.values, target_bus_df_id], 0, None) * carbon_intensities.values
            carbon_emissions_rate_mw += carbon_contributions.sum()

        assert math.isclose(
            total_power_delivered_mw,
            E_distribution_mw[:, target_bus_df_id].sum(), 
            rel_tol=1e-3
        ), "Total power delivered to load does not match total contributions from sources according to E distribution"

        assert total_power_delivered_mw != 0, "Total power delivered to load is zero"

        carbon_intensity = carbon_emissions_rate_mw / total_power_delivered_mw
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
        # self.process_grid.sync_generator_costs(
        #     net                 = self._net, 
        #     source_label        = self._source_label,
        #     onsite_gen_label    = self._onsite_gen_label
        # )
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
        cloudcover: float = current_weather["cloudcover"] # type: ignore
        windspeed: float = current_weather["windspeed_10m"] # type: ignore

        for gen_type in GenerationType:

            power_ratio: float

            # If weather variation is disabled, all generation sources stay constant at their maximum power
            if not weather_variation:
                power_ratio = 1.0
            
            # Dispatch list to calculate the power generation depending on source
            elif gen_type == GenerationType.SOLAR:
                power_ratio = 1.0 - cloudcover / 100

            elif gen_type == GenerationType.WIND:
                power_ratio = windspeed / self._max_wind_speed                

            else: 
                power_ratio = 1.0

            for elem in CustomGridGenElem:
                source_mask = self._net[elem.value][self._source_label] == gen_type.value
                gen_capacity = self._net[elem.value].loc[source_mask, self._power_limit_label]
                self._net[elem.value].loc[source_mask, "max_p_mw"] = gen_capacity * power_ratio

        # Active power does not need to be set since it is calculated using optimal power flow
        for elem in CustomGridGenElem:
            self._net[elem.value]["p_mw"] = 0.0

        self.__sync_generator_costs(date = date)
        assert self._diagnostics.check_generation_vs_load(), "Generation does not meet demand"
        assert self._diagnostics.check_all_poly_cost_exists(), "Not all generation elements have polynomial cost entries"

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


    def get_max_generation_mw(self) -> float:
        """
        Helper function to calculate total maximum generation in the grid.

        Returns:
            Total maximum generation in MW.
        """
        total_max_generation_mw: float = 0.0
        for elem in CustomGridGenElem:
            max_p_mw = self._net[elem.value]["max_p_mw"].clip(lower=0)
            total_max_generation_mw += max_p_mw.sum()
        return total_max_generation_mw

    
    def get_grid_dispatch_price(self) -> float:
        """
        Gets the current grid dispatch price for this grid.

        Returns:
            float: The current grid dispatch price for this grid.
        """
        return self._net.res_cost.sum()

        
    def get_energy_profile(self) -> dict[GenerationType, float]:
        """
        Gets the current energy profile for this grid.

        Returns:
            A dictionary mapping each generation type to its current energy contribution share.
        """
        max_generation_mw = self.get_max_generation_mw()
        if max_generation_mw == 0:
            return {gen_type: 0.0 for gen_type in GenerationType}

        energy_profile: dict[GenerationType, float] = defaultdict(float)
        for elem in CustomGridGenElem:
            for gen_type in GenerationType:
                source_mask = self._net[elem.value][self._source_label] == gen_type.value
                if source_mask.any():
                    energy_profile[gen_type] += (
                        self._net[elem.value].loc[source_mask, "p_mw"].clip(lower=0).sum() 
                        / max_generation_mw
                    )

        return energy_profile

    
    def get_res_energy_profile(self) -> dict[GenerationType, float]:
        """
        Gets the energy profile of this grid after running a power flow analysis.

        Returns:
            A dictionary mapping each generation type to its current energy contribution share.
        """
        total_generation_mw = self.get_total_generation_mw()
        if total_generation_mw == 0:
            return {gen_type: 0.0 for gen_type in GenerationType}
        
        if not all(f"res_{elem.value}" in self._net for elem in CustomGridGenElem):
            raise ValueError(
                f"Power flow results not available. "
                f"Please run a power flow analysis before calling get_res_energy_profile()."
                )

        energy_profile: dict[GenerationType, float] = defaultdict(float)
        for elem in CustomGridGenElem:
            for gen_type in GenerationType:
                source_mask = self._net[elem.value][self._source_label] == gen_type.value
                if source_mask.any():
                    energy_profile[gen_type] += (
                        self._net[elem.value].loc[source_mask, "p_mw"].clip(lower=0).sum() 
                        / total_generation_mw
                )
        
        return energy_profile
        
   
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
        for elem in CustomGridGenElem:
            self._net[elem.value][source_label] = None
        return source_label
    
    
    def __initialize_onsite_gen_label(self) -> str:
        """
        Initializes the onsite generator label for each generator.

        Returns:
            str: The label name for onsite generators.
        """
        onsite_gen_label = "is_onsite_gen"
        for elem in CustomGridGenElem:
            self._net[elem.value][onsite_gen_label] = False
        return onsite_gen_label
    
    
    def __initialize_backup_gen_label(self) -> str:
        """
        Initializes the backup generator label for each generator.

        Returns:
            str: The label name for backup generators.
        """
        backup_gen_label = "is_backup_gen"
        for elem in CustomGridGenElem:
            self._net[elem.value][backup_gen_label] = False
        return backup_gen_label

    
    def __initialize_energy_max_power(self) -> str:
        """
        Initializes the maximum power limit for each generator.
        
        Returns:
            The label name for the maximum power limit of generators.
        """
        power_limit_label: str = "power_limit"
        for elem in CustomGridGenElem:
            net_elem = self._net[elem.value]
            # sgens are constant-power injections and have no max_p_mw column;
            # seed it from their setpoint p_mw so they can act as capped sources.
            if "max_p_mw" not in net_elem.columns:
                net_elem["max_p_mw"] = net_elem["p_mw"]
            net_elem[power_limit_label] = net_elem["max_p_mw"]
        return power_limit_label
        

    def __assert_at_most_one_source_per_bus(self) -> None:
        """
        Raises if any bus has more than one generation source across all element types.
        """
        bus_counts = defaultdict(int)
        for elem in CustomGridGenElem:
            for bus_id in self._net[elem.value]["bus"].values:
                bus_counts[bus_id] += 1

        offending = {bus: n for bus, n in bus_counts.items() if n > 1}
        assert not offending, f"Buses with more than one generation source: {offending}"


    def __set_generation_source_consumption_allowed(self) -> None:
        """
        Sets whether generation sources are allowed to consume power (i.e. have negative p_mw) based on their type.

        Renewable energy sources should not be allowed to consume power, while non-renewable sources should be allowed to consume power to allow for load balancing.
        """
        # for elem in CustomGridGenElem:
        #     source_types = self._net[elem.value][self._source_label]

        #     for source_type in GenerationType:
        #         source_mask = (source_types == source_type.value)
        #         if source_mask.any():
        #             if source_type in [GenerationType.SOLAR, GenerationType.WIND]:
        #                 # Renewable sources should not be allowed to consume power
        #                 self._net[elem.value].loc[source_mask, "min_p_mw"] = 0.0
        #             else:
        #                 # Non-renewable sources should be allowed to consume power for load balancing
        #                 self._net[elem.value].loc[source_mask, "min_p_mw"] = -np.inf
            
        # # Backup generators are not allowed to consume power
        # backup_generators_mask = self._net.gen[self._is_backup_gen_label] == True
        # if backup_generators_mask.any():
        #     self._net.gen.loc[backup_generators_mask, "min_p_mw"] = 0.0
            
        # # Onsite generators are not allowed to consume power
        # onsite_generators_mask = self._net.gen[self._onsite_gen_label] == True
        # if onsite_generators_mask.any():
        #     self._net.gen.loc[onsite_generators_mask, "min_p_mw"] = 0.0
                        
        # Extended grid can inject or consume excess power for slack/load balancing
        # self._net.ext_grid["max_p_mw"] = 0.0
        # self._net.ext_grid["min_p_mw"] = -np.inf
        # Generators must have a finite lower bound. A -inf minimum lets every unit
        # absorb unlimited power (act as an infinite load), which makes the DC-OPF
        # feasible region unbounded and produces enormous circulating flows; once the
        # real branch thermal limits are applied the problem becomes ill-posed and
        # rundcopp reports OPFNotConverged. Floor dispatch at 0 (units can ramp down
        # to zero output but not consume power).
        self._net.gen["min_p_mw"] = 0.0

    def __sync_generator_costs(self, date: datetime) -> None:
        """
        Synchronizes the generator costs in the pandapower network with the global cost constants.

        This function updates the cost parameters for each generator in the network based on its source type.
        """
        for elem in CustomGridGenElem:
            et_rows = self._net.poly_cost[self._net.poly_cost["et"] == elem.value]
            src = self._net[elem.value][self._source_label]
            
            ids = src.index[src == GenerationType.CRUDE_OIL.value].tolist()
            mask = et_rows.index[et_rows["element"].isin(ids)]
            if len(mask) == 0:
                continue

            # Oil generation costs are handled separately due to their dependence on the date
            oil_cost_date     = date.strftime("%m/%d/%Y")
            oil_cost_per_mwh  = OIL_DOLLARS_PER_BARREL_BY_DAY.loc[oil_cost_date, "Price"] * PETROLEUM_BARREL_PER_MWH # type: ignore
            self._net.poly_cost.loc[mask, "cp0_eur"] = BASE_GENERATION_COST[GenerationType.CRUDE_OIL]
            self._net.poly_cost.loc[mask, "cp1_eur_per_mw"] = oil_cost_per_mwh
            self._net.poly_cost.loc[mask, "cp2_eur_per_mw2"] = QUADRATIC_GENERATION_COST_PER_MWh2[GenerationType.CRUDE_OIL]

       
    def __set_sgen_controllable(self) -> None:
        """
        Sets all sgens in the grid to be controllable.

        This is necessary for the optimal power flow to work correctly, since sgens are constant-power injections and cannot be controlled by default.
        """
        if self._net.sgen.empty:
            return
        self._net.sgen["controllable"] = True
        # Initialize the min and max power limits so they can be controlled
        # by optimal power flow. min_p_mw must be finite (see
        # __set_generation_source_consumption_allowed): a -inf lower bound makes the
        # DC-OPF feasible region unbounded and causes OPFNotConverged once branch
        # limits bind. Floor dispatch at 0.
        self._net.sgen["min_p_mw"] = 0.0
        self._net.sgen["max_p_mw"] = self._net.sgen["p_mw"]
        self._net.sgen["p_mw"] = 0.0
        ids = list(self._net.sgen.index.values)
        pp.create_poly_costs(
            net             = self._net,
            elements        = ids,
            et              = "sgen",
            cp0_eur         = 0.0,
            cp1_eur_per_mw  = 0.0,
            cp2_eur_per_mw2 = 0.0
        )

    def __setup_ext_grid_controllable(self) -> None:
        """
        Sets the external grid to be controllable.
        """
        self._net.ext_grid["controllable"] = True
        self._net.ext_grid["min_p_mw"] = -np.inf
        self._net.ext_grid["max_p_mw"] = 0.0
        self._net.ext_grid["p_mw"] = 0.0
        mask = self._net.poly_cost["et"] == "ext_grid"
        self._net.poly_cost.loc[mask, ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]] = 0.0

    
    def __update_generation_cost(
        self,
        gen_type: CustomGridGenElem,
        id: int,
        energy_source: GenerationType,
        is_onsite: bool,
        is_backup: bool
        ) -> None:
        """
        Updates the generation cost for a specific generator based on its energy source type.
        
        Args:
            gen_type: The type of generation element (GEN or SGEN)
            id: The id of the generation element
            energy_source: The type of energy source (e.g. solar, wind, coal)
            is_onsite: Whether the generator is onsite
            is_backup: Whether the generator is a backup source
        """
        elem_mask = (
            (self._net.poly_cost["et"] == gen_type.value)
            & (self._net.poly_cost["element"] == id)
        )
        if is_onsite:
            self._net.poly_cost.loc[elem_mask, "cp0_eur"] = ONSITE_GENERATION_BASE_COST
            self._net.poly_cost.loc[elem_mask, "cp1_eur_per_mw"] = ONSITE_GENERATION_COST_PER_MW
            self._net.poly_cost.loc[elem_mask, "cp2_eur_per_mw2"] = ONSITE_GENERATION_COST_PER_MW2
        elif is_backup:
            self._net.poly_cost.loc[elem_mask, "cp0_eur"] = BACKUP_GENERATION_BASE_COST
            self._net.poly_cost.loc[elem_mask, "cp1_eur_per_mw"] = BACKUP_GENERATION_COST_PER_MW
            self._net.poly_cost.loc[elem_mask, "cp2_eur_per_mw2"] = BACKUP_GENERATION_COST_PER_MW2
        else:
            self._net.poly_cost.loc[elem_mask, "cp0_eur"] = BASE_GENERATION_COST[energy_source]
            self._net.poly_cost.loc[elem_mask, "cp1_eur_per_mw"] = LINEAR_GENERATION_COST_PER_MWh[energy_source]
            self._net.poly_cost.loc[elem_mask, "cp2_eur_per_mw2"] = QUADRATIC_GENERATION_COST_PER_MWh2[energy_source]