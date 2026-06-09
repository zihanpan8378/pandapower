import numpy as np

from typing import List
from pandapower import pandapowerNet
from constants import (
    NONRENEWABLE_BASE_COST,
    RENEWABLE_BASE_COST
)
from run_bialek import GENERATION_SOURCES


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
        for elem, _ in GENERATION_SOURCES:

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
            RENEWABLE_BASE_COST,
            NONRENEWABLE_BASE_COST
        )
        net.poly_cost["cp2_eur_per_mw2"] = np.where(
            is_renewable_gen,
            RENEWABLE_BASE_COST,
            NONRENEWABLE_BASE_COST
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