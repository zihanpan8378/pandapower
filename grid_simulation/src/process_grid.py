import numpy as np

from typing import List
from pandapower import pandapowerNet
from constants import (
    SOURCE_LIMITS,
    SOURCE_COSTS,
)
from run_bialek import GENERATION_SOURCES


class ProcessGrid:
    """
    Class that contains helper methods to modify the grid for simulation.
    """

    def sync_generator_limits(self, net: pandapowerNet, source_label: str) -> None:
        """
        Updates the generation limits for non-renewable grid generation components.

        Args:
            net: The pandapower network to modify
            source_label: The label for the column containing the source types
        """
        gross_load = net.load["p_mw"].sum()        

        for elem, _ in GENERATION_SOURCES:
            if elem not in net or len(net[elem]) == 0:
                continue
            source_types = net[elem][source_label]
            for source_type, limit_factor in SOURCE_LIMITS.items():
                source_mask = source_types == source_type
                if source_mask.any():
                    net[elem].loc[source_mask, "max_p_mw"] = limit_factor * gross_load


    def sync_generator_costs(self, net: pandapowerNet, source_label: str) -> None:
        """
        Updates costs for all elements in poly_cost. 
        
        'Gen' elements marked as renewable will be set to have a cheap electricity prices.
        Other elements will have expensive electricity prices. This is so pandapower dc optimal power 
        flow will dispatch the renewable energy sources before the non-renewable ones.

        Args:
            net: The pandapower network to modify 
            source_label: The label for the column containing the source types
        """
        for elem, _ in GENERATION_SOURCES:
            if elem not in net or len(net[elem]) == 0:
                continue
            source_types = net[elem][source_label]
            for source_type, cost in SOURCE_COSTS.items():
                source_mask = source_types == source_type
                matched_indices = net[elem].index[source_mask]
                if source_mask.any():
                    # We update the cost for all elements of this type 
                    # with the same generation source type being considered
                    net.poly_cost.loc[
                        net.poly_cost["element"].isin(matched_indices) 
                        & (net.poly_cost["et"] == elem), "cp1_eur_per_mw"
                    ] = cost
  

    def modify_non_datacenter_load(
        self, 
        net: pandapowerNet, 
        dc_label: str
    ) -> None:
        """
        Updates the active power of non data center loads to be 0.
        
        Args:
            net: The pandapower network to modify
            data_centers: The list of load ids that are data centers
        """
        non_dc_mask = ~net.load[dc_label]
        net.load.loc[non_dc_mask, "p_mw"] = 0.0


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