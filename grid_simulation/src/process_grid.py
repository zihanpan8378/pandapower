import numpy as np

from typing import List
from pandapower import pandapowerNet
from run_bialek import PP_GENERATION_SOURCES
from generation_types import GenerationType 
from constants import ONSITE_GENERATION_BASE_COST, ONSITE_GENERATION_COST_PER_MW, ONSITE_GENERATION_COST_PER_MW2


class ProcessGrid:
    """
    Class that contains helper methods to modify the grid for simulation.
    """

    # TODO: Conflicts with modifying data load inside the run_flow function
    # Coal sources are constant and have no variation, this should be set in initialization
    
    # def sync_generator_limits(self, net: pandapowerNet, source_label: str) -> None:
    #     """
    #     Updates the generation limits for non-renewable grid generation components.

    #     Args:
    #         net: The pandapower network to modify
    #         source_label: The label for the column containing the source types
    #     """
    #     gross_load = net.load["p_mw"].sum()        

    #     for elem, _ in GENERATION_SOURCES:
    #         if elem not in net or len(net[elem]) == 0:
    #             continue
    #         source_types = net[elem][source_label]
    #         for source_type, limit_factor in SOURCE_LIMITS.items():
    #             source_mask = (source_types == source_type)
    #             if source_mask.any():
    #                 net[elem].loc[source_mask, "max_p_mw"] = limit_factor * gross_load
    #                 net[elem].loc[source_mask, "min_p_mw"] = 0.0 


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
        non_dc_mask = ~net.load[dc_label].astype(bool)
        net.load.loc[non_dc_mask, "p_mw"] = 0.0


    def unconstrain_tranmission(self, net: pandapowerNet) -> None:
        """
        Helper function to update the transmission components of the grid 
        to have (virtually) limitless capacity.

        Args:
            net: The pandapower network to update
        """
        net.line['max_i_ka'] = 99999.0
        net.line["max_loading_percent"] = 99999.0
        
        net.bus['max_vm_pu'] = 99999.0
        net.bus['min_vm_pu'] = 0.0

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