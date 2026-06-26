import pandapower as pp

from run_bialek import PP_GENERATION_SOURCES


class GridDiagnostic:
    """
    A class that provides diagnostic methods for analyzing the grid configuration
    and simulation results.
    """

    def __init__(self, pp_grid: pp.pandapowerNet):
        """
        Initializes the GridDiagnostic object with a pandapower network.

        Args:
            pp_grid: A pandapower network object to analyze.
        """
        self.grid = pp_grid

        
    def check_generation_vs_load(self) -> bool:
        """
        Checks if the total generation capacity is sufficient to meet the total load demand.

        Returns:
            True if generation capacity is greater than or equal to load demand, False otherwise.
        """
        total_generation_capacity = self.grid.gen["max_p_mw"].sum() + self.grid.sgen["max_p_mw"].sum()
        total_load_demand = self.grid.load["p_mw"].sum()
        
        return total_generation_capacity >= total_load_demand

        
    def check_all_poly_cost_exists(self) -> bool:
        """
        Checks if all generation elements have corresponding polynomial cost entries. 

        Returns:
            True if all generation elements have polynomial cost entries, False otherwise.
        """
        for elem, _ in PP_GENERATION_SOURCES:
            if elem not in self.grid or len(self.grid[elem]) == 0:
                continue
            
            elem_ids = self.grid[elem].index.values
            elem_cost_mask = self.grid.poly_cost["et"] == elem
            elem_cost_ids = self.grid.poly_cost.loc[elem_cost_mask, "element"].values
            if set(elem_ids) != set(elem_cost_ids):
                return False

        return True
        