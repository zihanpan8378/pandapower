import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from simulation_interface import GridSimulation
from typing import List
from pandapower import pandapowerNet
from sim_config import GridConfig, DataCenterConfig
from grid_regions import GridRegion



# The number of steps to run the simulation for
_MAX_TIME_STEP: int = 200

# data_path = "/mnt/grid-cloud-migration-estimates_copy/simulator/src/case_1/output/datacenter_result_no_shifting_S41.csv"
# NO_SHIFTING_POWER_DATA = pd.read_csv(data_path)

# data_on = NO_SHIFTING_POWER_DATA[NO_SHIFTING_POWER_DATA["dc_name"] == "CA_ON"]
# data_ciso = NO_SHIFTING_POWER_DATA[NO_SHIFTING_POWER_DATA["dc_name"] == "US_CAL_CISO"]
# data_pjm = NO_SHIFTING_POWER_DATA[NO_SHIFTING_POWER_DATA["dc_name"] == "US_MIDA_PJM"]
# data_erco = NO_SHIFTING_POWER_DATA[NO_SHIFTING_POWER_DATA["dc_name"] == "US_TEX_ERCO"]

sim_config_path = "/mnt/pandapower/grid_simulation/src/sim_config.yaml"

def run_simulation():
    """
    Main function to run the grid simulation. Initializes the simulation interface, runs the simulation, and plots the results.
    """
    # Define the grid configuration and data center configuration for the simulation
    grid_config = GridConfig(
            pp_grid=pn.case300(), # type: ignore
            renewable_share=0.3, 
            region=GridRegion.CA_ON
        )
    data_center_config = DataCenterConfig(
        load_share=0.5, 
        onsite_overprovision_factor=1.1
    )
    load_id = grid_config.add_data_center(data_center_config)

    # Initialize the grid simulation interface with the defined configurations
    sim_interface = GridSimulation(
        grid_configs=[grid_config],
        shifting_threshold=10.0,
        enable_weather_variation=False
    )
    
        
    # net = sim_interface._grids[0]._net
    # net.gen.to_csv("gen_output.csv") 
    # net.poly_cost.to_csv("poly_cost_output.csv")
    # net.res_gen.to_csv("res_gen_output.csv")
    # net.load.to_csv("load_output.csv")
    # net.res_ext_grid.to_csv("ext_grid_output.csv")

    print(f"[done] Simulation complete. Generating plots...")
    # sim_interface.generate_plots()
    print(f"[done] Plots saved.")
    
    
if __name__ == "__main__":
    run_simulation() 