import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from simulation_interface import GridSimulation
from typing import List
from pandapower import pandapowerNet



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
    print(f"[init] Loading simulation config from {sim_config_path}")
    num_grids = 2
    raw_grids: List[pandapowerNet] = [pn.case300() for _ in range(num_grids)] # type: ignore
    print(f"[init] Initializing {num_grids} grid(s)...")
    sim_interface = GridSimulation(sim_config=sim_config_path, raw_grids=raw_grids)
    print(f"[init] Grids initialized. Starting simulation for {_MAX_TIME_STEP} time steps.")

    for time_step in range(_MAX_TIME_STEP):
        if time_step % 10 == 0:
            print(f"[step] Time step {time_step}/{_MAX_TIME_STEP}")

        # For simplicity, we use the same weather index for all grids at each time step.
        # In a more complex simulation, each grid could have its own weather profile.
        nets = [grid._net for grid in sim_interface._grids]
        total_generations = [net.res_gen["p_mw"].sum() for net in nets]
        total_loads = [net.load["p_mw"].sum() for net in nets]
        print(f"  Total generation: {total_generations}, Total load: {total_loads}")
        sim_interface.step(shifting_enabled=True, weather_index=time_step)
        
    # sim_interface.step(shifting_enabled=True, weather_index=0)
    # net = sim_interface._grids[0]._net
    # net.gen.to_csv("gen_output.csv") 
    # net.poly_cost.to_csv("poly_cost_output.csv")
    # net.res_gen.to_csv("res_gen_output.csv")
    # net.load.to_csv("load_output.csv")
    # net.res_ext_grid.to_csv("ext_grid_output.csv")

    print(f"[done] Simulation complete. Generating plots...")
    sim_interface.generate_plots()
    print(f"[done] Plots saved.")
    
    
if __name__ == "__main__":
    run_simulation() 
    