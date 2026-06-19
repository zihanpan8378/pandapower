import simpy
import numpy as np
import csv
import json
import os
import multiprocessing
import pandapower.networks as pn

from tqdm import tqdm
from datetime import datetime

from datacenter import DataCenter
from grid_regions import GridRegion
from util import environment_updater, load_job_trace, datacenter_state_logger, init_users

from simulation_interface import GridSimulation
from sim_config import GridConfig, DataCenterConfig

from config import (
    UNIT_PRICES, EGRESS_RATE, BANDWIDTH, HARDWARE_MODELS, 
    RTT_MATRIX, NETWORK_ENERGY_PER_GB, WORKLOAD_TRACE_LENGTH_DAYS, SIMULATION_START_TIME, AWARE_QUEUE, 
    ALLOCATION_FAILED_RETRY, DECISION_METHOD, ELECTRICITY_PRICES
)

def run_simulation_case(params):
    seed = params['seed']
    capacity_factor = params['capacity_factor']
    carbon_scaling_factor = params['carbon_scaling_factor']
    shifting_mode = params['shifting_mode']
    
    # ==========================================
    # 1. Setup input and output file paths based on parameters
    # ==========================================
    user_datacenter_config_path = f"/mnt/grid-cloud-migration-estimates_copy/simulator/src/case_1/input/simulation_config_811_seed_{seed}.json"
    
    if shifting_mode != 'none':
        job_result_path = f"output/job_result_{shifting_mode}_shifting_S{seed}_F{capacity_factor:.2f}_C{carbon_scaling_factor:.3f}.csv"
        datacenter_result_path = f"output/datacenter_result_{shifting_mode}_shifting_S{seed}_F{capacity_factor:.2f}_C{carbon_scaling_factor:.3f}.csv"
    else:
        # Base case
        job_result_path = f"output/job_result_no_shifting_S{seed}.csv"
        datacenter_result_path = f"output/datacenter_result_no_shifting_S{seed}.csv"
        
    # ==========================================
    # 2. Environment config
    # ==========================================
    env = simpy.Environment()
    env.egress_rate = EGRESS_RATE
    env.bandwidth = BANDWIDTH
    env.hardware_model = HARDWARE_MODELS
    env.shifting_mode = shifting_mode
    env.simulation_start_time = SIMULATION_START_TIME
    env.rtt_matrix = RTT_MATRIX
    env.network_energy_per_gb = NETWORK_ENERGY_PER_GB
    env.workload_trace_length_days = WORKLOAD_TRACE_LENGTH_DAYS
    env.aware_queue = AWARE_QUEUE
    env.allocation_failed_retry = ALLOCATION_FAILED_RETRY
    env.decision_method = DECISION_METHOD
    env.carbon_scaling_factor = carbon_scaling_factor
        
    # ==========================================
    # 3. Initialize datacenters
    # ==========================================
    env.datacenters = []
    with open(user_datacenter_config_path, 'r') as f:
        config = json.load(f)
    datacenter_capacity_config = config['regions']

    # First pass: build grid configs only
    grid_configs: list[GridConfig] = []
    for dc_name, _ in datacenter_capacity_config.items():
        grid_config = GridConfig(
            pp_grid = pn.case300(), # type: ignore
            region  = GridRegion(dc_name)
        )
        grid_config.add_data_center(DataCenterConfig(
            load_share                  = 0.0,
            onsite_overprovision_factor = 1.2
        ))
        grid_configs.append(grid_config)

    grid_sim_interface: GridSimulation = GridSimulation(
        grid_configs=grid_configs,
        shifting_threshold=0.0,
        enable_weather_variation=False
    )
    env.grid_simulator_interface: GridSimulation = grid_sim_interface # type: ignore

    # Second pass: create DataCenter instances with the real interface
    env.datacenters = []
    for dc_name, dc_config in datacenter_capacity_config.items():
        gpu_cap = int(dc_config.get('gpu_slots_capacity', 100) * capacity_factor)
        cpu_cap = int(dc_config.get('cpu_capacity', 100) * capacity_factor)
        mem_cap = int(dc_config.get('mem_capacity', 100) * capacity_factor)

        grid_trace_path = f"/mnt/grid-cloud-migration-estimates_copy/data/grid/processed/{dc_name.replace('_', '-')}.csv"

        dc = DataCenter(
            env,
            dc_name,
            capacity=[gpu_cap, cpu_cap, mem_cap],
            unit_resource_prices=UNIT_PRICES[dc_name],
            pue=1.2,
            grid_trace_path=grid_trace_path,
            simulation_interface=grid_sim_interface,
            electricity_prices=ELECTRICITY_PRICES[dc_name]
        )
        env.datacenters.append(dc)

        
    # ==========================================
    # 4. Initialize users and job
    # ==========================================
    env.users_dict = {}
    init_users(env, user_datacenter_config_path)
    load_job_trace(env, user_datacenter_config_path)
    env.process(environment_updater(env))
    
    env.simulation_done = env.event()
    env.pbar = tqdm(total=env.num_total_job, disable=True) 
    
    # ==========================================
    # 5. Setup result buffers and CSV writers
    # ==========================================
    env.job_scheduling_result_buffer = []
    env.datacenter_state_buffer = []
    env.buffer_flush_threshold = 1000
    env.datacenter_state_log_interval = 60
    
    # Make sure output directory exists
    os.makedirs("output", exist_ok=True)
    
    f_job_scheduling = open(job_result_path, "w", newline="")
    env.results_writer = csv.DictWriter(f_job_scheduling, fieldnames=[
        "user_id", "job_id", 
        "home_dc", "target_dc", 
        "submit_time", 
        "num_retry", "temporal_delay", "failed_retry_delay", "shift_delay", "queue_delay", "duration", 
        "shift_cost", "compute_cost",
        "expected_shift_carbon", "expected_compute_carbon",
        "shift_carbon", "compute_carbon", 
        "expected_score", "score"
    ])
    env.results_writer.writeheader()
    
    f_datacenter_state = open(datacenter_result_path, "w", newline="")
    env.timeseries_writer = csv.DictWriter(f_datacenter_state, fieldnames=[
        "time", "time_human",
        "dc_name", "carbon_intensity",
        "cpu_total", "cpu_used", 
        "gpu_total", "gpu_used", 
        "mem_total", "mem_used", 
        "power",
        "peak_power",
        "cumulative_energy_consumed",
        "cumulative_energy_cost",
        "queue_length",
    ])
    env.timeseries_writer.writeheader()
    
    # ==========================================
    # 6. Run
    # ==========================================
    try:
        env.process(datacenter_state_logger(env))
        env.run(until=env.simulation_done)
        
        # Flush residual buffers
        if env.job_scheduling_result_buffer:
            env.results_writer.writerows(env.job_scheduling_result_buffer)
        if env.datacenter_state_buffer:
            env.timeseries_writer.writerows(env.datacenter_state_buffer)
    finally:
        f_job_scheduling.close()
        f_datacenter_state.close()
        env.pbar.close()
        
    return f"Success: Seed={seed}, Datacenter_Scale={capacity_factor:.2f}, Carbon_Scaling={env.carbon_scaling_factor:.4f}, Shifting={shifting_mode}"


if __name__ == "__main__":
    # ==========================================
    # Case Parameters
    # ==========================================
    seeds = [ 41 ] #, 42, 43]#, 44, 45, 46]
    
    # From 1.00 to 2.05, step 0.05, and a 4.0 at the end to represent unlimited capacity
    # capacity_factor = np.round(np.append(np.arange(1.00, 2.05, 0.05), 4.0), 2)
    capacity_factor = [1.5]
    
    carbon_scaling_factor = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    
    shifting_modes = ['spatial', 'temporal', 'spatial_temporal'] # 'temporal' can be added when temporal shifting logic is implemented
    
    tasks = []
    
    for seed in seeds:
        # Baseline case (no shifting)
        tasks.append({
            'seed': seed,
            'capacity_factor': 1.25,
            'carbon_scaling_factor': 0.1,
            'shifting_mode': 'none'
        })
        
        # 2. Cases with shifting
        for d in capacity_factor:
            for c in carbon_scaling_factor:
                tasks.append({
                    'seed': seed,
                    'capacity_factor': d,
                    'carbon_scaling_factor': c,
                    'shifting_mode': 'spatial'
                })
                # tasks.append({
                #     'seed': seed,
                #     'capacity_factor': d,
                #     'carbon_scaling_factor': c,
                #     'shifting_mode': 'temporal'
                # })
                # tasks.append({
                #     'seed': seed,
                #     'capacity_factor': d,
                #     'carbon_scaling_factor': c,
                #     'shifting_mode': 'spatial_temporal'
                # })

    print(f"Total simulation cases to run: {len(tasks)}")
    print(f"Using 15 CPU cores for parallel execution...")

    # ==========================================
    # Worker process pool
    # ==========================================
    pool = multiprocessing.Pool(processes=15)
    
    results = []
    with tqdm(total=len(tasks), desc="Total Progress") as pbar:
        for res in pool.imap_unordered(run_simulation_case, tasks):
            results.append(res)
            tqdm.write(res)
            pbar.update(1)
            
    pool.close()
    pool.join()
    
    print("All simulations completed successfully!")