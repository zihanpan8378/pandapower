import json
from datetime import datetime, timedelta
import pandas as pd

from simulation_interface import GridSimulation
from user import User
from collections import defaultdict


def environment_updater(env):
    """
    Background coroutine: simulate the time-varying carbon intensity in the real world (Grid Feedback)
    Currently update the carbon intensity based on grid trace datasets ever hour and assume workload shifting has no effect on carbon intensity
    Will add grid simulation to analyze the impact of grid dynamics for other case studies
    """
    while True:
        current_datetime: datetime          = env.simulation_start_time + timedelta(seconds = env.now)
        current_datetime_slot: datetime     = current_datetime.replace(minute = 0, second = 0, microsecond = 0)
        grid_sim_interface: GridSimulation  = env.grid_simulator_interface
        prediction_window_hours: int        = 24

        new_flexible_price = 12 # Set constant for now
        start_datetime = current_datetime_slot

        # We store the predicted carbon intensity values per region
        ci_predictions_per_region: defaultdict[str, list[float]] = defaultdict(list)

        # Simulate the grid over the prediction window
        for _ in range(prediction_window_hours):

            grid_sim_interface.run_power_flow(date = current_datetime_slot)
            
            for datacenter in env.datacenters:
                new_carbon_intensity = grid_sim_interface.get_carbon_intensity_for_region(region = datacenter.name)
                ci_predictions_per_region[datacenter.name].append(new_carbon_intensity)

            current_datetime_slot += timedelta(hours=1)

        # Update the carbon intensity predictions in the grid condition for each datacenter
        for datacenter in env.datacenters:
            datacenter.grid_condition.update_environment_state(
                start_time = start_datetime,
                carbon_intensities = ci_predictions_per_region[datacenter.name], 
                new_flexible_price = new_flexible_price
            )
        
        # Refresh exactly when the current 24h window ends so consecutive persistent horizons tile with no gap
        next_update_time = int((start_datetime + timedelta(hours=prediction_window_hours) - current_datetime).total_seconds())
        yield env.timeout(next_update_time)
    

def load_job_trace(env, config_path):
    # Load Parquet
    parquet_path = config_path.replace('.json', '_jobs.parquet')
    df_jobs = pd.read_parquet(parquet_path)
    
    df_jobs = df_jobs[df_jobs['start_time'] < env.workload_trace_length_days * 24 * 3600] # Filter jobs that start within the simulation trace length
    df_jobs = df_jobs.sort_values(by='start_time') # Sort by job arrival time
    
    env.num_total_job = len(df_jobs)
    env.num_simulated_job = 0
    
    # Start job dispatcher
    env.process(job_dispatcher(env, df_jobs))
    return env.num_total_job

def job_dispatcher(env, df_jobs):
    """Background coroutine: dispatch jobs according to the job trace, and trigger User logic for each job arrival time"""
    for row in df_jobs.itertuples(index=False):
        delay = row.start_time - env.now
        if delay > 0:
            yield env.timeout(delay)
            
        user = env.users_dict[row.user_name]
        job_req = {
            'gpu_type': row.gpu_type, 
            'gpu_count': row.gpu_count, 'gpu_usage': row.gpu_usage,
            'cpu_count': row.cpu_count, 'cpu_usage': row.cpu_usage,
            'mem_count': row.mem_count, 'mem_usage': row.mem_usage,
        }
        
        duration = row.end_time - row.start_time
        env.process(user.generate_and_dispatch_job(
            job_id=row.job_id,
            job_req=job_req, 
            duration=duration, 
            data_io_size=row.network_data,
            temporal_slack=row.temporal_slack
        ))

def init_users(env, config_path):
    """Initialize User instances based on the simulation configuration (for synthetic job generation)"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    user_configs = config['users']
    
    for user_id, user_config in user_configs.items():
        env.users_dict[user_id] = User(
            env=env, user_id=user_id, home_region=user_config['home_region'], 
            thresholds = {'carbon': user_config['carbon_threshold'], 'cost': user_config['cost_threshold'], 'latency': user_config['latency_threshold']},
            sensitivities = {'carbon': user_config['weight_carbon'], 'cost': user_config['weight_cost'], 'latency': user_config['weight_latency']},
        )
  
def datacenter_state_logger(env):
    """
    Background coroutine: periodically log the state of datacenters (e.g., available resources, current carbon intensity, queue length)
    """
    
    while True:
        yield env.timeout(0.000001)
        for dc in env.datacenters:
            used = dc.capacity - dc.available

            # The horizon is indexed by absolute hour since simulation start, so index the current hour
            current_hour_idx = int(env.now // 3600)
            current_carbon_intensity = dc.grid_condition.carbon_intensity_at(current_hour_idx)

            record = {
                "time": round(env.now, 2),
                "time_human": convert_time_to_human_readable(env.now),
                "dc_name": dc.name,
                "carbon_intensity": round(current_carbon_intensity, 2),
                "gpu_total": round(dc.capacity[0], 2),
                "gpu_used": round(used[0], 2),
                "cpu_total": round(dc.capacity[1], 2),
                "cpu_used": round(used[1], 2),
                "mem_total": round(dc.capacity[2], 2),
                "mem_used": round(used[2], 2),
                "power": round(dc.current_power, 4),
                "peak_power": round(dc.grid_condition.peak_power, 4),
                "cumulative_energy_consumed": round(dc.grid_condition.cumulative_energy_consumed, 4),
                "cumulative_energy_cost": round(dc.grid_condition.cumulative_energy_cost, 4),
                "queue_length": len(dc.queue),
            }
            
            env.datacenter_state_buffer.append(record)
            # Flush to disk if buffer exceeds threshold to avoid memory overflow
        if len(env.datacenter_state_buffer) >= env.buffer_flush_threshold:
            env.timeseries_writer.writerows(env.datacenter_state_buffer)
            env.datacenter_state_buffer.clear() # Clear buffer after flushing
            
        yield env.timeout(env.datacenter_state_log_interval)
 
def convert_time_to_human_readable(seconds):
    """Convert simulation time in seconds back to human-readable format for logging purposes"""
    days = int(seconds // 86400)
    remaining = seconds % 86400
    hours = int(remaining // 3600)
    remaining = remaining % 3600
    minutes = int(remaining // 60)
    secs = int(remaining % 60)
    milliseconds = int((remaining - int(remaining)) * 1000)
    
    return f"{days}d-{hours}h-{minutes}m-{secs}s-{milliseconds}ms"