import numpy as np
from datetime import datetime, timedelta
from collections import deque

import simpy

from config import WORKLOAD_TRACE_LENGTH_DAYS
from util import convert_time_to_human_readable
from simulation_interface import GridSimulation



class Grid_Condition:

    def __init__(self, 
        env: simpy.Environment, 
        name: str, 
        peak_demand_price: float,
        fixed_energy_price: float,
        grid_simulation_interface: GridSimulation
    ):
        self.env = env
        self.name = name
        
        # --- 1. Electricity bill components ---
        self.peak_power_price = peak_demand_price                    # $/kW
        self.fixed_energy_price = fixed_energy_price                 # $/kWh
        self.current_flexible_energy_price = 0.0                     # $/kWh
        self.peak_power = 0.0                                # kW
        self.cumulative_energy_consumed = 0.0                        # kWh
        self.cumulative_energy_cost = 0.0                            # $
    
        # --- 2. Carbon intensity ---
        # Index represents hours since beginning of simulation. Unpopulated hours are
        # NaN (not 0) so a not-yet-predicted hour is never mistaken for a genuinely
        # zero-carbon grid; reads go through carbon_intensity_at() which fills NaNs.
        self.carbon_intensity_horizon: np.ndarray = np.full(
            shape = 24 * (WORKLOAD_TRACE_LENGTH_DAYS + 1),
            fill_value = np.nan
        )

        #  --- 3. Grid Simulation Interface --- 
        self.grid_simulation_interface: GridSimulation = grid_simulation_interface

        
    def update_power_peak(self, current_power: float) -> None:
        """
        Update the peak power if the current power exceeds the previous peak. This is used for calculating the demand charge in electricity bill.

        Args:
            current_power: the current power consumption in kW
        """
        current_power_kw = current_power
        if current_power_kw > self.peak_power:
            self.peak_power = current_power_kw
            

    def update_environment_state(
        self, 
        start_time: datetime, 
        carbon_intensities: list[float],
        new_flexible_price: float
    ) -> None:
        """
        Update the grid condition based on the current time and the carbon intensity and energy price traces.
        
        Args:
            start_time: The current time as a datetime object
            carbon_intensities: A list of carbon intensities for the next 24 hours
            new_flexible_price: The new flexible energy price to update for this grid condition
        """
        self.current_flexible_energy_price = new_flexible_price
        seconds_in_hour = 3600
        start_time_idx = int((start_time - self.env.simulation_start_time).total_seconds() // seconds_in_hour) # type: ignore

        # Clamp the write to the preallocated capacity; predictions past the horizon are dropped
        # (get_grid_info falls back to an average for windows beyond the horizon)
        horizon_len = len(self.carbon_intensity_horizon)
        end_idx = min(start_time_idx + len(carbon_intensities), horizon_len)
        n = end_idx - start_time_idx
        if n > 0:
            self.carbon_intensity_horizon[start_time_idx:end_idx] = carbon_intensities[:n]


    def carbon_intensity_at(self, hour_idx: int) -> float:
        """
        Returns the predicted carbon intensity (gCO2/kWh) for an absolute hour index
        since the start of the simulation.

        Unpopulated hours are stored as NaN. Rather than treating those as a genuine
        zero-carbon grid, we persist the most recently known prediction (forward
        fill); if nothing has been predicted at or before this hour we use the
        earliest available prediction, and if the horizon is still completely empty
        we fall back to the live grid carbon intensity.

        Args:
            hour_idx: The absolute hour index since the start of the simulation

        Returns:
            The carbon intensity for that hour
        """
        horizon = self.carbon_intensity_horizon
        hour_idx = max(0, min(hour_idx, len(horizon) - 1))

        if not np.isnan(horizon[hour_idx]):
            return float(horizon[hour_idx])

        # Persist the most recent known prediction at or before this hour.
        known_before = horizon[: hour_idx + 1][~np.isnan(horizon[: hour_idx + 1])]
        if known_before.size > 0:
            return float(known_before[-1])

        # Nothing predicted at or before this hour: use the earliest known prediction,
        # or the live grid carbon intensity if the horizon is still completely empty.
        known_any = horizon[~np.isnan(horizon)]
        if known_any.size > 0:
            return float(known_any[0])
        return float(self.grid_simulation_interface.get_carbon_intensity_for_region(self.name))


    def get_grid_info(self, power: float, start_time: float, end_time: float) -> tuple[float, float]:

        total_carbon_integral: float    = 0
        total_cost_integral: float      = 0
        seconds_in_hour: int            = 3600

        start_datetime = self.env.simulation_start_time + timedelta(seconds=start_time) # type: ignore
        end_datetime = self.env.simulation_start_time + timedelta(seconds=end_time) # type: ignore
        
        horizon_hours = 24
        start_hour_idx = int((start_datetime - self.env.simulation_start_time).total_seconds() // seconds_in_hour) # type: ignore
        
        # If the job execution goes beyond the carbon intensity horizon, or exceeds a day,
        # We estimate carbon intesnity over that range by taking the average carbon intensity
        if (
            (end_datetime > self.env.simulation_start_time + timedelta(hours=len(self.carbon_intensity_horizon)))
            or (end_datetime - start_datetime).total_seconds() > horizon_hours * seconds_in_hour
            ): 
            # Take the average carbon intensity from start_datetime to end of horizon
            end_hour_idx = min(start_hour_idx + horizon_hours, len(self.carbon_intensity_horizon))
            avg_intensity = self.carbon_intensity_at(start_hour_idx)
            if end_hour_idx > start_hour_idx:
                avg_intensity = float(np.mean([
                    self.carbon_intensity_at(h) for h in range(start_hour_idx, end_hour_idx)
                ]))
            
            # Intensity * time * power = carbon emission
            total_carbon_integral = avg_intensity * (
                (end_datetime - start_datetime).total_seconds() / seconds_in_hour
                ) * power
            total_cost_integral = self.current_flexible_energy_price * (
                (end_datetime - start_datetime).total_seconds() / seconds_in_hour
                ) * power 
            return total_carbon_integral, total_cost_integral
            
        end_ci_hour_idx = int((end_datetime - self.env.simulation_start_time).total_seconds() // seconds_in_hour) # type: ignore

        for ci_hour_idx in range(start_hour_idx, end_ci_hour_idx + 1):

            # We integrate carbon intensity and energy price over time by taking the intensity and price at the beginning of 
            # each hour slot, and multiplying by the time in that slot until the next hour or the end time, whichever comes first.
            intensity = self.carbon_intensity_at(ci_hour_idx)
            seconds_to_end = (end_datetime - (self.env.simulation_start_time + timedelta(hours=ci_hour_idx))).total_seconds() # type: ignore
            time_in_slot = min(seconds_to_end, seconds_in_hour) 
            total_carbon_integral += intensity * (time_in_slot / seconds_in_hour) * power
            price = self.current_flexible_energy_price
            total_cost_integral += price * (time_in_slot / seconds_in_hour) * power
             
        return total_carbon_integral, total_cost_integral
    

class DataCenter:
    
    def __init__(
        self, 
        env: simpy.Environment, 
        name: str, 
        capacity: list[float], # gpu, mem, cpu
        unit_resource_prices: list[float], # gpu, mem, cpu price per hour
        pue: float, 
        electricity_prices: dict[str, float], 
        simulation_interface: GridSimulation,
        grid_trace_path: str | None = None
    ) -> None:
        self.env: simpy.Environment = env
        self.name: str = name
        
        # ---- 1. Resource capacity, availability, and price ----
        self.capacity: np.ndarray               = np.array(capacity, dtype=float)
        self.available: np.ndarray              = np.array(capacity, dtype=float)
        self.unit_resource_prices: list[float]  = unit_resource_prices
       
        # ---- 2. Power related ----
        self.pue: float                         = pue
        self.current_power: float               = 0
        self.grid_trace_path: str | None        = grid_trace_path
        self.max_power: float                   = 0.0
        self.min_power: float                   = 0.0

        # Nameplate (peak) and idle power are derived from the same per-unit power
        # model as get_job_power_kw so that current_power / max_power is a true
        # fraction in [0, 1]. Per-unit constants are in WATTS (see config.py
        # hardware_model); we apply PUE and convert W -> kW to match current_power.
        gpu_slots, mem_slots, cpu_slots = capacity
        _GPU_PEAK_W, _MEM_PEAK_W, _CPU_PEAK_W = 70.0, 0.838, 400 / 96
        _GPU_IDLE_W, _MEM_IDLE_W, _CPU_IDLE_W = 14.0, 0.326, 400 * 0.2 / 96

        self.max_power = (_GPU_PEAK_W * gpu_slots + _MEM_PEAK_W * mem_slots + _CPU_PEAK_W * cpu_slots) * self.pue / 1000  # kW

        # Idle floor: a real facility still draws power (idle servers, cooling) even
        # with no jobs running, so the grid load never collapses to zero.
        self.min_power = (_GPU_IDLE_W * gpu_slots + _MEM_IDLE_W * mem_slots + _CPU_IDLE_W * cpu_slots) * self.pue / 1000  # kW

        # ---- 3. Grid condition and simulation interface ----
        self.grid_simulation_interface: GridSimulation = simulation_interface

        self.grid_condition: Grid_Condition     = Grid_Condition(
            env = env, name = name,
            peak_demand_price           = electricity_prices['fixed_demand_price'],
            fixed_energy_price          = electricity_prices['fixed_energy_price'],
            grid_simulation_interface   = simulation_interface 
        )
        
        # ---- 3. Scheduling and queuing ----
        self.queue: deque                       = deque()
        self.max_backfill_depth: int            = 500
        self.scheduler_trigger: simpy.Event     = env.event() 

        # # Initialize carbon intensity horizon values to starting horizon
        # self.horizon_length = len(self.grid_condition.carbon_intensity_horizon)
        # for idx in range(self.horizon_length):
        #     self.grid_simulation_interface.run_power_flow(
        #         date   = self.env.simulation_start_time + timedelta(hours=idx), # type: ignore
        #     )
        #     self.grid_condition.carbon_intensity_horizon[idx] = self.grid_simulation_interface.get_carbon_intensity_for_region(
        #         region = self.name, 
        #     )

        # Start the scheduler process for this datacenter
        self.env.process(self.scheduler())


    def submit(self, job):
        """When a job arrives at the datacenter, add it to a queue..."""

        job['arrive_time'] = self.env.now
        job['power'] = self.get_job_power_kw(job)
        
        gpu_type = str(job['req'].get('gpu_type', 'MISC'))
        gpu_model = self.env.hardware_model['GPU'].get(gpu_type, self.env.hardware_model['GPU']['MISC']) # type: ignore
        
        job['req_gpu_slots'] = job['req']['gpu_count'] * gpu_model['slot']
        job['req_cpu'] = job['req']['cpu_count']
        job['req_mem'] = job['req']['mem_count']
        
        self.queue.append(job)
        
        if len(self.queue) <= self.max_backfill_depth:
            if not self.scheduler_trigger.triggered:
                self.scheduler_trigger.succeed()


    def scheduler(self):
        while True:
            if not self.queue:
                yield self.scheduler_trigger
                self.scheduler_trigger = self.env.event()
                
            while self.queue:
                scheduled_any = False
                still_waiting = []
                
                # Count how many jobs we have checked for backfilling
                check_count = min(self.max_backfill_depth, len(self.queue))
                
                # Pop jobs from the front of the queue
                for _ in range(check_count):
                    job = self.queue.popleft() 
                    
                    if self.allocate_resources(job):
                        job['start_time'] = self.env.now
                        self.env.process(self.run_job(job))
                        scheduled_any = True
                    else:
                        still_waiting.append(job)
                
                # Put the jobs that are still waiting back to the front of the queue, preserving their order
                if still_waiting:
                    # extendleft will reverse the order, so we first reverse still_waiting to maintain FIFO order
                    still_waiting.reverse()
                    self.queue.extendleft(still_waiting)
                
                if not scheduled_any:
                    break
            
            if self.queue:
                yield self.scheduler_trigger
                self.scheduler_trigger = self.env.event()

    def run_job(self, job):
        # Simulate the job execution time
        yield self.env.timeout(job['duration'])
        
        # Job finishes, release resources and log the scheduling result
        self.release_resources(job)
        
        job['end_time'] = self.env.now
        
        # Calculate this job's execution carbon emission
        job['compute_carbon'], job['compute_energy_cost'] = self.grid_condition.get_grid_info(job['power'], job['start_time'], job['end_time'])
        job_compute_energy = job['power'] * (job['duration'] / 3600) # kWh

        if job['home_region'] != self.name and job['shift_config']['transmission_delay'] > 0:
            net_start = job['arrive_time'] - job['shift_config']['transmission_delay']
            net_end = job['arrive_time']
            # Take the average carbon intensity of the source and target datacenter during the transmission period
            network_power = job['data_io'] * self.env.network_energy_per_gb / ((net_end - net_start) / 3600) # kW # type: ignore
            dest_shift_carbon, dest_shift_energy_cost = self.grid_condition.get_grid_info(network_power, net_start, net_end)
            source_shift_carbon, source_shift_energy_cost = job['shift_config']['home_dc'].grid_condition.get_grid_info(network_power, net_start, net_end)
            job['shift_carbon'] = (dest_shift_carbon + source_shift_carbon) / 2
            job['shift_energy_cost'] = (dest_shift_energy_cost + source_shift_energy_cost) / 2 
            job_shift_energy = network_power * (job['shift_config']['transmission_delay'] / 3600) # kWh
        else:
            job['shift_carbon'] = 0
            job['shift_energy_cost'] = 0
            job_shift_energy = 0
        
        self.grid_condition.cumulative_energy_cost += (job['compute_energy_cost'] + job['shift_energy_cost'])
        self.grid_condition.cumulative_energy_consumed += job_compute_energy + job_shift_energy
            
        total_carbon = job['compute_carbon'] + job['shift_carbon']
        total_cost = job['shift_config']['compute_cost'] + job['shift_config']['egress_cost']
        total_delay = job['start_time'] - job['submit_time']
        
        baseline_expected_carbon = job['shift_config']['baseline_expected_carbon']
        baseline_expected_cost = job['shift_config']['baseline_expected_cost']
        baseline_expected_delay = job['shift_config']['baseline_expected_delay']
        
        carbon_change_rate = (baseline_expected_carbon - total_carbon) / baseline_expected_carbon if baseline_expected_carbon > 0 else 0
        cost_change_rate = (total_cost - baseline_expected_cost) / baseline_expected_cost if baseline_expected_cost > 0 else 0
        duration_change_rate = (total_delay - baseline_expected_delay) / job['duration'] if job['duration'] > 0 else 0
        
        score = job['user'].carbon_sensitivity * carbon_change_rate * self.env.carbon_scaling_factor - job['user'].cost_sensitivity * cost_change_rate - job['user'].latency_sensitivity * duration_change_rate # type: ignore

        job_scheduling_record = {
            "user_id": job['user'].user_id,
            "job_id": job['job_id'],
            "home_dc": job['home_region'],
            "target_dc": self.name,
            
            "submit_time": round(job['submit_time'], 4),
            
            # Record these timesteps but not include them in the output to save space
            # "submit_time_human": convert_time_to_human_readable(job['submit_time']),
            # "arrive_time": round(job['arrive_time'], 4),
            # "arrive_time_human": convert_time_to_human_readable(job['arrive_time']),
            # "start_time": round(job['start_time'], 4),
            # "start_time_human": convert_time_to_human_readable(job['start_time']),
            # "end_time": round(job['end_time'], 4),
            # "end_time_human": convert_time_to_human_readable(job['end_time']),
            
            # "carbon_threshold": job['carbon_threshold'],
            # "cost_threshold": job['cost_threshold'],
            # "delay_threshold": job['latency_threshold'],
            # "temporal_slack": job['temporal_slack'],
            
            "num_retry": job['shift_config'].get('num_retry', 0),
            "temporal_delay": round(job['shift_config']['temporal_delay'], 4),
            "failed_retry_delay": round(job['shift_config'].get('failed_retry_delay', 0), 4),
            "shift_delay": round(job['shift_config']['transmission_delay'] + job['shift_config']['rtt_delay'], 4),
            "queue_delay": round(job['start_time'] - job['arrive_time'], 4),
            "duration": round(job['duration'], 4),
            
            "shift_cost": round(job['shift_config']['egress_cost'], 4),
            "compute_cost": round(self.get_job_cost(job), 4),
            
            # Record these energy cost but not include them in the output to save space
            # "shift_energy_cost": round(job['shift_energy_cost'], 4),
            # "compute_energy_cost": round(job['compute_energy_cost'], 4),
            
            "expected_shift_carbon": round(job['shift_config']['shift_carbon'], 4),
            "expected_compute_carbon": round(job['shift_config']['compute_carbon'], 4),
            
            "shift_carbon": round(job['shift_carbon'], 4),
            "compute_carbon": round(job['compute_carbon'], 4),
            
            "expected_score": round(job['shift_config']['score'], 4),
            "score": round(score, 4),
        }
        self.env.job_scheduling_result_buffer.append(job_scheduling_record)
        # Flush to disk if buffer exceeds threshold to avoid memory overflow
        if len(self.env.job_scheduling_result_buffer) >= self.env.buffer_flush_threshold: 
            self.env.results_writer.writerows(self.env.job_scheduling_result_buffer)
            self.env.job_scheduling_result_buffer.clear() # Clear buffer after flushing
        
        if not self.scheduler_trigger.triggered:
            self.scheduler_trigger.succeed()
            
        self.env.num_simulated_job += 1
        self.env.pbar.update(1)  # Update tqdm progress bar
        if self.env.num_simulated_job >= self.env.num_total_job and not self.env.simulation_done.triggered:
            self.env.simulation_done.succeed()  # Trigger simulation end event
            
            
    def __update_grid_datacenter_power(self) -> None:
        """
        Updates the data center's power share in the grid simulation interface based on its current power consumption. 
        """
        assert self.max_power > 0, f"Max power for data center {self.name} is zero, cannot update grid load share."
        # Fraction of dynamic (job) capacity in use, then mapped onto [min_power,
        # max_power] so an idle facility still draws min_power.
        dynamic_utilization = self.current_power / self.max_power
        facility_power = self.min_power + (self.max_power - self.min_power) * dynamic_utilization
        new_dc_power_share = facility_power / self.max_power
        dc_load_ids = self.grid_simulation_interface.get_load_ids_for_region(self.name)
        assert len(dc_load_ids) == 1, f"Expected 1 load id for region {self.name}, found {len(dc_load_ids)}"
        self.grid_simulation_interface.update_dc_load_share(
            region = self.name,
            load_ids = dc_load_ids,
            utilizations = [ new_dc_power_share ]
        )

           
    def allocate_resources(self, job) -> bool:
        """
        Allocates resources for a job if enough resources are available. Returns True if allocation is successful, False otherwise.

        Args:
            job: The job to allocate resources for

        Returns:
            True if resources are successfully allocated, False otherwise
        """

        enough_gpu = job['req_gpu_slots'] <= (self.available[0] + 1e-5)
        enough_cpu = job['req_cpu'] <= (self.available[1] + 1e-5)
        enough_mem = job['req_mem'] <= (self.available[2] + 1e-5)
        
        if enough_gpu and enough_cpu and enough_mem:
            self.available[0] -= job['req_gpu_slots']
            self.available[1] -= job['req_cpu']
            self.available[2] -= job['req_mem']
            self.current_power += job['power']
            
            self.__update_grid_datacenter_power()
            
            self.grid_condition.update_power_peak(self.current_power)
            return True
        return False
    
    
    def has_resource_available(self, job):
        gpu_type = str(job['req'].get('gpu_type', 'MISC'))
        gpu_model = self.env.hardware_model['GPU'].get(gpu_type, self.env.hardware_model['GPU']['MISC'])
        gpu_slots = job['req']['gpu_count'] * gpu_model['slot']
        
        enough_gpu = gpu_slots <= (self.available[0] + 1e-5)
        enough_cpu = job['req']['cpu_count'] <= (self.available[1] + 1e-5)
        enough_mem = job['req']['mem_count'] <= (self.available[2] + 1e-5)
        return enough_gpu and enough_cpu and enough_mem 
    

    def has_queue(self):
        return len(self.queue) > 0


    def release_resources(self, job) -> None:
        """
        Releases resources allocated to a job after it finishes.

        Args:
            job: The job for which to release resources
        """

        self.available[0] += job['req_gpu_slots']
        self.available[1] += job['req_cpu']
        self.available[2] += job['req_mem']    
        self.current_power -= job['power']

        self.__update_grid_datacenter_power()

        
    def get_job_power_kw(self, job) -> float:
        """
        Placeholder: Calculate the power consumption of a job based on its resource requirements and utilization
        """
        gpu_type = job['req']['gpu_type']
        gpu_count = job['req']['gpu_count']
        gpu_usage = job['req']['gpu_usage']
        gpu_power = gpu_count * (self.env.hardware_model['GPU'][gpu_type]['idle'] + (self.env.hardware_model['GPU'][gpu_type]['peak'] - self.env.hardware_model['GPU'][gpu_type]['idle']) * gpu_usage)
        
        cpu_count = job['req']['cpu_count']
        cpu_usage = job['req']['cpu_usage']
        cpu_power = cpu_count * (self.env.hardware_model['CPU']['default']['idle'] + (self.env.hardware_model['CPU']['default']['peak'] - self.env.hardware_model['CPU']['default']['idle']) * cpu_usage)
        
        mem_count = job['req']['mem_count']
        mem_usage = job['req']['mem_usage']
        mem_power = mem_count * (self.env.hardware_model['MEM']['default']['idle'] + (self.env.hardware_model['MEM']['default']['peak'] - self.env.hardware_model['MEM']['default']['idle']) * mem_usage)
        
        return (gpu_power + cpu_power + mem_power) * self.pue / 1000 # Convert from W to kW

    def get_job_cost(self, job):
        """
        Placeholder: Calculate the cost of running a job based on its resource requirements and duration
        """
        job_req = job['req']
        duration = job['duration']
        gpu_cost = job_req['gpu_count'] * self.unit_resource_prices[0] * self.env.hardware_model['GPU'][job_req['gpu_type']]['slot'] * (duration / 3600)
        cpu_cost = job_req['cpu_count'] * self.unit_resource_prices[1] * (duration / 3600)
        mem_cost = job_req['mem_count'] * self.unit_resource_prices[2] * (duration / 3600)
        return gpu_cost + cpu_cost + mem_cost