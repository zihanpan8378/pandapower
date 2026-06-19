import simpy

from simulation_interface import GridSimulation

from typing import Generator, Any


class User:

    def __init__(
        self, 
        env: simpy.Environment, 
        user_id: int, 
        home_region: str, 
        thresholds: dict, 
        sensitivities: dict,
    ) -> None:
        self.env: simpy.Environment             = env
        self.user_id: int                       = user_id
        self.home_region: str                   = home_region # User's default region, used to determine if migration is needed

        # self.carbon_threshold = thresholds['carbon']
        self.carbon_threshold: float            = 0
        self.cost_threshold: float              = thresholds['cost']
        self.latency_threshold: float           = thresholds['latency']

        self.carbon_sensitivity: float          = sensitivities['carbon']
        self.cost_sensitivity: float            = sensitivities['cost']
        self.latency_sensitivity: float         = sensitivities['latency']

        self.jobs_submitted: int                = 0


    def generate_and_dispatch_job(
        self, 
        job_id: int, 
        job_req: dict, 
        duration: float, 
        data_io_size: float, 
        temporal_slack: float = 0.0
    ) -> Generator[Any, Any, None]:
        """_summary_

        Args:
            job_id (int): _description_
            job_req (dict): _description_
            duration (float): _description_
            data_io_size (float): _description_
            temporal_slack (float, optional): _description_. Defaults to 0.0.

        Yields:
            Generator[Any, Any, None]: _description_
        """

        self.jobs_submitted += 1
        job_name = f"U{self.user_id}-J{self.jobs_submitted}"
        
        job = {
            'job_id': job_id,
            'job_name': job_name,
            'user': self,
            'req': job_req,
            'duration': duration,
            'data_io': data_io_size,
            'submit_time': self.env.now,
            'home_region': self.home_region,
            'carbon_threshold': self.carbon_threshold,
            'cost_threshold': self.cost_threshold,
            'latency_threshold': self.latency_threshold,
            'temporal_slack': temporal_slack
        }
        
        # TODO: Change this to use grid simulation 
        
        # ==================== Baseline (No Shifting) ======================
        home_datacenter = [dc for dc in self.env.datacenters if dc.name == self.home_region][0] # type: ignore
        home_compute_cost = home_datacenter.get_job_cost(job)
        home_job_power = home_datacenter.get_job_power_kw(job)
        
        home_compute_carbon, _ = home_datacenter.grid_condition.get_grid_info(home_job_power, self.env.now, self.env.now + duration)
        
        no_shift_config = {
            "home_dc": home_datacenter,
            "target_dc": home_datacenter,
            "temporal_delay": 0,        # Delay due to temporal shifting, waiting for a greener time window
            "transmission_delay": 0,    # Delay due to spatial shifting, network transfer time
            "rtt_delay": 0,             # Round-trip time delay
            "compute_cost": home_compute_cost,
            "egress_cost": 0,
            "compute_carbon": home_compute_carbon,
            "shift_carbon": 0,
            "score": 0,
        }
        
        baseline_expected_result = {
            "baseline_expected_cost": no_shift_config["compute_cost"],
            "baseline_expected_carbon": no_shift_config["compute_carbon"],
            "baseline_expected_delay": 0
        }
        
        # ==================== Search Space ======================
        shifting_mode = self.env.shifting_mode
        
        # Spatial shifting candidates
        if shifting_mode in ["spatial", "spatial_temporal"]:
            spatial_candidates = self.env.datacenters
        elif shifting_mode == "temporal":
            spatial_candidates = [home_datacenter]
        else:
            spatial_candidates = []
        
        if shifting_mode in ["temporal", "spatial_temporal"]:
            max_delay = temporal_slack
            temporal_candidates = list(range(0, int(max_delay) + 1, 3600))
            if max_delay not in temporal_candidates:
                temporal_candidates.append(max_delay)
        else:
            temporal_candidates = [0]
            
        if self.env.aware_queue:
            spatial_candidates = [dc for dc in spatial_candidates if not dc.has_queue()]
            if home_datacenter not in spatial_candidates:
                spatial_candidates.append(home_datacenter) # Ensure the home datacenter is always a candidate
            
        # ==================== Decision Making ======================
        valid_configs = [no_shift_config]
        
        for dc in spatial_candidates:
            for temporal_delay in temporal_candidates:
                # Skip the baseline case without any shifting
                if dc.name == self.home_region and temporal_delay == 0:
                    continue
                
                # Network latency estimation for spatial shifting
                transmission_duration = 0 if ((self.env.bandwidth == 0) or (self.home_region == dc.name)) else data_io_size / self.env.bandwidth
                rtt_delay = self.env.rtt_matrix[self.home_region][dc.name] if dc.name != self.home_region else 0
                shift_delay = (transmission_duration + rtt_delay) if dc.name != self.home_region else 0
        
                # Cost estimation
                compute_cost = dc.get_job_cost(job)
                egress_cost = 0 if dc.name == self.home_region else data_io_size * self.env.egress_rate
                
                # Carbon estimation
                job_power = dc.get_job_power_kw(job)
                plan_start_time = self.env.now + temporal_delay + shift_delay
                plan_end_time = plan_start_time + duration

                # Grid has no concept of time, we first apply the shift to see the compute carbon impact
                compute_carbon, _ = dc.grid_condition.get_grid_info(job_power, plan_start_time, plan_end_time)

                # if self.emission_prediction_method == "oracle_intensity":
                #     compute_carbon, _ = dc.grid_condition.get_grid_info(job_power, plan_start_time, plan_end_time)
                # elif self.emission_prediction_method == "current_intensity":
                #     compute_carbon = job_power * dc.grid_condition.current_carbon * (duration / 3600)
                # else:
                #     compute_carbon = 0 # Placeholder for other prediction methods
                    
                shift_carbon = 0
                # if dc.name != self.home_region and transmission_duration > 0:
                #     network_power = data_io_size * self.env.network_energy_per_gb / (transmission_duration / 3600) # kW
                #     if self.emission_prediction_method == "oracle_intensity":
                #         net_start = self.env.now + temporal_delay
                #         net_end = net_start + transmission_duration
                #         # Take the average carbon intensity of the source and target datacenter during the transmission period
                #         source_shift_carbon, _ = home_datacenter.grid_condition.get_grid_info(network_power, net_start, net_end)
                #         target_shift_carbon, _ = dc.grid_condition.get_grid_info(network_power, net_start, net_end)
                #         shift_carbon = (source_shift_carbon + target_shift_carbon) / 2
                #     elif self.emission_prediction_method == "current_intensity":
                #         shift_carbon = (
                #             network_power * dc.grid_condition.current_carbon * (transmission_duration / 3600) + 
                #             network_power * home_datacenter.grid_condition.current_carbon * (transmission_duration / 3600)
                #         ) / 2
                #     else:
                #         shift_carbon = 0 # Placeholder for other prediction methods

                if dc.name != self.home_region and transmission_duration > 0:
                    network_power = data_io_size * self.env.network_energy_per_gb / (transmission_duration / 3600) # kW # type: ignore
                    net_start = self.env.now + temporal_delay
                    net_end = net_start + transmission_duration
                    # Take the average carbon intensity of the source and target datacenter during the transmission period
                    source_shift_carbon, _ = home_datacenter.grid_condition.get_grid_info(network_power, net_start, net_end)
                    target_shift_carbon, _ = dc.grid_condition.get_grid_info(network_power, net_start, net_end)
                    shift_carbon = (source_shift_carbon + target_shift_carbon) / 2
                                        
                shift_config = {
                    "home_dc": home_datacenter,
                    "target_dc": dc,
                    "transmission_delay": transmission_duration,
                    "rtt_delay": rtt_delay,
                    "temporal_delay": temporal_delay,
                    "compute_cost": compute_cost,
                    "egress_cost": egress_cost,
                    "compute_carbon": compute_carbon,
                    "shift_carbon": shift_carbon,
                    "score": 0,
                }
        
                base_total_carbon = baseline_expected_result["baseline_expected_carbon"]
                new_total_carbon = shift_config["compute_carbon"] + shift_config["shift_carbon"]
                carbon_change_rate = (base_total_carbon - new_total_carbon) / base_total_carbon if base_total_carbon > 0 else 0
                
                base_total_cost = baseline_expected_result["baseline_expected_cost"]
                new_total_cost = shift_config["compute_cost"] + shift_config["egress_cost"]
                cost_change_rate = (new_total_cost - base_total_cost) / base_total_cost if base_total_cost > 0 else 0
                
                duration_change_rate = (shift_config["transmission_delay"] + shift_config["rtt_delay"]) / duration if duration > 0 else 0
                
                if self.env.decision_method == 'score':
                    score = self.carbon_sensitivity * carbon_change_rate * self.env.carbon_scaling_factor - self.cost_sensitivity * cost_change_rate - self.latency_sensitivity * duration_change_rate
                    shift_config["score"] = score
                    valid_configs.append(shift_config)
                elif self.env.decision_method == 'threshold':
                    shift_config["score"] = -(shift_carbon + compute_carbon)
                    if carbon_change_rate > self.carbon_threshold and cost_change_rate < self.cost_threshold and duration_change_rate < self.latency_threshold:
                        valid_configs.append(shift_config)
        
        # ==================== Try Submit ======================
        
        # Sort the valid configs based on the total carbon
        valid_configs.sort(key=lambda x: x["score"], reverse=True)
        best_score = valid_configs[0]["score"]
        
        # Temporal shifting delay
        if valid_configs[0]["temporal_delay"] > 0:
            yield self.env.timeout(valid_configs[0]["temporal_delay"])
        
        if self.env.allocation_failed_retry and shifting_mode == 'spatial':
            successful_config = None
            accumulated_retry_delay = 0
            num_retry = 0
            
            if not (len(valid_configs) == 1 and valid_configs[0]["target_dc"].name == self.home_region):
            
                for config in valid_configs:
                    target_dc = config["target_dc"]
                
                    # 1. Sending API request to target data center
                    handshake_delay = config["rtt_delay"]
                    yield self.env.timeout(handshake_delay)
                    
                    # 2. Datacenter check if it still has capacity to accept the job
                    if target_dc.has_resource_available(job):
                        # Has capacity, go with this region
                        successful_config = config
                        break 
                    else:
                        # No capacity, try the next candidate after a short delay (simulate control plane processing time)
                        yield self.env.timeout(0.5)
                        accumulated_retry_delay += handshake_delay
                        accumulated_retry_delay += 0.5 # Assume 0.5s for control plane processing
                        num_retry += 1
                        continue
            
            # If all candidates fail, fallback to the no-shift option
            if successful_config is None:
                successful_config = no_shift_config
                successful_config["transmission_delay"] = 0
            
            # Record the accumulated failed retry delay for this job
            successful_config['failed_retry_delay'] = accumulated_retry_delay
            
            # Transfer the data to the target datacenter
            if successful_config["transmission_delay"] > 0:
                yield self.env.timeout(successful_config["transmission_delay"])
                
            successful_config["num_retry"] = num_retry
            successful_config["score"] = best_score # The user was expected to get the best score, but ended up with retries
            job['shift_config'] = successful_config
            job['shift_config'].update(baseline_expected_result)
            successful_config["target_dc"].submit(job)
        else:
            # For the case of queueing or temporal shifting, we directly go with the best valid config
            job['shift_config'] = valid_configs[0]
            job['shift_config'].update(baseline_expected_result)
            total_delay_before_submit = job['shift_config']["transmission_delay"] + job['shift_config']["rtt_delay"] + job['shift_config']["temporal_delay"]
            if total_delay_before_submit > 0:
                yield self.env.timeout(total_delay_before_submit)
            job['shift_config']["target_dc"].submit(job)