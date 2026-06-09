import csv
import numpy as np

from typing import List


rng = np.random.default_rng()

def generate_power_trace(num_data_centers: int, init_ci: List[float]) -> None:
    """
    Generates a power trace that emulates the behaviour of carbon-aware workload
    shifting between data centers.

    Args:
       num_data_centers: The number of data centers involved in the shifting. 
    """
    assert num_data_centers == len(init_ci)
    dc_to_ci = { idx: ci for idx, ci in enumerate(init_ci)}



    
