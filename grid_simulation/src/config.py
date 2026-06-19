import numpy as np
from datetime import datetime

# ==========================================
# 1. Hardware & Infrastructure Configs
# ==========================================
# Resource price [GPU_price, CPU_price, Mem_price] per hour
UNIT_PRICES: dict[str, np.ndarray] = {# Google Cloud north virginia, https://cloud.google.com/products/compute/pricing/general-purpose, C4 instances
    "CA_ON": np.array([0.35, 0.03465, 0.003938]), 
    "US_CAL_CISO": np.array([0.35, 0.03465, 0.003938]),
    "US_MIDA_PJM": np.array([0.35, 0.03465, 0.003938]),
    "US_TEX_ERCO": np.array([0.35, 0.03465, 0.003938])
}

ELECTRICITY_PRICES: dict[str, dict[str, float]] = { # [$/kWh, $/kW]
    "CA_ON": {'fixed_energy_price': 0.0, 'fixed_demand_price': 12}, # Placeholder values from https://dl.acm.org/doi/epdf/10.1145/2000064.2000105
    "US_CAL_CISO": {'fixed_energy_price': 0.0, 'fixed_demand_price': 12},
    "US_MIDA_PJM": {'fixed_energy_price': 0.0, 'fixed_demand_price': 12},
    "US_TEX_ERCO": {'fixed_energy_price': 0.0, 'fixed_demand_price': 12},
}

HARDWARE_MODELS: dict[str, dict[str, dict[str, float]]] = { # in W
    "GPU": { # Per GPU, slot weight is estimated by price from https://cloud.google.com/compute/gpus-pricing
        "T4":      {"idle": 14, "peak": 70, "slot": 1}, # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-product-brief.pdf, assume 20% idle power
        "V100":    {"idle": 50, "peak": 250, "slot": 7}, # https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf, assume 20% idle power
        "V100M32": {"idle": 50, "peak": 250, "slot": 8}, # https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf, assume 20% idle power
        "P100":    {"idle": 50, "peak": 250, "slot": 4}, # https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf, assume 20% idle power
        "MISC":    {"idle": 40, "peak": 200, "slot": 1}, # No specs available, assume lower than P100 and V100 and 20% idle power
        "None":    {"idle": 0, "peak": 0, "slot": 0}
    },
    "CPU": { # Per core
        "default": {"idle": 400 * 0.2 / 96, "peak": 400 / 96} # Assume 20% idle power and 400W for 96 cores AMD EPYC™ 9655 https://www.amd.com/en/products/processors/server/epyc/9005-series.html#specifications
    },
    "MEM": { # Per GB
        "default": {"idle": 0.326, "peak": 0.838} # From Hanan's paper
    }
}

# ==========================================
# 2. Network Configs
# ==========================================
EGRESS_RATE: float = 0.02  # Cross region egress fee ($/GB)
BANDWIDTH: int = 10     # Cross region transmission bandwidth (GB/tick)
NETWORK_ENERGY_PER_GB: float = 0.005 # kWh per GB, from Hanan's paper

RTT_MATRIX: dict[str, dict[str, float]] = { # in seconds, from https://doi.org/10.1145/3627703.3650079 repo
    "CA_ON": {
        "CA_ON": 0.000, "US_CAL_CISO": 0.0670, "US_MIDA_PJM": 0.0210, "US_TEX_ERCO": 0.0330
    },
    "US_CAL_CISO": {
        "CA_ON": 0.0670, "US_CAL_CISO": 0.000, "US_MIDA_PJM": 0.0650, "US_TEX_ERCO": 0.0340
    },
    "US_MIDA_PJM": {
        "CA_ON": 0.0210, "US_CAL_CISO": 0.0650, "US_MIDA_PJM": 0.000, "US_TEX_ERCO": 0.0320
    },
    "US_TEX_ERCO": {
        "CA_ON": 0.0330, "US_CAL_CISO": 0.0340, "US_MIDA_PJM": 0.0320, "US_TEX_ERCO": 0.000
    }
}

# ==========================================
# 3. Simulation Configs
# ==========================================
WORKLOAD_TRACE_LENGTH_DAYS: int = 28 # Simulate with 28 days workload trace
SIMULATION_START_TIME: datetime = datetime.strptime('2020-08-01 00:00:00', '%Y-%m-%d %H:%M:%S')
AWARE_QUEUE: bool = False
ALLOCATION_FAILED_RETRY: bool = True
DECISION_METHOD: str = 'score' # 'score' or 'threshold'
# Other configs are defined in the running scripts