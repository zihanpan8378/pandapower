import os

from typing import List, Dict

from grid_regions import GridRegion


OVERPROVISION_FACTOR: float = 0.7

# https://zenodo.org/records/17404776
# Carbon intensity values in gCO2eq/kwh sourced from above paper
CARBON_INTENSITIES: Dict[str, float] = {
    "coal": 95.0,
    "solar": 45.0,
    "wind": 11.0
}

# TODO: change energy costs, other constants to be per type based
NONRENEWABLE_BASE_COST: float = 100_000_000.0
RENEWABLE_BASE_COST: float = 100.0

SOURCE_BASE_COSTS: Dict[str, float] = {
    "coal": NONRENEWABLE_BASE_COST,
    "solar": RENEWABLE_BASE_COST,
    "wind": RENEWABLE_BASE_COST
}

SOURCE_LINEAR_COSTS: Dict[str, float] = {
    "coal": 10.0,
    "solar": 0.0,
    "wind": 0.0
}

# Define paths for weather data per region
weather_data_base_path = "/mnt/grid-cloud-migration-estimates_copy/data/weather"
WEATHER_DATA_PATHS: Dict[GridRegion, str] = {
    GridRegion.CA_ON: os.path.join(weather_data_base_path, "processed", "CA-ON.csv"),
    GridRegion.US_CAL_CISO: os.path.join(weather_data_base_path, "processed", "US-CAL-CISO.csv"),
    GridRegion.US_MIDA_PJM: os.path.join(weather_data_base_path, "processed", "US-MIDA-PJM.csv"),
    GridRegion.US_TEX_ERCO: os.path.join(weather_data_base_path, "processed", "US-TEX-ERCO.csv"),
}
