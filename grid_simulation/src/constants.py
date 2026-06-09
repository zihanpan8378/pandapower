import os

from typing import List, Dict


RENEWABLE_SOURCE_TYPES: List[str] = [
    "solar",
    "wind"
]

OVERPROVISION_FACTOR: float = 1.2

# https://zenodo.org/records/17404776
# Carbon intensity values in gCO2eq/kwh sourced from above paper
CARBON_INTENSITIES: Dict[str, float] = {
    "coal": 95.0,
    "solar": 45.0,
    "wind": 11.0
}

# Define paths for weather data per region
weather_data_base_path = "/mnt/grid-cloud-migration-estimates_copy/data/weather"
WEATHER_DATA_PATHS: Dict[str, str] = {
    "CA_ON": os.path.join(weather_data_base_path, "processed", "CA-ON.csv"),
    "US_CAL_CISO": os.path.join(weather_data_base_path, "processed", "US-CAL-CISO.csv"),
    "US_MIDA_PJM": os.path.join(weather_data_base_path, "processed", "US-MID-PJM.csv"),
    "US_TEX_ERCO": os.path.join(weather_data_base_path, "processed", "US-TEX-ERCO.csv"),
}

# TODO: change energy costs, other constants to be per type based
NONRENEWABLE_BASE_COST: float = 100_000_000.0
RENEWABLE_BASE_COST: float = 100.0


ALLOWED_GRID_REGIONS: List[str] = [
    "CA_ON",
    "US_CAL_CISO",
    "US_MIDA_PJM",
    "US_TEX_ERCO"
]