import os

from typing import List, Dict

from grid_regions import GridRegion
from generation_types import GenerationType


# https://zenodo.org/records/17404776
# Carbon intensity values in gCO2eq/kwh sourced from above paper
CARBON_INTENSITIES: Dict[GenerationType, float] = {
    GenerationType.COAL: 95.0,
    GenerationType.SOLAR: 45.0,
    GenerationType.WIND: 0.0 # 11.0
}

ONSITE_GENERATION_BASE_COST: float = 0.0
ONSITE_GENERATION_COST_PER_MW: float = 0.0
ONSITE_GENERATION_COST_PER_MW2: float = 0.0

# Define paths for weather data per region
weather_data_base_path = "/mnt/grid-cloud-migration-estimates_copy/data/weather"
WEATHER_DATA_PATHS: Dict[GridRegion, str] = {
    GridRegion.CA_ON: os.path.join(weather_data_base_path, "processed", "CA-ON.csv"),
    GridRegion.US_CAL_CISO: os.path.join(weather_data_base_path, "processed", "US-CAL-CISO.csv"),
    GridRegion.US_MIDA_PJM: os.path.join(weather_data_base_path, "processed", "US-MIDA-PJM.csv"),
    GridRegion.US_TEX_ERCO: os.path.join(weather_data_base_path, "processed", "US-TEX-ERCO.csv"),
}
