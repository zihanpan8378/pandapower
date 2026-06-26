import os

from grid_regions import GridRegion
from enum import Enum


ONSITE_GENERATION_BASE_COST: float      = 0.0
ONSITE_GENERATION_COST_PER_MW: float    = 0.0
ONSITE_GENERATION_COST_PER_MW2: float   = 0.0


# Define paths for weather data per region
weather_data_base_path = "/mnt/grid-cloud-migration-estimates_copy/data/weather"
WEATHER_DATA_PATHS: dict[GridRegion, str] = {
    GridRegion.CA_ON: os.path.join(weather_data_base_path, "processed", "CA-ON.csv"),
    GridRegion.US_CAL_CISO: os.path.join(weather_data_base_path, "processed", "US-CAL-CISO.csv"),
    GridRegion.US_MIDA_PJM: os.path.join(weather_data_base_path, "processed", "US-MIDA-PJM.csv"),
    GridRegion.US_TEX_ERCO: os.path.join(weather_data_base_path, "processed", "US-TEX-ERCO.csv"),
}


# Backup generation should be the most expensive source of generation 
# to ensure pandapower deploys it only when necessary to meet demand
BACKUP_GENERATION_BASE_COST: float      = 1000.0
BACKUP_GENERATION_COST_PER_MW: float    = 1000.0
BACKUP_GENERATION_COST_PER_MW2: float   = 0.0


class CustomGridGenElem(Enum):
    """
    Custom enumeration for grid generation elements.
    """
    GEN = "gen"
    SGEN = "sgen"
