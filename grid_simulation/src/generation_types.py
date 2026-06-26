import pandas as pd

from pandas import DataFrame
from enum import Enum

from collections.abc import Callable


class GenerationType(Enum):
    # We only consider utility solar, not community or C & I
    SOLAR                   = "solar"
    SOLAR_NON_VARYING       = "solar_non_varying"
    WIND                    = "wind"
    WIND_NON_VARYING        = "wind_non_varying"
    COAL                    = "coal"
    BIOMASS                 = "biomass"
    # Gas is combined cycle, not gas peaking
    GAS                     = "gas"
    OIL                     = "oil"
    GEOTHERMAL              = "geothermal"
    HYDRO_RIVER             = "hydro_river"
    HYDRO_RESERVOIR         = "hydro_reservoir"
    MARINE                  = "marine"
    NUCLEAR                 = "nuclear"
    WASTE                   = "waste"


# Found from https://www.eia.gov/tools/faqs/faq.php?id=667&t=6
# For petroleum cost calculations, we use the following constants
_PETROLEUM_GALLON_PER_KWH: float = 0.08
_PETROLEUM_GALLON_PER_MWH: float = _PETROLEUM_GALLON_PER_KWH * 1000.0
_BARREL_TO_GALLON: float = 42.0
PETROLEUM_BARREL_PER_MWH: float = _PETROLEUM_GALLON_PER_MWH / _BARREL_TO_GALLON


_crude_oil_cost_filepath = "/mnt/pandapower/grid_simulation/src/Europe_Brent_Spot_Price_FOB_fixed.csv"
OIL_DOLLARS_PER_BARREL_BY_DAY: DataFrame = pd.read_csv(
    filepath_or_buffer  = _crude_oil_cost_filepath, 
    index_col           = "Date"
)

    
# Levelized cost of energy (LCOE) values in $/MWh sourced from Lazard's LCOE 2025 report
# With carbon pricing
# URI: https://www.lazard.com/media/5tlbhyla/lazards-lcoeplus-june-2025-_vf.pdf
# Cost is the average of the range provided in the report for each generation type
# Because we run pandapower simulation with a time step of 1 hour, we convert the LCOE values to $/MWh
LINEAR_GENERATION_COST_PER_MWh: dict[GenerationType, float] = {
    GenerationType.SOLAR:                       (50.0 + 131.0) / 2.0,
    GenerationType.SOLAR_NON_VARYING:           (50.0 + 131.0) / 2.0,

    # Price for these two sources are overriden since they are backup sources only
    GenerationType.WIND:                        0.0,
    GenerationType.WIND_NON_VARYING:            0.0,

    GenerationType.COAL:                        (108.0 + 249.0) / 2.0,
    
    # Estimated cost of biomass from the following source:
    # https://www.eia.gov/outlooks/aeo/electricity_generation/pdf/LCOE_report.pdf
    GenerationType.BIOMASS:                     84.54,
    
    GenerationType.GAS:                         (63.0 + 132.0) / 2.0,

    # Price of energy generation from oil depends on market
    GenerationType.OIL:                          0.0,

    GenerationType.GEOTHERMAL:                  (66.0 + 109.0) / 2.0,
    
    # Not included in Lazard's report, so from source below:
    # https://waterpowercanada.ca/wp-content/uploads/2026/02/True-Value-of-Hydropower-WPC-Report.pdf
    GenerationType.HYDRO_RIVER:                 103.88,  
    GenerationType.HYDRO_RESERVOIR:             103.88,
    
    GenerationType.MARINE: 0.0,
    GenerationType.NUCLEAR:                     (141.0 + 220.0) / 2.0,
    GenerationType.WASTE: 0.0,
}


BASE_GENERATION_COST: dict[GenerationType, float] = {
    GenerationType.SOLAR: 0.0,
    GenerationType.SOLAR_NON_VARYING: 0.0,
    GenerationType.WIND: 0.0,
    GenerationType.WIND_NON_VARYING: 0.0,
    GenerationType.COAL: 0.0,
    GenerationType.BIOMASS: 0.0,
    GenerationType.GAS: 0.0,
    GenerationType.OIL: 0.0,
    GenerationType.GEOTHERMAL: 0.0,
    GenerationType.HYDRO_RIVER: 0.0,
    GenerationType.HYDRO_RESERVOIR: 0.0,
    GenerationType.MARINE: 0.0,
    GenerationType.NUCLEAR: 0.0,
    GenerationType.WASTE: 0.0,
}


QUADRATIC_GENERATION_COST_PER_MWh2: dict[GenerationType, float] = {
    GenerationType.SOLAR: 0.0,
    GenerationType.SOLAR_NON_VARYING: 0.0,
    GenerationType.WIND: 0.0,
    GenerationType.WIND_NON_VARYING: 0.0,
    GenerationType.COAL: 0.0,
    GenerationType.BIOMASS: 0.0,
    GenerationType.GAS: 0.0,
    GenerationType.OIL: 0.0,
    GenerationType.GEOTHERMAL: 0.0,
    GenerationType.HYDRO_RIVER: 0.0,
    GenerationType.HYDRO_RESERVOIR: 0.0,
    GenerationType.MARINE: 0.0,
    GenerationType.NUCLEAR: 0.0,
    GenerationType.WASTE: 0.0,
}


# https://arxiv.org/pdf/2601.11623
# Carbon intensity values in gCO2eq/kwh sourced from above paper
# We add the operational and embodied carbon footprint
# CO2 Impact (gCO2eq/kWh)
SOURCE_CARBON_INTENSITIES: dict[GenerationType, float] = {
    GenerationType.COAL:                790.0 + 936.0,
    GenerationType.SOLAR:               36.95,
    GenerationType.WIND:                (14.4 + 12.4) / 2,

    GenerationType.SOLAR_NON_VARYING:   36.95,
    GenerationType.WIND_NON_VARYING:    (44.0 + 123.0) / 2,

    GenerationType.BIOMASS:             1030.0 + 230.0,
    GenerationType.GAS:                 370.0 + 434.0,
    GenerationType.OIL:                 600.0 + 778.0,
    GenerationType.GEOTHERMAL:          38.0,
    GenerationType.HYDRO_RIVER:         10.7,
    GenerationType.HYDRO_RESERVOIR:     10.7,
    GenerationType.MARINE:              17.0,
    GenerationType.NUCLEAR:             5.13,
    GenerationType.WASTE:               240.0 + 580.0,       
}
