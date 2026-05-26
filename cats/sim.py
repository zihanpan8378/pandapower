import pandas as pd
import pandapower as pp

from utils import import_cats_geojson_to_pandapower

from pandapower.plotting import simple_plot
from pandapower.create import create_sgen

from pandapower import pandapowerNet
from pandas import DataFrame


cats_grid: pandapowerNet = pp.create_empty_network()

buses_csv = "CATS_buses.csv"
gens_csv = "CATS_gens.csv"
lines_json = "CATS_lines.json"

# Load bus and generator information from CSV files
bus_df: DataFrame = pd.read_csv(buses_csv) 
gens_df: DataFrame = pd.read_csv(gens_csv)

for bus in bus_df.itertuples():
    pp.create_bus(
        net = cats_grid,
        index = bus.bus_i, # type: ignore
        vn_kv = bus.kV, # type: ignore
        geodata = (bus.Lon, bus.Lat), # type: ignore
        type = "b" if (bus.Type == "substation") else "n", 
        classification = bus.Type
    )

# TODO: For all static generators, create pwl_cost

for gen in gens_df.itertuples():
    create_sgen(
        net = cats_grid,
        bus = gen.bus, # type: ignore
        p_mw = gen.Pg, # type: ignore
        q_mvar = gen.Qg, # type: ignore
        max_p_mw = gen.Pmax, # type: ignore
        min_p_mw = gen.Pmin, # type: ignore
        max_q_mvar = gen.Qmax, # type: ignore
        max_q_min = gen.Qmin, # type: ignore
        fuel_type = gen.FuelType, # type: ignore
        gen_id = gen.GenID, # type: ignore
        plant_code = gen.PlantCode, # type: ignore
        geodata_lon = gen.Lon, # type: ignore
        geodata_lat = gen.Lat
    )

import_cats_geojson_to_pandapower(
    lines_json_file = lines_json,
    net = cats_grid,
    s_base_mva = 100.0,
    freq_hz = 60.0
)

simple_plot(net = cats_grid)

