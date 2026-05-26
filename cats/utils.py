import json
import math
import pandapower as pp

from typing import List, Collection, Any
from pandapower import pandapowerNet 


def haversine_distance_km(
    rad_lon1: float, 
    rad_lat1: float,
    rad_lon2: float, 
    rad_lat2: float,
) -> float:
    """
    Calculates the geographic distance between two lat/lon points in km.

    Args:
        rad_lon1: The longitude of the start point in radians.
        rad_lat1: The lattidue of the start point in radians.
        rad_lon2: The longitude of the end point in radians.
        rad_lat2: The lattidue of the end point.

    Return:
        distance_km: The harvesine distance between the two points in km.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = rad_lat2 - rad_lat1
    dlon = rad_lon2 - rad_lon1
    
    # Apply the harvesine formula
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rad_lat1) * math.cos(rad_lat2)) * math.sin(dlon / 2) ** 2 
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c
    
    return distance_km


def calculate_linestring_length(
    coordinates: List[Collection[float]]
) -> float:
    """
    Calculates the distance of a line in km.

    Args:
        coordinates: The geodisical coordinates that make up the line.

    Return:
        distance_km: The distance of the line in km.
    """

    total_length: float = 0.0
    for i in range(len(coordinates) - 1):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[i+1]
        total_length += haversine_distance_km(lon1, lat1, lon2, lat2)

    return total_length


def import_cats_geojson_to_pandapower(
    lines_json_file: str, 
    net: pandapowerNet,
    s_base_mva: float = 100.0, 
    freq_hz: float = 60.0
) -> None:
    """
    This function does the following:
        1. Loads the data from the lines json file.
        2. Preprocesses the data and converts fields to pandapower interface.
        3. Creates the tranmissions line in the given pandapower network.

    Args:
       lines_json_file: The name of the file containing the lines json data.
       net: The pandapower network.
       s_base_mva: The base power of the grid.
       freq_hz: The frequency of the grid.
    """

    with open(lines_json_file, 'r') as file:
        data = json.load(file)
        
    features = data.get('features', [])
    
    for feature in features:
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        if props.get('transformer', False):
            continue
            
        from_bus = props['f_bus']
        to_bus = props['t_bus']
        v_base_kv = props['kV']
        cats_id = props.get('CATS_ID', f"{from_bus}-{to_bus}")
        
        # 1. Calculate Actual Geographic Length
        coords = geom.get('coordinates', [])


        # Flatten coordinates array
        if coords and isinstance(coords[0][0], list):
            coords = [ point for segment in coords for point in segment]

        if coords and len(coords) >= 2:
            actual_length_km = calculate_linestring_length(coords)
        else:
            raise ValueError("Missing coordinates for line part")

        # 2. Calculate total values from per unit values
        z_base = (v_base_kv ** 2) / s_base_mva
        r_total_ohm = props['br_r'] * z_base
        x_total_ohm = props['br_x'] * z_base
        
        b_siemens = props['br_b'] / z_base
        c_total_nf = (b_siemens / (2 * math.pi * freq_hz)) * 1e9
        
        rate_a_mva = props['rate_a']
        max_i_ka = rate_a_mva / (math.sqrt(3) * v_base_kv) if rate_a_mva > 0 else 9999.0
            
        # 3. Convert Totals to "Per KM" values using the actual length
        r_ohm_per_km = r_total_ohm / actual_length_km
        x_ohm_per_km = x_total_ohm / actual_length_km
        c_nf_per_km = c_total_nf / actual_length_km
            
        # 4. Create Line in pandapower
        try:
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=actual_length_km, 
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=c_nf_per_km,
                max_i_ka=max_i_ka,
                name=f"Line_{cats_id}",
                in_service=True,
                index = feature["id"]
            )
        except Exception as e:
            print(f"Failed to create Line {cats_id}: {e}")