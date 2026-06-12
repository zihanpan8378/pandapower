import os
import argparse
import pandapower.networks as pn
import pandas as pd
import pandapower as pp

from grid_custom import DatacenterGrid


def export_network_summary(net, filename="network_summary.csv", save_csv=False):
    """
    Takes a pandapower network, counts the elements in each component table,
    and optionally exports the breakdown to a CSV file.
    """
    summary_data = []
    
    # Iterate through every attribute in the pandapower network
    for component_name, component_data in net.items():
        # Component tables in pandapower are stored as pandas DataFrames
        if isinstance(component_data, pd.DataFrame):
            count = len(component_data)
            
            # Only record components that actually have elements in them
            if count > 0:
                summary_data.append({
                    "Component": component_name,
                    "Count": count
                })
                
    # Convert our list of counts into a DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Sort alphabetically by component name for easier reading
    df_summary = df_summary.sort_values(by="Component").reset_index(drop=True)
    
    # Conditionally export to CSV based on the command line argument
    if save_csv:
        df_summary.to_csv(filename, index=False)
        print(f"Successfully exported {len(df_summary)} component types to '{filename}'.")
    else:
        print(f"Analyzed {len(df_summary)} component types. (CSV export disabled)")
        
    print("\nPreview of counts:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Analyze a Pandapower grid and optionally export a component summary.")
    
    # action="store_true" means it defaults to False, but becomes True if the flag is used
    parser.add_argument("--save-csv", action="store_true", help="Include this flag to save the output to a CSV file.")
    
    args = parser.parse_args()

    # 1. Load the network
    # grid_m = os.path.join('central-illinois-200', 'case_ACTIVSg200.m')
    # net = from_mpc(grid_m)
    net = pn.case300()
    grid = DatacenterGrid(net=net, grid_region="CA_ON") # type: ignore
    # print(net.poly_cost)
    # print(net.gen)
    # print(net.line["max_i_ka"])

    # 2. Run the export function, passing the boolean argument
    export_network_summary(grid._net, filename="case300_breakdown.csv", save_csv=args.save_csv)