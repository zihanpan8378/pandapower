import numpy as np

from pandapower import pandapowerNet
from typing import Tuple


# All types of line elements and their corresponding powerflow result, bus connection, and active power columns
_BRANCH_SPECS = (
    # (element_table, result_table, from_col,  to_col,   flow_col)
    ("line",      "res_line",      "from_bus", "to_bus", "p_from_mw"),
    ("trafo",     "res_trafo",     "hv_bus",   "lv_bus", "p_hv_mw"),
    ("impedance", "res_impedance", "from_bus", "to_bus", "p_from_mw"),
)

# Types of generation components considered by this bialek's tracing implementation for gross supply,
# and their corresponding powerflow result column
_GENERATION_SOURCES = [
    ("ext_grid", "res_ext_grid"),
    ("gen", "res_gen"),     # Voltage-controlled generatos
    ("sgen", "res_sgen"),   # Constant power generators

]

# Types of demand components considered by this bialek's tracing implementation for gross demand,
# and their correspodning poweflow result column
_DEMAND_SOURCES = [
    ("load", "res_load"),
    ("shunt", "res_shunt"),
    ("ward", "res_ward"),
    ("xward", "res_xward"),
    ("motor", "res_motor")
]


def _collect_branches(net: pandapowerNet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single source of truth for the branch axis of the trace.
 
    Walks _BRANCH_SPECS in order, keeps only in-service branches, and returns
    aligned arrays so the incidence matrix and the flow vector are guaranteed
    to share the same column ordering.

    Args:
        net: The pandapower net to collect networks
 
    Returns:
        from_bus : (M,) sending-bus IDs
        to_bus   : (M,) receiving-bus IDs
        flows    : (M,) sending-end active power (MW)
    """
    # Buffers for line start, end, and powerflow
    from_ids, to_ids, flows = [], [], []

    # Collect all the branch flows for all possible pandapower line elements
    for elem, res, from_col, to_col, flow_col in _BRANCH_SPECS:

        if elem not in net or len(net[elem]) == 0:
            continue

        # Get the underlying dataframe for that element
        tbl = net[elem]

        # Only consider elements that are 'in_service'
        if "in_service" in tbl.columns:
            mask = tbl["in_service"].values.astype(bool)
        else:
            # Assume every element is in_service if the 'in_service' column doesn't exist
            mask = np.ones(len(tbl), dtype=bool)

        # Ensure that flow direction is consistent w.r.t entire grid
        from_ids.append(tbl[from_col].values[mask])
        to_ids.append(tbl[to_col].values[mask])
        flows.append(net[res][flow_col].values[mask])

    # Return empty arrays if there are no lines
    if not from_ids:
        empty = np.array([])
        return empty, empty, empty
 
    return (np.concatenate(from_ids),
            np.concatenate(to_ids),
            np.concatenate(flows).reshape(-1))


def extract_branch_flows(net: pandapowerNet) -> np.ndarray:
    """
    Sending-end active power for all in-service two-terminal branches,
    in the SAME order as create_grid_incidence_matrix (both delegate to
    _collect_branches, so column alignment is structural rather than
    a convention the two functions must remember to share).

    Args:
        net: The given pandapower network

    Returns:
        The branch flows for the network
    """
    _, _, flows = _collect_branches(net)
    return flows


def create_grid_incidence_matrix(net: pandapowerNet) -> np.ndarray:
    """
    Bus-branch incidence matrix over all in-service two-terminal branches
    (lines, transformers, impedances). Rows = contiguous bus indices,
    columns = branches. +1 at the sending bus, -1 at the receiving bus.

    Let M be the incidence matrix this function returns.
    M[i,j] =  1 if bus i is the starting bus for line j,
             -1 if bus i is the ending bus for line j,
              0 otherwise

    Args:
        net: The given pandapower network.

    Returns:
        The incidence matrix of the network
    """
    bus_count = len(net.bus)

    # Return empty matrix if grid has no buses
    if bus_count == 0:
        return np.empty(shape=(0, 0))
 
    # Maps pandapower's bus_id to the underlying bus dataframe index
    # Both uniquely identify bus, but can be different
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(net.bus.index)}
    
    # Get all line connections
    from_bus, to_bus, _ = _collect_branches(net)
    branch_count = len(from_bus)
 
    res_mat = np.zeros(shape=(bus_count, branch_count))
    for line_idx in range(branch_count):
        line_start_bus_df_idx = bus_idx_map[from_bus[line_idx]]
        line_end_bus_df_idx = bus_idx_map[to_bus[line_idx]]
        res_mat[line_start_bus_df_idx, line_idx] = 1
        res_mat[line_end_bus_df_idx, line_idx] = -1

    return res_mat


def gross_gen_demand(net: pandapowerNet) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-bus gross generation and demand (MW), summed from element tables.
 
    Need gross generation and demadn because 
    res_bus['p_mw'] is the net injection and cannot be used for bialek's tracing.
    A single bus can host both generation and load simultaneously, and the
    netted value loses that information (and mislabels which buses are sources / loads).

    Note: This function assumes that all generation and demand components in 'net' are in
    '_GENERATION_SOURCES' and '_DEMAND_SOURCES' respectively.

    Args:
        net: The pandapower network to calculate the gross demands for
        
    Returns:
        P_G: The per-bus gross generation in MW
        P_D: The per-bus gross demand in MW
    """
    bus_count = len(net.bus)
    bus_idx_map = {bus_id: i for i, bus_id in enumerate(net.bus.index)}

    # Buffers for gross supply and demand of power
    P_G: np.ndarray = np.zeros(bus_count) # P_G[i] is the gross generation at bus i
    P_D: np.ndarray = np.zeros(bus_count) # P_D[i] is the gross demand at bus i

    # HELPER FUNCTIONS TO UPDATE GROSS DEMAND AND GENERATION

    def _in_service_mask(elem: str) -> np.ndarray:
        """
        Returns the in-service mask for the given 'elem'.

        Args:
            elem: The name of the element

        Returns:
            in_service: Shape (len(elem),), 
                        in_service[i] is true if element i is in service, and false otherwise
        """
        tbl = net[elem]
        if "in_service" in tbl.columns:
            return tbl["in_service"].values.astype(bool)
        return np.ones(len(tbl), dtype=bool)
    

    def _update_gross_demand(field_and_result: Tuple[str, str]) -> None:
        """
        Updates 'P_D' by incorporating contributions from the demand
        element given by 'field_and_result'.

        It is possible for a demand element to supply power instead of consuming.
        
        Args:
            field_and_result[0]: The name of the demand element (i.e. load)
            field_and_result[1]: The name of the result powerflow table name (i.e. res_load)
        """
        elem, res = field_and_result
        if elem not in net or len(net[elem]) == 0:
            return

        mask = _in_service_mask(elem)
        buses = net[elem]["bus"].values[mask]
        results = net[res]["p_mw"].values[mask]

        for bus_id, val in zip(buses, results):
            # Positive value for 'p_mw' is demand, negative is supply
            if val >= 0:
                P_D[bus_idx_map[bus_id]] += val
            else:
                P_G[bus_idx_map[bus_id]] += -val
            

    def _update_gross_generation(field_and_result: Tuple[str, str]) -> None:
        """
        Updates 'P_G' by incorporating contributions from the generation
        element given by 'field_and_result'.

        It is possible for a generation element to consume power instead of supplying.
        
        Args:
            field_and_result[0]: The name of the generation element (i.e. sgen, gen)
            field_and_result[1]: The name of the result powerflow table name (i.e. res_sgen, res_gen)
        """
        elem, res = field_and_result
        if elem not in net or len(net[elem]) == 0:
            return

        mask = _in_service_mask(elem)
        buses = net[elem]["bus"].values[mask]
        results = net[res]["p_mw"].values[mask]

        for bus_id, val in zip(buses, results):
            # Positive value of 'p_mw' is generation, negative is consumption
            if val >= 0:
                P_G[bus_idx_map[bus_id]] += val
            else:
                P_D[bus_idx_map[bus_id]] += -val

    # Check all possible generation sources for determining gross generation
    for source in _GENERATION_SOURCES:
        _update_gross_generation(source)

    # Check all possible demand sources for determining gross generation
    for source in _DEMAND_SOURCES:
        _update_gross_demand(source)

    return P_G, P_D


def run_bialek_downstream_trace(net: pandapowerNet) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of Bialek downstream-looking trace.
 
    The downstream relation is  A_d @ P = P_D, hence  P = A_d_inv @ P_D,
    where P is the gross nodal throughput (demand + outflows). The inverse
    matrix distributes each generator's output across the loads.
 
    Args:
        net: The given pandapower network to run the trace
 
    Returns:
        A_d_inv : Shape (N, N), inverse downstream distribution matrix
        P       : Shape (N,),  gross nodal throughput vector
    """
    bus_count = len(net.bus)
    if bus_count == 0:
        return np.array([]), np.array([])
 
    B = create_grid_incidence_matrix(net)
    F = extract_branch_flows(net)
 
    # Orient every branch along the actual direction of power flow.
    signs = np.sign(F)
    signs[signs == 0] = 1 # No generation is set to positive direction by default
    B_dir = B * signs     # Flips the +1, -1 start-end relation if power flow is negative,
                          # enforces start-end to align with direction of positive power flow
    F_dir = np.abs(F)     # Branch flows all positive after previous step
 
    B_send = np.where(B_dir == 1, 1.0, 0.0)    # Node is the source of the branch
    B_recv = np.where(B_dir == -1, 1.0, 0.0)   # Node is the sink of the branch
 
    # Gross nodal throughput (downstream): P_i = P_Di + outflows leaving i.
    _, P_D = gross_gen_demand(net)
    P = P_D + (B_send @ F_dir)  # B_send @ F_dir computes the outflows for each node,
                                # B_send[i] indicates whether any of the lines start at node i
 
    P_inv = np.zeros_like(P)
    mask = P > 1e-12
    P_inv[mask] = 1.0 / P[mask] # Prevent division by 0
 
    # Downstream distribution matrix:
    #   A_d[i,i] = 1
    #   A_d[i,k] = -|P_{k->i}| / P_k   for a branch flowing from k into i.
    # (B_recv @ diag(F_dir) @ B_send.T)[i,k] = branch flow on bus k->i.
    # Scaling column k by 1/P_k divides by the sending node's throughput.
    C = (B_recv * F_dir) @ B_send.T
    A_d = np.eye(bus_count) - C * P_inv  # We subtract instead of add since C contains negative values
 
    try:
        A_d_inv = np.linalg.inv(A_d)
    except np.linalg.LinAlgError:
        # Singular A_d usually signals islands / pathological topology.
        A_d_inv = np.linalg.pinv(A_d)
 
    # Inverse represents global bus-to-bus downstream powerflow contribution
    return A_d_inv, P


def calculate_source_to_load_contributions(net: pandapowerNet) -> Tuple:
    """
    Source-to-load power contributions via the downstream trace.
 
    Returns:
        E_mw                : (N, N) E[i,j] = MW flow from generator i to load j.
        prop_gen_to_load    : (N, N) fraction of generator i's output reaching load j
                              (rows sum to 1 over active loads).
        prop_load_from_gen  : (N, N) fraction of load j's demand supplied by gen i
                              (columns sum to 1 over active gens).
    """
    A_d_inv, P = run_bialek_downstream_trace(net)
    bus_count = len(net.bus)
    if bus_count == 0 or A_d_inv.size == 0:
        return np.array([]), np.array([]), np.array([])
 
    P_G, P_D = gross_gen_demand(net)
 
    P_inv = np.zeros_like(P)
    mask_P = P > 1e-12
    P_inv[mask_P] = 1.0 / P[mask_P]
 
    # E[i,j] = P_G_i * ( A_d_inv[j,i] / P_D_j ) * P_j
    #
    # P_G_i is the total generation produced at bus_i
    # A_d_inv[j, i] contains the power distribution factor from bus j to bus i
    # P_D_j / P_j is the fraction of gross power flowing through j that's consumed
    #
    # (load row j, gen column i; scaled by the load node's throughput P_j).
    PDj_over_Pj = (P_D * P_inv).reshape(-1, 1)          # [j, 1]
    E_jT = PDj_over_Pj * A_d_inv * P_G.reshape(1, -1)   # [j, i]
    E_mw = E_jT.T                                       # [i, j]
 
    P_G_inv = np.zeros_like(P_G); m = P_G > 1e-12; P_G_inv[m] = 1.0 / P_G[m]
    P_D_inv = np.zeros_like(P_D); m = P_D > 1e-12; P_D_inv[m] = 1.0 / P_D[m]
 
    # Fraction of total generation per generator for each generator to each load
    prop_gen_to_load = E_mw * P_G_inv.reshape(-1, 1)
    prop_load_from_gen = E_mw * P_D_inv.reshape(1, -1)
 
    return E_mw, prop_gen_to_load, prop_load_from_gen