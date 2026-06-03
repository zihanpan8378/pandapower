import numpy as np
import pandapower as pp
import pandas as pd
import pytest
import math

from src.run_bialek import calculate_source_to_load_contributions, gross_gen_demand
from pandapower import pandapowerNet 
from numpy.testing import assert_allclose


def test_one_line_grid() -> None:
    """
    Test source to load distribution function with a grid of the following configuration:

        Load -- Bus 1 -- Line -- Bus 2 -- Generator

    The expected result should be that generator should suppply 100% of load's demand.
    """
    net = pp.create_empty_network()
    load_demand = 10.0
    bus_voltage = 110
    bus_count = 2

    b1 = pp.create_bus(net, vn_kv=bus_voltage, name="Node 1")
    b2 = pp.create_bus(net, vn_kv=bus_voltage, name="Node 2")
    pp.create_line(net, from_bus=b1, to_bus=b2, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", length_km=1.0)

    pp.create_gen(net, bus=b1, p_mw=load_demand, vm_pu=1.0, slack=True, name="Power Source", max_p_mw=10.0)
    pp.create_load(net, bus=b2, p_mw=load_demand, q_mvar=2.0, name="Data Center Load")

    pp.runpp(net)
    E_mw, prop_gen_to_load, prop_load_from_gen = calculate_source_to_load_contributions(net=net)
    tolerance = 1e-3
    
    # Generator connected to bus 1 should produce load demand on bus 2
    assert E_mw.shape[0] == bus_count
    assert E_mw.shape[1] == bus_count
    assert math.isclose(E_mw[0, 1], load_demand, abs_tol=tolerance)
    

def test_perfect_mixer_proportions(custom_radial_net: pandapowerNet) -> None:
    """
    Test load proportionality calculation using the network defined in the 'custom_radial_net' fixture.
    """
    net = custom_radial_net
    pp.rundcpp(net)
    load_allocations, _, _ = calculate_source_to_load_contributions(net) 
    load_to_bus_2 = load_allocations[:, 2]
        
    # 1. Verify Global Balance: Allocations must sum to the actual P_load
    actual_load = net.res_load.p_mw.at[0]
    assert sum(load_to_bus_2) == pytest.approx(actual_load, rel=1e-4)
    
    # 2. Verify Proportionality
    # Because lines have minor losses, Gen 1 won't be exactly 100MW. 
    # We must calculate the ratio based on the actual arrival power at Bus 2.
    flow_from_gen1 = net.res_line.p_to_mw.at[0] # Flow arriving at Bus 2 from Bus 0 # type: ignore
    flow_from_gen2 = net.res_line.p_to_mw.at[1] # Flow arriving at Bus 2 from Bus 1 # type: ignore
    total_inflow = flow_from_gen1 + flow_from_gen2  # type: ignore
    
    expected_gen1_ratio = flow_from_gen1 / total_inflow   # type: ignore
    expected_gen1_mw = expected_gen1_ratio * actual_load  # type: ignore
    
    gen_0_to_load = load_allocations[0, 2]
    assert gen_0_to_load == pytest.approx(expected_gen1_mw, rel=1e-4)


@pytest.mark.parametrize(
    "net_fixture",
    ["ieee_case14", "ieee_case39", "ieee_case300"], # "central_illinois_200"],
    indirect=True
)
def test_general_case(net_fixture: pandapowerNet) -> None:
    """General test verifying load proportionality on larger grids."""
    net = net_fixture
    pp.rundcpp(net)
    load_allocations, _, _ = calculate_source_to_load_contributions(net)

    # Reference demand must use the SAME gross definition as the trace
    # (loads + shunts + wards + motors), not res_load alone, or the column
    # sums diverge wherever non-load sinks exist (e.g. case300's 29 shunts).
    _, actual_demand = gross_gen_demand(net)

    calculated_demand = np.sum(load_allocations, axis=0)

    assert_allclose(
        calculated_demand,
        actual_demand,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Per-bus load allocation does not match gross demand.",
    )
