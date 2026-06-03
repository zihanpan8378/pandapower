import os
import pytest
import pandapower as pp
import pandapower.networks as nw

from pandapower import pandapowerNet
from pandapower.converter.matpower import from_mpc


@pytest.fixture
def custom_radial_net() -> pandapowerNet:
    """
    A simple 3-bus system.
    Gen1 (Bus 0) and Gen2 (Bus 1) both feed Load (Bus 2).
    """
    net = pp.create_empty_network()
    
    b0 = pp.create_bus(net, vn_kv=110) # Gen 1 bus
    b1 = pp.create_bus(net, vn_kv=110) # Gen 2 bus
    b2 = pp.create_bus(net, vn_kv=110) # Load bus
    
    pp.create_ext_grid(net, b0, vm_pu=1.0) # Slack
    pp.create_sgen(net, b1, p_mw=50, q_mvar=0) # Gen 2
    pp.create_load(net, b2, p_mw=150, q_mvar=0) # Load
    
    # Connect both generators to the load
    pp.create_line(net, b0, b2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
    pp.create_line(net, b1, b2, length_km=10, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")
    
    return net

@pytest.fixture
def ieee_case14() -> pandapowerNet:
    return nw.case14() # type: ignore

@pytest.fixture
def ieee_case39() -> pandapowerNet:
    return nw.case39() # type: ignore

# @pytest.fixture
# def central_illinois_200() -> pandapowerNet:
#     grid_m = os.path.join('central-illinois-200', 'case_ACTIVSg200.m')
#     net = from_mpc(grid_m) 
#     return net

@pytest.fixture
def ieee_case300() -> pandapowerNet:
    return nw.case300() # type: ignore

@pytest.fixture
def net_fixture(request):
    """
    Parameterized fixture using for retriveing other grid fixtures. 
    """
    return request.getfixturevalue(request.param)

