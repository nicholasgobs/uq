"""Tests the running of individual cross section files on their own
"""
from rba_plot import *
from rigid_beam_all_outputs import *
from modify_cross_sections import *

names = ["uq_pitchford", "uq_phelps", "uq_itakawa",  "uq_kawaguchi"]

for name in names:
    print(name)
    output_dir = f"../{name}_test"
    xsec_list = read_xsec(name)
    find_and_write_rates("phelps", xsec_list)
    sim = single_run(output_dir, "phelps_rates", run=True)
    plot_outputs(output_dir, use_file=False, sim=sim)
