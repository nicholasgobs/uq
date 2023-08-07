import lxcat_data_parser as ldp
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from bolos import parse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plasma_chemistry import generate_rates, constructors, maxwellian, factory
from uq_plot import *
import turbopy
import rigidbeam.chemistry as ch
import copy


def configure_sim(rates_dir: str, output_dir: str):
	"""Returns the input dictionary for a rigid beam simulation with specified directories

	:param rates_dir: Directory where rate file will be found
	:type rates_dir: str
	:param output_dir: Directory to store outputs
	:type output_dir: str
	:return: Input dictionary for rigid beam simulation
	:rtype: dict
	"""	
	problem_config = {
	'Tools':
	{'FiniteDifference': {}},
	'Grid':
	{'N': 100, 'max': 0.0865, 'min': 0},
	'Clock':
	{'dt': 5e-11, 'start_time': 0e-09, 'end_time': 100e-09, 'print_time': False},
	'PhysicsModules':
	{
	 'PlasmaChemistry': {'ConvertFromCm': False,
	                     'RateFile': f'../rate_files/{rates_dir}/rates.txt',
						 'file_format': 'matrix'},
	  'CurrentSourceFromFile':  {'amplitude_prefactor': 10,
	                            'beam_energy': './exp_traces/beam_energy.csv',
				    			#'beam_energy': './exp_traces/beam_energy-20ns.csv',
	                            'current_profile': './exp_traces/S5238RadProfRCNorm.txt',
	                            'current_pulse': './exp_traces/beam_current.csv',
				    			#'current_pulse': './exp_traces/beam_current-20ns.csv',
	                            'is_gamma_beta': False},
	 #'RigidBeamCurrentSource': {'amplitude': 400e3,
	 #                          'radius': 0.04,
	 #                          'rise_time': 50e-09,
	 #                          'set_energy_peak': 1e4,
	 #                          'profile': 'gaussian'},
	 'ThermalFluidPlasma': {'initial_conditions': {'N2(X1)': {'energy': 0.05,
	                                                          'pressure': 1.0,
	                                                          'velocity': 0.0},
	                                               'N2(X2:ion)': {'energy': 0.05,
	                                                              'pressure': 1e-10,
	                                                              'velocity': 0.0},
	                                               'e': {'energy': 0.05,
	                                                     'pressure': 1e-10,
	                                                     'velocity': 0.0}},
	                        'use_ground_state_rates': False},
	 'FluidElectrons': {},
	 'ElectronEnergyDynamicEquation': {},
	 'BeamImpactRateWithStoppingPower': {'atomic_number': 14,
	                                     'atomic_weight': 28,
	                                     'energy_loss_per_pair': 34},                        
	 'FluidConductivity': {},
	 'FieldUpdateInversionModel': {},

	 'SyntheticNetCurrent': {},
	 'SyntheticIntegralProbe': {'fields': ['Fields:J_plasma',
	                                       'CurrentSource:J',
	                                       'SyntheticNetCurrent_J']},
	 'SyntheticInterferometer': {},
	 'SyntheticDensitiesProbe': {'interval': 0.5e-9},
	 'MomentumTransferFrequencies': {},
	},
	'ComputeTools': {'FiniteDifference': {}},
	'Diagnostics':
	{'PlasmaChemistryDiagnostic': {'filename': 'chemistry.nc'},
	 'directory': f'{output_dir}/',
	 'histories': [{'filename': 'history.nc',
	                'interval': 0.5e-9,
	                'traces': [{'name': 'synthetic_integral:Fields:J_plasma',
	                            'long_name': 'Plasma Current', ######
	                            'coords': ['dim_0'],
	                            'units': 'A'},
	                           {'name': 'synthetic_integral:CurrentSource:J',
	                            'long_name': 'Beam Current',
	                            'coords': ['dim_0'],
	                            'units': 'A'},
	                            {'name': 'synthetic_integral:SyntheticNetCurrent_J',
	                            'long_name': 'Net Current',
	                            'coords': ['dim_0'],
	                            'units': 'A'},
	                           {'name': 'line_density'},
	                           {'name': 'SyntheticNetCurrent_J'}]},
	               {'filename': 'history_fields.nc',
	                'interval': 0.5e-9,
	                'traces': [{'name': 'fluid_ionization', ########
	                            'long_name': 'Thermal Ionization Rate',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'beam_rate', #######
	                            'long_name': 'Beam Impact Ionization Rate',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'Fields:E',
	                           'long_name': 'Electric Field',
	                           'coords': ['grid'],
	                            'units': 'V*m^{-1}'},
	                           {'name': 'species_densities'},
	                           {'name': 'Fields:electron_energy',
	                           'long_name': 'Mean Electron Energy',
	                            'coords': ['grid'],
	                            'units': 'eV'},
	                           {'name': 'Fields:nu_momentum',
	                            'long_name': 'Momentum Transfer Frequency',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'Fields:J_plasma',
	                           'long_name': 'Plasma Current Density',
	                           'coords': ['grid'],
	                            'units': 'A*m^{-2}'},
	                           {'name': 'SyntheticNetCurrent_J'},
	                           {'name': 'Fields:sigma',
	                            'long_name': 'Conductivity',
	                            'coords': ['grid'],
	                            'units': 'A^{2}*s^{3}*kg^{-1}*m{-3}'},
	                           {'name': 'Fields:nu_ee',
	                            'long_name': 'Electron-Electron Collision Frequency',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'Fields:nu_ei',
	                            'long_name': 'Electron-Ion Collision Frequency',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'Fields:coulomb_logarithm',
	                            'long_name': 'Coulomb Logarithm',
	                            'coords': ['grid'],
	                            'units': 'unitless'},
	                            {'name': 'CurrentSource:beam_energy',
	                            'long_name': 'Electron Beam Mean Energy',
	                            'coords': ['grid'],
	                            'units': 'eV'},
	                            #{'name': 'RadialProfile:f',
	                            #'long_name': 'Profile',
	                            #'coords': ['grid']},
	                            #{'name': 'RadialProfile:dfdr',
	                            #'long_name': 'Derivative',
	                            #'coords': ['grid']},
								{'name':'recombination_rate',
	                            "long_name": "Thermal Dissociative Recombination Rate",
				     			'coords': ['grid'],
								"units": "s^{-1}"}
	                        ]}],
	                },
	}
	return problem_config


def log_all_outputs(output_dir: str, use_file: bool, sim: turbopy.Simulation):
	"""Records quantities of interest from simulation or from .nc file into text files (append mode)

	:param output_dir: Directory to store outputs
	:type output_dir: str
	:param use_file: Boolen to read from file in output_dir or use simulation object
	:type use_file: bool
	:param sim: Simulation to record QOI from
	:type sim: turbopy.Simulation
	"""	
	
	orig_output_dir = copy.deepcopy(output_dir)
	
	#Read from .nc files or from simulation object
	if (use_file):
		ds = xr.open_dataset(f'{output_dir}/' + 'history.nc')
		ds_fields = xr.open_dataset(f'{output_dir}/' + 'history_fields.nc')
	else:
		ds = sim.diagnostics[1]._traces
		ds_fields = sim.diagnostics[2]._traces

	#outdir = Path(f"{output_dir}/")
	#outdir.mkdir(parents=True, exist_ok=True)
	
	#1: All peak (on-axis) densities
	desnities_outdir = Path(f"{output_dir}/peak_densities/")
	desnities_outdir.mkdir(parents=True, exist_ok=True)
	for i, name in enumerate(ds_fields['densities'][0, :, 0]):
		#print(f'{i} {name.species_id.values}')
		file_name = str(name.species_id.values)
		value = np.max(ds_fields['densities'].values[:, i, 0])
		with open(f'{output_dir}/peak_densities/' + f'{file_name}' + '.txt','a+') as file:
			file.write(str(value)+ '\n')

	output_dir = f"{output_dir}/QOI"
	QOI_outdir = Path(output_dir)
	QOI_outdir.mkdir(parents=True, exist_ok=True)

	#1.1 Peak on-axis ion density
	with open(f'{output_dir}/' + 'ion_density.txt','a+') as file:
		file.write(str(ds_fields['densities'].sel(species_id='N2(X2:ion)').values[-2,0]) + '\n')
	
	#2: Peak net current ~~~~~~~~
	with open(f'{output_dir}/' + 'net_current.txt','a+') as file:
		file.write(str(np.max(ds['synthetic_integral:SyntheticNetCurrent_J'].values)) + '\n')

	#3.1 Min JPlasma
	with open(f'{output_dir}/' + 'min_JPlasma.txt','a+') as file:
		min_Jplasma = np.min(ds_fields['Fields:J_plasma'].values[:, 0])
		file.write(str(min_Jplasma)+ '\n')

	#3.2 Max JPlasma
	with open(f'{output_dir}/' + 'max_JPlasma.txt','a+') as file:
		max_Jplasma = np.max(ds_fields['Fields:J_plasma'].values[:, 0])
		file.write(str(max_Jplasma)+ '\n')

	#4: Line integrated electron density ~~~~~~~~
	with open(f'{output_dir}/' + 'line_density.txt','a+') as file:
		file.write(str(np.max(ds['line_density'].values)) + '\n')
	
	#6: Peak E-field magnitude
	with open(f'{output_dir}/' + 'peak_E-field_mag.txt','a+') as file:
		peak_E_field_mag = np.max(np.absolute(ds_fields['Fields:E'].values[:, 0]))
		file.write(str(peak_E_field_mag)+ '\n')

	#7: Electron energy ~~~~~~~~~
	with open(f'{output_dir}/' + 'electron_energy.txt','a+') as file:
		peak_electron_energy = np.max(ds_fields['Fields:electron_energy'].values[:,0])
		file.write(str(peak_electron_energy) + '\n')

	#8: Peak nu_momentum
	with open(f'{output_dir}/' + 'nu_momentum.txt','a+') as file:
		peak_nu_momentum = np.max(ds_fields['Fields:nu_momentum'].values[:,0])
		file.write(str(peak_nu_momentum) + '\n')


	#9: Peak nu_ei
	with open(f'{output_dir}/' + 'nu_ei.txt','a+') as file:
		peak_nu_ei = np.max(ds_fields['Fields:nu_ei'].values[:,0])
		file.write(str(peak_nu_ei) + '\n')

	#10: Peak nu_ee
	with open(f'{output_dir}/' + 'nu_ee.txt','a+') as file:
		peak_nu_ee = np.max(ds_fields['Fields:nu_ee'].values[:,0])
		file.write(str(peak_nu_ee) + '\n')

	#Time values
	times = ds_fields['time']
	times_outdir = Path(f'{orig_output_dir}/QOI_times')
	times_outdir.mkdir(parents=True, exist_ok=True)
	
	#Max JPlasma
	with open(f'{orig_output_dir}/QOI_times/max_JPlasma_time.txt', 'a+') as file:
		vals = ds_fields['Fields:J_plasma'].values[:, 0]
		max_Jplasma_time_idx = np.where(vals == max_Jplasma)
		max_Jplasma_time = times[max_Jplasma_time_idx].values[0]
		file.write(str(max_Jplasma_time) + "\n")

	#Min JPlasma
	with open(f'{orig_output_dir}/QOI_times/min_JPlasma_time.txt', 'a+') as file:
		vals = ds_fields['Fields:J_plasma'].values[:, 0]
		min_Jplasma_time_idx = np.where(vals == min_Jplasma)
		min_Jplasma_time = times[min_Jplasma_time_idx].values[0]
		file.write(str(min_Jplasma_time) + "\n")

	#Peak E-field
	with open(f'{orig_output_dir}/QOI_times/peak_E-field_mag_time.txt', 'a+') as file:
		vals = np.absolute(ds_fields['Fields:E'].values[:, 0])
		E_mag_time_idx = np.where(vals == peak_E_field_mag)
		E_mag_time = times[E_mag_time_idx].values[0]
		file.write(str(E_mag_time) + "\n")

	#Peak electron energy
	with open(f'{orig_output_dir}/QOI_times/peak_electron_energy_time.txt', 'a+') as file:
		vals = ds_fields['Fields:electron_energy'].values[:, 0]
		peak_e_energy_time_idx = np.where(vals == peak_electron_energy)
		e_energy_time = times[peak_e_energy_time_idx].values[0]
		file.write(str(e_energy_time) + "\n")

	#Nu_momentum
	with open(f'{orig_output_dir}/QOI_times/peak_nu_momentum_time.txt', 'a+') as file:
		vals = ds_fields['Fields:nu_momentum'].values[:, 0]
		nu_peak_time_idx = np.where(vals == peak_nu_momentum)
		nu_peak_time = times[nu_peak_time_idx].values[0]
		file.write(str(nu_peak_time) + "\n")

	#Nu_ei
	with open(f'{orig_output_dir}/QOI_times/peak_nu_ei_time.txt', 'a+') as file:
		vals = ds_fields['Fields:nu_ei'].values[:, 0]
		nu_peak_time_idx = np.where(vals == peak_nu_ei)
		nu_peak_time = times[nu_peak_time_idx].values[0]
		file.write(str(nu_peak_time) + "\n")

	#Nu_ee
	with open(f'{orig_output_dir}/QOI_times/peak_nu_ee_time.txt', 'a+') as file:
		vals = ds_fields['Fields:nu_ee'].values[:, 0]
		nu_peak_time_idx = np.where(vals == peak_nu_ee)
		nu_peak_time = times[nu_peak_time_idx].values[0]
		file.write(str(nu_peak_time) + "\n")
	
	#Min Plasma Current
	with open(f'{output_dir}/' + 'min_Plasma_current.txt','a+') as file:
		min_plasma_current = np.min(ds['synthetic_integral:Fields:J_plasma'].values)
		file.write(str(min_plasma_current)+ '\n')

	#Max Plasma Current
	with open(f'{output_dir}/' + 'max_Plasma_current.txt','a+') as file:
		max_plasma_current = np.max(ds['synthetic_integral:Fields:J_plasma'].values)
		file.write(str(max_plasma_current)+ '\n')

	#Peak fluid ionization rate
	with open(f'{output_dir}/' + 'fluid_ionization.txt','a+') as file:
		peak_fluid_ionization = np.max(ds_fields['fluid_ionization'].values[:,0])
		file.write(str(peak_fluid_ionization) + '\n')
	
	#Peak beam rate
	with open(f'{output_dir}/' + 'beam_rate.txt','a+') as file:
		peak_beam_rate = np.max(ds_fields['beam_rate'].values[:,0])
		file.write(str(peak_beam_rate) + '\n')
	
	
def single_run(output_dir: str, rates_dir: str, run: bool = True):
	"""Configures and runs the rigid beam app once

	:param output_dir: Directory to store outputs
	:type output_dir: str
	:param rates_dir: Directory to store rate file
	:type rates_dir: str
	:param run: Whether or not to run the simulation once it is configured, defaults to True
	:type run: bool, optional
	:return: Simulation object
	:rtype: turbopy.Simulation
	"""	
	problem_config = configure_sim(rates_dir, output_dir)
	sim = turbopy.Simulation(problem_config)
	if run:
		sim.run()
	return sim