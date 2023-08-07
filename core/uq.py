import lxcat_data_parser as ldp
import numpy as np
import xarray as xr
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plasma_chemistry import generate_rates
from uq_plot import *
import turbopy
import rigidbeam

# def generate_random_xsec_set(input_file, output_file):
# 	xsec_dataset = ldp.CrossSectionSet(input_file)
# 	sigma = np.array([5e-21, 1e-23, 1e-22, 1e-21])
# 	for i, s in enumerate(sigma):
# 		rand_xsec = np.random.normal(0, s)
# 		xsec_dataset.cross_sections[i].data['cross section'] = xsec_dataset.cross_sections[i].data['cross section'] + rand_xsec
# 	xsec_dataset.write(output_file)

def generate_random_xsec_set2(input_file, output_file):
	## this randomization stuff should be organized into a class or package
	# purpose of this is to test the ability to just add a flat percent +/- error
	# to the cross sections
	xsec_dataset = ldp.CrossSectionSet(input_file)
	# flat_error = 0.002
	flat_error = 0
	sigma = np.array([flat_error, flat_error, flat_error, flat_error])
	## sigma written this way because eventually this information needs to come from somewhere i.e. not hardcoded
	for i, s in enumerate(sigma):
		rand_percentage = np.random.normal(0, s)
		xsec_dataset.cross_sections[i].data['cross section'] = xsec_dataset.cross_sections[i].data['cross section'] * (1 + rand_percentage)
	xsec_dataset.write(output_file)

# def write_rate_file():
# 	rate_cm3_s = np.concatenate([np.array([EN]), np.array([energy]), np.array(eq_rates) * 1e6]).T
# 	np.savetxt(
# 	    'rates_cm3_s_bolos.csv', 
# 	    np.flipud(rate_cm3_s), 
# 	    delimiter=', ', 
# 	    header='E/N, Energy, ' + ', '.join(pname_list))

def compare_xsection_files(xsec_file_1, xsec_file_2):
	# attempt here to make this like the histroy file compare and accept aribtary number of sets
	# xsection_datasets = [ldp.CrossSectionSet('/Users/ianrit/codes/nepc-bolos-rbapp-uq/cross_section_files/asr.txt'),
	# 					 ldp.CrossSectionSet('/Users/ianrit/codes/nepc-bolos-rbapp-uq/cross_section_files/asr-randomized.txt')]

	# # test = [x.cross_sections for x in xsection_datasets]
	# # print(test)
	# for i, reaction in enumerate(xsection_datasets[0].cross_sections):
	# 	plt.figure()
	# 	for set in enumerate(xsection_datasets):
	# 		plt.loglog(set.cross_sections[i].data['energy'], set.cross_sections[i].data['cross section'])
	# 	plt.show()

	data1 = ldp.CrossSectionSet(xsec_file_1)
	data2 = ldp.CrossSectionSet(xsec_file_2)

	for i, stuff in enumerate(data1.cross_sections):
		plt.figure()
		plt.loglog(data1.cross_sections[i].data['energy'], data1.cross_sections[i].data['cross section'], label=xsec_file_1)
		plt.loglog(data2.cross_sections[i].data['energy'], data2.cross_sections[i].data['cross section'], label=xsec_file_2)
		plt.title(stuff)
		plt.legend()
		plt.grid()
		plt.xlabel('energy [eV]')
		plt.ylabel('xsection [m^-2]')
		plt.tight_layout()
		# plt.savefig('xsec_' + str(i) + '.pdf')
		plt.show()

def compare_rate_files():
	rates1 = np.loadtxt('/Users/ianrit/codes/nepc-bolos-rbapp-uq/N2_rates_simple_bolos.txt', comments='!')
	rates2 = np.loadtxt('/Users/ianrit/codes/nepc-bolos-rbapp-uq/imr-randomized-rates.txt', comments='!')

	data1 = ldp.CrossSectionSet('/Users/ianrit/codes/nepc-bolos-rbapp-uq/cross_section_files/asr.txt')
	rate_names = data1.cross_sections
	
	for i, stuff in enumerate(rate_names):
		plt.figure()
		plt.plot(rates1[:, 0], rates1[:, i + 1])
		plt.plot(rates2[:, 0], rates2[:, i + 1])
		plt.xlabel('energy [eV]')
		plt.ylabel('rate [m^3/s]')
		plt.title(stuff)
		plt.grid()
		plt.tight_layout()
		plt.savefig('rate_' + str(i) + '.pdf')

def history_file_comparison_plots():
	history_datasets = [xr.open_dataset('/Users/ianrit/simulation/uq/test/example_run/history.nc'), 
						xr.open_dataset('/Users/ianrit/simulation/uq/rand-test/example_run/history.nc')]
	label_names = ['standard', 'random']

	for data_name in [names for names in history_datasets[0].data_vars]:
		plt.figure()
		for i, history in enumerate(history_datasets):
			plt.plot(np.trim_zeros(history.time.data, 'b'), np.trim_zeros(history[data_name], 'b'), label=label_names[i])
		plt.xlabel(history['time'].name + ' [' + history['time'].units + ']')
		plt.ylabel(history[data_name].name + ' [' + history[data_name].units + ']')
		plt.legend()
		plt.grid()
		plt.tight_layout()
		plt.savefig(data_name + '.pdf')
		plt.close()

def run_rigid_beam_problem():
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
	                     #'RateFile': '../rate_files/random_rates/rates.txt',
	                     'RateFile': '../rate_files/phelps_rates/rates.txt',
						 'file_format': 'matrix'},
	#  'CurrentSourceFromFile':  {'amplitude_prefactor': 10,
	#                            'beam_energy': './exp_traces/beam_energy.csv',
	#                            'current_profile': './exp_traces/S5238RadProfRCNorm.txt',
	#                            'current_pulse': './exp_traces/beam_current.csv',
	#                            'is_gamma_beta': False},
	 'RigidBeamCurrentSource': {'amplitude': 400e3,
	                           'radius': 0.04,
	                           'rise_time': 50e-09,
	                           'set_energy_peak': 1e4,
	                           'profile': 'gaussian'},
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
	 'directory': 'random_run/',
	 'histories': [{'filename': 'history.nc',
	                'interval': 0.5e-9,
	                'traces': [{'name': 'synthetic_integral:Fields:J_plasma',
	                            'long_name': 'Plasma Current',
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
	                'traces': [{'name': 'fluid_ionization',
	                            'long_name': 'Thermal Ionization Rate',
	                            'coords': ['grid'],
	                            'units': 's^{-1}'},
	                           {'name': 'beam_rate',
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
	                            {'name': 'RadialProfile:f',
	                            'long_name': 'Profile',
	                            'coords': ['grid']},
	                            {'name': 'RadialProfile:dfdr',
	                            'long_name': 'Derivative',
	                            'coords': ['grid']}
	                        ]}],
	                },
	}

	# rcParams['font.family'] = 'serif'
	# rcParams['font.serif'] = ['Times New Roman']
	# rcParams['font.size'] = 12
	# rcParams['font.weight'] = 400

	# rcParams['mathtext.rm'] = 'serif'
	# rcParams['mathtext.it'] = 'serif:italic'
	# rcParams['mathtext.bf'] = 'serif:bold'
	# rcParams['mathtext.fontset'] = 'custom'

	sim = turbopy.Simulation(problem_config)
	sim.run()
	

	ds = sim.diagnostics[1]._traces
	ds_fields = sim.diagnostics[2]._traces

	# # diagnose
	# # ds
	# # ds_fields

	#print(ds['time'])
	#ds['time'][-1] = np.nan
	plt_ind = np.where(ds['time'][1:]==0)[0][0] + 1
	print(plt_ind)

	chem = xr.open_dataset('./random_run/chemistry.nc')

	plt.figure(figsize=(10,8))
	for item in chem:
		chem[item].plot(label=item)
	plt.yscale('log')
	plt.xscale('log')
	#plt.legend(bbox_to_anchor=(1.04,1.01), loc="lower right")
	plt.legend(loc="best")
	
	plt.grid()
	plt.savefig("random_run/chemistry.pdf", bbox_inches='tight')

	plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:Fields:J_plasma'][:plt_ind]/1000, label='Plasma')
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:CurrentSource:J'][:plt_ind]/1000, label='Beam Current')
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:SyntheticNetCurrent_J'][:plt_ind]/1000, label='Net')
	plt.xlabel('Time [ns]')
	plt.ylabel('Current [kA]')
	plt.title('Pressure: 1 Torr')
	plt.legend()
	plt.grid()
	plt.legend(['Plasma', 'Beam', 'Net'])
	plt.xlim([0, 100])
	plt.tight_layout()
	plt.savefig('random_run/currents.pdf')

	plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds['line_density'][:plt_ind])
	plt.xlabel('Time [ns]')
	plt.ylabel(r'Line-integrated electron density [#/cm$^2$]')
	plt.title(f'Pressure: 1 Torr')
	plt.grid()
	plt.xlim([0, 100])
	# plt.ylim([0, 7e18])
	plt.savefig("random_run/line_density.pdf")

	plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds_fields['Fields:E'][:plt_ind, 0]/1000, label='Beam Energy')
	plt.xlabel('Time [ns]')
	plt.ylabel('E-Field [kV/cm]')
	plt.title('Pressure: 1 Torr')
	plt.legend()
	plt.grid()
	plt.xlim([0, 100])
	plt.tight_layout()
	plt.savefig('random_run/e-field.pdf')

	plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds_fields['CurrentSource:beam_energy'][:plt_ind, 0]/1000, label='Beam Energy')
	plt.xlabel('Time [ns]')
	plt.ylabel('Energy [keV]')
	plt.title('Pressure: 1 Torr')
	plt.legend()
	plt.grid()
	plt.xlim([0, 100])
	plt.tight_layout()
	plt.savefig('random_run/beam_energy.pdf')

	f = plt.figure()
	plt.plot(ds_fields['densities'].sel(species_id='N2(X2:ion)')[:,0]/1e6, linewidth=3)
	plt.ylabel(r'$n_e$ [cm$^{-3}$] on axis', fontweight='bold')
	plt.grid()
	# plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	# file_name = str(simulation_path / 'electron_density.jpg')
	f.savefig('random_run/ion_density.pdf', bbox_inches='tight')
	plt.show()
	plt.close(f)

	f = plt.figure()
	plt.plot(ds_fields['Fields:electron_energy'][:,0], linewidth=3)
	plt.ylabel(r'E$_e$ [eV] on axis', fontweight='bold')
	plt.grid()
	# plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	# file_name = str(simulation_path / 'electron_energy.jpg')
	f.savefig('random_run/electron_energy', bbox_inches='tight')
	plt.show()
	plt.close(f)


def log_results():
	ds = xr.open_dataset('random_run/history.nc')
	ds_fields = xr.open_dataset('random_run/history_fields.nc')
	with open('line_density.txt','a+') as file:
		file.write(str(np.max(ds['line_density'].values)) + '\n')
	with open('net_current.txt','a+') as file:
		file.write(str(np.max(ds['synthetic_integral:SyntheticNetCurrent_J'].values)) + '\n')
	with open('electron_energy.txt','a+') as file:
		file.write(str(np.max(ds_fields['Fields:electron_energy'].values[:,0])) + '\n')
	with open('ion_density.txt','a+') as file:
		file.write(str(ds_fields['densities'].sel(species_id='N2(X2:ion)').values[-2,0]) + '\n')


def main():
	histogram = False

	if histogram:
		for x in range(5):
			generate_random_xsec_set2('../cross_section_files/asr.txt', '../cross_section_files/random.txt')
			# history_file_comparison_plots()
			# compare_xsection_files('/Users/ianrit/codes/uq/cross_section_files/asr.txt', 
			# 	'/Users/ianrit/codes/uq/cross_section_files/random.txt')
			generate_rates.run('random')
			# compare_rate_files()
			run_rigid_beam_problem()
			log_results()

		output_files = ['line_density', 'net_current', 'electron_energy', 'ion_density']
		output_descriptions = [r'integrated line density [m$^{-2}$]', 'net current [A]', 'electron energy [eV]', r'ion density [m^{-3}]']
		plot_results(output_files, output_descriptions)
		plot_normal_distribution()
	else: 
		#generate_random_xsec_set2('../cross_section_files/asr.txt', '../cross_section_files/random.txt')
		generate_random_xsec_set2('../cross_section_files/phelps.txt', '../rate_files/phelps.txt')
		generate_rates.run("phelps") 
		run_rigid_beam_problem()



if __name__=='__main__':
	main()