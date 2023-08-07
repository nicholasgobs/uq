import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from turbopy import Simulation
def plot_outputs(output_dir : str, use_file : bool, sim : Simulation):
	"""Plots relevant outputs from a run of the rigid beam app

	:param output_dir: Directory to store the output files
	:type output_dir: str
	:param use_file: Boolean to use a file for data (.nc files) or use simulation object
	:type use_file: bool
	:param sim: Simulation object to plot the traces from
	:type sim: Simulation
	"""	
	plot_path = output_dir + '/plots/'
	outdir = Path(f"{plot_path}")
	outdir.mkdir(parents=True, exist_ok=True)

	if use_file:
		ds = xr.open_dataset(f'{output_dir}/' + 'history.nc')
		ds_fields = xr.open_dataset(f'{output_dir}/' + 'history_fields.nc')
	else:
		ds = sim.diagnostics[1]._traces
		ds_fields = sim.diagnostics[2]._traces

	run_info = sim.input_data
	
	pressure = run_info['PhysicsModules']['ThermalFluidPlasma']['initial_conditions']['N2(X1)']['pressure']
	#current = run_info['PhysicsModules']['RigidBeamCurrentSource']['peak_current']
	#current = run_info['PhysicsModules']['RigidBeamCurrentSource']['set_energy_peak']
	current = '(exp_traces)'
	#beam_radius = run_info['PhysicsModules']['RigidBeamCurrentSource']['beam_radius']

	#tmax = np.max(np.ceil(ds['time'][-1].values*1e9))
	tmax = run_info['Clock']['end_time'] * 1e9

	plt_ind = np.where(ds['time'][1:]==0)[0][0] + 1
	ds       ['time'][plt_ind] = tmax * 1e-9
	ds_fields['time'][plt_ind] = tmax * 1e-9

	f = plt.figure()
	for i, name in enumerate(ds_fields['densities'][0, :, 0]):
		#print(f'{i} {name.species_id.values}')
		if name.species_id.values == 'N2(X1)':
			n2_x1_i = i
			continue
		plt.plot(ds_fields['time'][:plt_ind]*1e9, ds_fields['densities'][:plt_ind, i, 0],
           label=name.species_id.values)
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.ylabel('Species Density [m^{-3}]', fontweight='bold')
	plt.grid()
	plt.legend(bbox_to_anchor = (1.01, 1.01))
	file_name = str(plot_path + 'densities_zoom.pdf')
	f.savefig(file_name, bbox_inches='tight')

	plt.plot(ds_fields['time'][:plt_ind]*1e9, ds_fields['densities'][:plt_ind, n2_x1_i, 0],
           label='N2(X1)')
	plt.legend(bbox_to_anchor = (1.01, 1.01))
	file_name = str(plot_path + 'densities.pdf')
	f.savefig(file_name, bbox_inches='tight') 
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:Fields:J_plasma'][:plt_ind]/1000,
			 label='Plasma', linewidth=3)
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:CurrentSource:J'][:plt_ind]/1000,
			 label='Beam', linewidth=3)
	plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:SyntheticNetCurrent_J'][:plt_ind]/1000,
			 label='Net', linewidth=3)
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.xlim([0, tmax])
	plt.ylabel('Current [kA]', fontweight='bold')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	plt.legend()
	plt.grid()
	file_name = str(plot_path + 'currents.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds['line_density'][:plt_ind], linewidth=3)
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.xlim([0, tmax])
	plt.ylabel('Line-Integrated Electron Density [cm$^{-2}$]', fontweight='bold')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	plt.grid()
	file_name = str(plot_path + 'line_density.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind] * 1e9, ds_fields['densities'].sel(species_id='e')[:,0][:plt_ind]/1e6, linewidth=3)
	plt.ylabel(r'$n_e$ [cm$^{-3}$] on axis', fontweight='bold')
	plt.grid()
	plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	file_name = str(plot_path + 'electron_density.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:E'][:,0][:plt_ind]*1e-5, linewidth=3)
	plt.ylabel(r'E [kV/cm] on axis', fontweight='bold')
	plt.grid()
	plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	file_name = str(plot_path + 'efield.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:electron_energy'][:,0][:plt_ind], linewidth=3)
	plt.ylabel(r'E$_e$ [eV] on axis', fontweight='bold')
	plt.grid()
	plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	file_name = str(plot_path + 'electron_energy.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:electron_energy'][:,0][:plt_ind]*2/3, linewidth=3)
	plt.ylabel(r'T$_e$ [eV] on axis', fontweight='bold')
	plt.grid()
	plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	file_name = str(plot_path + 'electron_temperature.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	ds_fields['Fields:nu_momentum'].plot.pcolormesh(x='r', y='time')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	plt.xlabel('r [m]', fontweight='bold')
	file_name = str(plot_path + 'nu_momentum_field.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	ds_fields['Fields:nu_ei'].plot()
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	file_name = str(plot_path + 'nu_ei.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:J_plasma'][:,0][:plt_ind]/1e4, linewidth=3, color='blue', linestyle='solid')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	plt.xlim([0, tmax])
	plt.grid()
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.ylabel('J$_p$ [A/cm$^2$]', fontweight='bold')
	file_name = str(plot_path + 'plasma_current.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds_fields['time']*1e9, ds_fields['beam_rate'][:,0]/1e6/1e9, label='Beam Rate', linewidth=3)
	plt.plot(ds_fields['time']*1e9, ds_fields['fluid_ionization'][:,0]/1e6/1e9, label='Thermal Ionization', linewidth=3)
	plt.plot(ds_fields['time']*1e9, ds_fields['recombination_rate'][:,0]/1e6/1e9, label='Recombination', linewidth=3, linestyle='--')
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.xlim([0, tmax])
	plt.ylabel('$\partial n_e$/$\partial t$ [cm$^{-3}$ / ns]', fontweight='bold')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	plt.grid()
	plt.legend()
	file_name = str(plot_path + 'electron_density_rates.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, ds_fields['Fields:nu_ee'][:,0][:plt_ind], label='e-e', linewidth=3)
	plt.plot(ds['time'][:plt_ind]*1e9, ds_fields['Fields:nu_ei'][:,0][:plt_ind], label='e-i', linewidth=3)
	plt.plot(ds['time'][:plt_ind]*1e9, ds_fields['Fields:nu_momentum'][:,0][:plt_ind], label='Momentum transfer', linewidth=3)
	plt.grid()
	plt.legend()
	plt.xlim([0,tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.ylabel('Freqency [Hz]', fontweight='bold')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	file_name = str(plot_path + 'collision_frequencies.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_ei'][:,0][:plt_ind], linewidth=3, color='blue', label=r'$\nu_{\rm ei}$')
	plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_ee'][:,0][:plt_ind], linewidth=3, color='cyan', label=r'$\nu_{\rm ee}$')
	plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_momentum'][:,0][:plt_ind], linewidth=3, color='red', label=r'$\nu_{\rm m}$')
	plt.grid()
	plt.xlim([0, tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.ylabel('On Axis Frequencies [GHz]', fontweight='bold')
	plt.legend()
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	file_name = str(plot_path + 'nu.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)


	f = plt.figure()
	plt.plot(ds['time'][:]*1e9, ds_fields['Fields:nu_ei'][:,0]/ds_fields['Fields:nu_momentum'][:,0], linewidth=3, color='blue')
	plt.grid()
	plt.xlim([0, tmax])
	plt.xlabel('Time [ns]', fontweight='bold')
	plt.ylabel(r'$\nu_{\rm ei}$/$\nu_{\rm m}$ on axis', fontweight='bold')
	plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
	file_name = str(plot_path + 'nu_bar.pdf')
	f.savefig(file_name, bbox_inches='tight')
	plt.close(f)
