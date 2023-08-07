from rigid_beam_all_outputs import *
from modify_cross_sections import *
from rba_plot import *
import copy


def run_removal():
    """Run the loop to run simulation with base case, then remove one cross section at a time
    """    
    outpath = Path("../removal_analysis")
    outpath.mkdir(parents=True, exist_ok=True)
    orig_xsec_list = read_xsec("phelps")
    for i, xsec in enumerate(orig_xsec_list):
        #if i != 23:
        #     continue
		
        #Run the base case with all cross sections included 
        if i == 0:
            print("Base case: ")
            find_and_write_rates("phelps", orig_xsec_list)
            output_dir = f"../removal_analysis/Base"
            outpath = Path(f"{output_dir}")
            outpath.mkdir(parents=True, exist_ok=True)
            sim = single_run(output_dir, "phelps_rates", True)
            log_all_outputs(output_dir, use_file=False, sim=sim)
            plot_outputs(output_dir, use_file = False, sim=sim)
            continue
        
		#Perform a deep copy of original cross section set as not to retroactively modify it
        temp_list = copy.deepcopy(orig_xsec_list)
        name = xsec['product']
        print("\n\nRemoving: " + name + "\n\n")
        
		#For ionization, instead of removing completely, fill with data that creates 0 rate coefficient at all energies to ensure compatibility with rigid beam app
        if i != 23:
            temp_list.remove(xsec)
        else:
            print(xsec['product'])
            temp_list[i]['data'] = [[1e11, 0], [1.1e11, 0]]
            temp_list[i]['threshold'] = 1e10
            print(temp_list[i]['data'])

		#Write the rates and run the simulation with custon cross section set, log QOIs
        find_and_write_rates("phelps", temp_list)
        output_dir = f"../removal_analysis/{name}"
        outpath = Path(f"{output_dir}")
        outpath.mkdir(parents=True, exist_ok=True)
        sim = single_run(output_dir, "phelps_rates", True)
        log_all_outputs(output_dir, use_file=False, sim=sim)
        plot_outputs(output_dir, use_file = False, sim=sim)
       # except:
        #    print("\n\nFailed to remove: " + name + "\n\n")
            
def plot_removal_data():
    """Plot percent change in QOI compared to base case for each cross section removed, ionization is on log scale due to order of magnitude difference
    """    
    #Get data from base case
    base = {}
    for base_qoi in Path("../removal_analysis/Base/QOI").glob("*"):
        name = str(base_qoi).replace("../removal_analysis/Base/QOI/", "").replace(".txt","").replace("_"," ")
        #print(name)
        num = np.loadtxt(f"{base_qoi}", max_rows=1)
        base[name] = num

    #Plot percent change in QOI for each cross section removed    
    all_removed_xsecs = Path("../removal_analysis/").glob("*")
    for removed_xsec in all_removed_xsecs:
        removed_xsec_name_str = str(removed_xsec).replace("../removal_analysis/", "")
        if removed_xsec_name_str == "Base":
            continue
        qois = Path(str(removed_xsec) + "/QOI/").glob("*")
        data = {}
        for q in qois:
            name = str(q).replace(str(removed_xsec) + "/QOI/", "").replace(".txt","").replace("_"," ")
            num = np.loadtxt(f"{q}", max_rows=1)
            data[name] = num
        f = plt.figure()
        for k, v in data.items():
            percent_change = 100 * (v - base[k]) / base[k]
            plt.bar(k, percent_change)
        plt.title("Removed: " + removed_xsec_name_str)
        plt.xlabel("QOI")
        plt.ylabel("Percent Change")
        if "N2^+" in removed_xsec_name_str:
             plt.yscale("log")
        plt.xticks(rotation = 45)
        plt.grid(axis = "y")
        plt.show()
        plt.close(f)

def plot_curves(source_dir):
	"""Plot curves over time of relevant quantities corresponding to the removal of each cross section

	:param source_dir: Directory in which data is found
	:type source_dir: str
	"""	
	all_removed_xsecs = Path(source_dir).glob("*")
	for removed_xsec in all_removed_xsecs:
		removed_xsec_name_str = str(removed_xsec).replace(source_dir, "")
		output_dir = f"{source_dir}{removed_xsec_name_str}"
	
		ds = xr.open_dataset(f'{output_dir}/' + 'history.nc')
		ds_fields = xr.open_dataset(f'{output_dir}/' + 'history_fields.nc')

		run_info = configure_sim('','')
	
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
		
		plot_path = ""
		
		i = 0
		
		i += 1
		f = plt.figure(i)
		plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:Fields:J_plasma'][:plt_ind]/1000, label = removed_xsec_name_str)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.xlim([0, tmax])
		plt.ylabel('Plasma Current [kA]', fontweight='bold')
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.grid(True)
		file_name = str(plot_path + 'currents.pdf')

		
		i += 1
		f = plt.figure(i)    
		plt.plot(ds['time'][:plt_ind]*1e9, ds['synthetic_integral:SyntheticNetCurrent_J'][:plt_ind]/1000, label = removed_xsec_name_str)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.xlim([0, tmax])
		plt.ylabel('Net Current [kA]', fontweight='bold')
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.grid(True)
		file_name = str(plot_path + 'currents.pdf')


		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]*1e9, ds['line_density'][:plt_ind], label = removed_xsec_name_str)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.xlim([0, tmax])
		plt.ylabel('Line-Integrated Electron Density [cm$^{-2}$]', fontweight='bold')
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.grid(True)
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'line_density.pdf')


		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind] * 1e9, ds_fields['densities'].sel(species_id='e')[:,0][:plt_ind]/1e6, label = removed_xsec_name_str)
		plt.ylabel(r'$n_e$ [cm$^{-3}$] on axis', fontweight='bold')
		plt.grid(True)
		plt.xlim([0,tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'electron_density.pdf')


		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:E'][:,0][:plt_ind]*1e-5, label = removed_xsec_name_str)
		plt.ylabel(r'E [kV/cm] on axis', fontweight='bold')
		plt.grid(True)
		plt.xlim([0,tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'efield.pdf')


		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:electron_energy'][:,0][:plt_ind], label = removed_xsec_name_str)
		plt.ylabel(r'E$_e$ [eV] on axis', fontweight='bold')
		plt.grid(True)
		plt.xlim([0,tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'electron_energy.pdf')

		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]* 1e9, ds_fields['Fields:J_plasma'][:,0][:plt_ind]/1e4, label = removed_xsec_name_str)
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.xlim([0, tmax])
		plt.grid(True)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.ylabel('J$_p$ [A/cm$^2$]', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'plasma_current.pdf')


		i += 1
		f = plt.figure(i)  
		plt.plot(ds_fields['time']*1e9, ds_fields['beam_rate'][:,0]/1e6/1e9, label = removed_xsec_name_str)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.xlim([0, tmax])
		plt.ylabel('$\partial n_e$/$\partial t$ [cm$^{-3}$ / ns] - Beam Rate', fontweight='bold')
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.grid(True)
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'electron_density_rates.pdf')
        
		i += 1
		f = plt.figure(i)  
		plt.plot(ds_fields['time']*1e9, ds_fields['fluid_ionization'][:,0]/1e6/1e9, label = removed_xsec_name_str)
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.xlim([0, tmax])
		plt.ylabel('$\partial n_e$/$\partial t$ [cm$^{-3}$ / ns] - Thermal Ionization', fontweight='bold')
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		plt.grid(True)
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		file_name = str(plot_path + 'electron_density_rates.pdf')
                


		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_ei'][:,0][:plt_ind], label = removed_xsec_name_str)
		plt.grid(True)
		plt.xlim([0, tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.ylabel('On Axis Frequencies [GHz] : ' + r'$\nu_{\rm ei}$', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		file_name = str(plot_path + 'nu.pdf')

		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_ee'][:,0][:plt_ind], label = removed_xsec_name_str)
		plt.grid(True)
		plt.xlim([0, tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.ylabel('On Axis Frequencies [GHz] : ' + r'$\nu_{\rm ee}$', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		file_name = str(plot_path + 'nu.pdf')
                
		i += 1
		f = plt.figure(i)  

		plt.plot(ds['time'][:plt_ind]*1e9, 1e-9*ds_fields['Fields:nu_momentum'][:,0][:plt_ind], label = removed_xsec_name_str)
		plt.grid(True)
		plt.xlim([0, tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.ylabel('On Axis Frequencies [GHz] : ' + r'$\nu_{\rm m}$', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		file_name = str(plot_path + 'nu.pdf')

		i += 1
		f = plt.figure(i)  
		plt.plot(ds['time'][:]*1e9, ds_fields['Fields:nu_ei'][:,0]/ds_fields['Fields:nu_momentum'][:,0], label = removed_xsec_name_str)
		plt.grid(True)
		plt.xlim([0, tmax])
		plt.xlabel('Time [ns]', fontweight='bold')
		plt.ylabel(r'$\nu_{\rm ei}$/$\nu_{\rm m}$ on axis', fontweight='bold')
		plt.legend(bbox_to_anchor = (1.01, 1.01))
		plt.title(f'Pressure: {pressure} Torr, Drive: {current} kA', fontweight='bold')
		file_name = str(plot_path + 'nu_bar.pdf')
	plt.show()

def run_3_xsec():
    """Run the removal analysis, but only keeping the 3 most sensitive cross sections
    """    
    outpath = Path("../removal_analysis_4_xsec")
    outpath.mkdir(parents=True, exist_ok=True)
    orig_xsec_list = read_xsec("phelps")

    # print("Base case: ")
    # find_and_write_rates("phelps", orig_xsec_list)
    # output_dir = f"../removal_analysis_3_xsec/Base"
    # outpath = Path(f"{output_dir}")
    # outpath.mkdir(parents=True, exist_ok=True)
    # sim = single_run(output_dir, "phelps_rates", True)
    # log_all_outputs(output_dir, use_file=False, sim=sim)
    # plot_outputs(output_dir, use_file = False, sim=sim)

    print("With Effective mom., N2^+, N2(SUM) case: ")
    temp_list = copy.deepcopy(orig_xsec_list)
    
    to_keep = ["N2(SUM)", "N2^+"]
    
    for i, single_xsec in enumerate(orig_xsec_list):
         if i == 0:
              continue
         if single_xsec['product'] not in to_keep:
              temp_list.remove(single_xsec)

    print(len(temp_list))

    find_and_write_rates("phelps", temp_list)
    output_dir = f"../removal_analysis_3_xsec/Keep_N2(SUM)_N2^+"
    outpath = Path(f"{output_dir}")
    outpath.mkdir(parents=True, exist_ok=True)
    sim = single_run(output_dir, "phelps_rates", True)
    log_all_outputs(output_dir, use_file=False, sim=sim)
    plot_outputs(output_dir, use_file = False, sim=sim)

if __name__ == "__main__":
    #run_removal()
    #plot_removal_data()
    plot_curves("../removal_analysis/")
    #run_3_xsec()
    #plot_curves("../removal_analysis_3_xsec/")
    #plot_curves("../removal_analysis_4_xsec/")
