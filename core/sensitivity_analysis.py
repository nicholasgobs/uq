import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from rigid_beam_all_outputs import single_run, log_all_outputs
from modify_cross_sections import modify_xsec, read_xsec, find_and_write_rates


def sens_analysis_all_ions(xsec_file : str, xsec_list : list, output_dir : str):
	"""Runs the sensitivity analysis, applying the error percentages in errors list

	:param xsec_file: Name of source of cross sections, used to generate specific rate file
	:type xsec_file: str
	:param xsec_list: List of cross section data, each product is one dictionary in the list
	:type xsec_list: list
	:param output_dir: Directory where output will be saved
	:type output_dir: str
	"""	

	errors = [-10, -5, 0, 5, 10]

	for ion_idx, ion in enumerate(xsec_list):

		if 'product' in ion:
			product_name = f"_to_{ion['product']}"
		else:
			product_name = ''
		temp_ion_dir = f"{output_dir}/{ion['kind']}{product_name}"
		
		ion_outpath = Path(f"{temp_ion_dir}")
		ion_outpath.mkdir(parents=True, exist_ok=True)

		for percent in errors:
			temp_percent_dir = f"{temp_ion_dir}/Error_{percent}%"

			print(f"Ion: {ion['kind']} -> {product_name} at {percent} %")

			percent_outpath = Path(f"{temp_percent_dir}")
			percent_outpath.mkdir(parents=True, exist_ok=True)

			temp_xsec_list = modify_xsec(xsec_list, percent, ion_idx)

			find_and_write_rates(xsec_file, temp_xsec_list)
			
			sim = single_run(temp_percent_dir, run=False)
			log_all_outputs(temp_percent_dir, sim=sim, use_file=True)

def plot_sens_analysis(output_dir :str):
	"""Plots the results of the sensitivity analysis for each reaction product

	:param output_dir: Directory to find the simulation output files
	:type output_dir: str
	"""
	qoi_plot_outpath = Path(f"{output_dir}/QOI_plots/")
	qoi_plot_outpath.mkdir(parents=True, exist_ok=True)

	time_plot_outpath = Path(f"{output_dir}/Time_plots/")
	time_plot_outpath.mkdir(parents=True, exist_ok=True)

	densities_plot_outpath = Path(f"{output_dir}/Density_plots/")
	densities_plot_outpath.mkdir(parents=True, exist_ok=True)

	all_changed_xsecs = Path(output_dir).glob("*")
	for changed_xsec in all_changed_xsecs:
		if str(changed_xsec).find("plots") != -1:
			continue
		xsec_name = str(changed_xsec)[len(output_dir) + 1:].replace("_to_", " -> ")
		print(f"\n{xsec_name}")

		all_QOI = {}
		all_times = {}
		all_densities = {}

		single_dir = Path(changed_xsec).glob("*")
		for error_idx, error_dir in enumerate(single_dir):
			error_int = int(str(error_dir)[str(error_dir).index('_', -5, -1) + 1:-1])
			#print(f"{error_idx}: {error_int}")
			
			qoi_dir = f"{error_dir}/QOI/"
			single_run_qois = Path(qoi_dir).glob("*")
			for qoi_name_idx, data_file in enumerate(single_run_qois):
				qoi_file_name = str(data_file)[len(str(error_dir) + "/QOI/"):]
				qoi_name = qoi_file_name.replace('.txt', '').replace('_', ' ')
				#print(qoi_file)
				#print(qoi_name)
				point = dict([(error_int, np.loadtxt(qoi_dir + qoi_file_name, max_rows=1))])
				if not qoi_name in list(all_QOI):
					all_QOI[qoi_name] = point
				else:
					all_QOI[qoi_name][list(point)[0]] = point[list(point)[0]]

			times_dir = f"{error_dir}/QOI_times/"
			single_run_times = Path(times_dir).glob("*")
			for times_name_idx, data_file in enumerate(single_run_times):
				time_file_name = str(data_file)[len(str(error_dir) + "/QOI_times/"):]
				time_name = time_file_name.replace('.txt', '').replace('_', ' ')

				point = dict([(error_int, np.loadtxt(times_dir + time_file_name, max_rows=1))])
				if not time_name in list(all_times):
					all_times[time_name] = point
				else:
					all_times[time_name][list(point)[0]] = point[list(point)[0]]

			densities_dir = f"{error_dir}/peak_densities/"
			single_run_densities = Path(densities_dir).glob("*")
			for density_name_idx, density_file in enumerate(single_run_densities):
				density_file_name = str(density_file)[len(str(error_dir) + "/peak_densities/"):]
				density_name = density_file_name.replace('.txt', '')
				point = dict([(error_int, np.loadtxt(densities_dir + density_file_name, max_rows=1))])
				if not density_name in list(all_densities):
					all_densities[density_name] = point
				else:
					all_densities[density_name][list(point)[0]] = point[list(point)[0]]

		normalized_qoi_data = {}
		for name, data in all_QOI.items():
			#print(data)
			no_error = data[0]
			single_dict = {}
			for error, value in data.items():
				single_dict[error] = 100 * (value - no_error) / no_error
			normalized_qoi_data[name] = single_dict

		normalized_time_data = {}
		for name, data in all_times.items():
			#print(data)
			no_error = data[0]
			single_dict = {}
			for error, value in data.items():
				single_dict[error] = (value - no_error)
			normalized_time_data[name] = single_dict

		normalized_densities_data = {}
		for name, data in all_densities.items():
			#print(data)
			no_error = data[0]
			single_dict = {}
			for error, value in data.items():
				single_dict[error] = 100 * (value - no_error) / no_error
			normalized_densities_data[name] = single_dict
		
	
		file_format = "pdf"

		f = plt.figure()
		for data in normalized_qoi_data.values():
			sorted_data = {}
			for k in sorted(data.keys()):
				sorted_data[k] = data[k]
			plt.plot(sorted(data.keys()), sorted_data.values())
		plt.legend(normalized_qoi_data.keys(), bbox_to_anchor = (1.01, 1.01))
		plt.title(xsec_name)
		plt.xlabel("Percent Change in Cross Section")
		plt.ylabel("Percent Change in QOI")
		plt.grid()
		#plt.show()
		f.savefig(f"./{output_dir}/QOI_plots/{xsec_name}.{file_format}", bbox_inches='tight')
		plt.close(f)
		
		f = plt.figure()
		for data in normalized_time_data.values():
			sorted_data = {}
			for k in sorted(data.keys()):
				sorted_data[k] = data[k]
			plt.plot(sorted(data.keys()), sorted_data.values())
		plt.legend(normalized_time_data.keys(), bbox_to_anchor = (1.01, 1.01))
		plt.title(xsec_name)
		plt.xlabel("Percent Change in Cross Section")
		plt.ylabel("Change in Time of QOI Peak")
		plt.grid()
		#plt.show()
		f.savefig(f"./{output_dir}/Time_plots/{xsec_name}_times.{file_format}", bbox_inches='tight')
		plt.close(f)

		f = plt.figure()
		for data in normalized_densities_data.values():
			sorted_data = {}
			for k in sorted(data.keys()):
				sorted_data[k] = data[k]
			plt.plot(sorted(data.keys()), sorted_data.values())
		plt.legend(normalized_densities_data.keys(), bbox_to_anchor = (1.01, 1.01))
		plt.title(xsec_name)
		plt.xlabel("Percent Change in Cross Section")
		plt.ylabel("Percent Change in Density")
		plt.grid()
		#plt.show()
		f.savefig(f"./{output_dir}/Density_plots/{xsec_name}_densities.{file_format}", bbox_inches='tight')
		plt.close(f)


def main():
    xsec_file = 'phelps'
    output_dir = "../phelps_sensitivity_analysis"

    xsec_list = read_xsec(xsec_file)
    sens_analysis_all_ions(xsec_file, xsec_list, output_dir)
    plot_sens_analysis(output_dir)
	

if __name__ == "__main__":
	main()