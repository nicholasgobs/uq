import numpy as np
from plasma_chemistry import generate_rates, constructors, maxwellian, factory
from bolos import parse
from pathlib import Path
import copy

def read_xsec(xsec_file : str):
	"""Reads in the cross section data from the string filename

	:param xsec_file: File name to read data from
	:type xsec_file: str
	:return: List of dictionaries of data for each reaction product
	:rtype: List
	"""    
	
	
	
	# processes is a list of dictionaries
    # the keys are target, mass_ratio, product, comment and data
    # data pairs the energy and cross section into a smaller lists of lists
	with open(f"../cross_section_files/{xsec_file}.txt") as fp:
		processes = parse.parse(fp)

	print([(p['kind'], p.get('product', 'N2(X1)')) for p in processes])
	# print([(p.get('product', 'N2(X1)'), p['data']) for p in processes])

	if xsec_file == 'phelps':
        # Took out the (v1res) and the second ionizaiton Xsec for Phelps
		processes = processes[0:3]+processes[4:25]
	else:
		processes = processes

	for p in processes:
		if p['comment'] == "":
			p['comment'] = " : "

	return processes

def find_and_write_rates(xsec_file : str, processes : list, custom_file : str):
	"""From a list of cross section dictionaries, writes the rate coefficients file

	:param xsec_file: Name to describe the cross section source (ex. "phelps"), used for writing the header of the rates file
	:type xsec_file: str
	:param processes: List of dictionaries, each one a reaction product
	:type processes: list
	:param custom_file: A custom rate file name to be generated
	:type custom_file: str
	"""	
	
	
	heavy_species = "N2"
	
	if custom_file is None:
		outdir = Path(f"../rate_files/{xsec_file}_rates")
		outdir.mkdir(parents=True, exist_ok=True)
		output_file = f'../rate_files/{xsec_file}_rates/rates.txt'
	else:
		output_file = custom_file

	bsolver, ratefile = maxwellian.maxwellian(processes, heavy_species)
	with open(output_file, 'w') as f:
		np.savetxt(f, np.r_['0,2', ratefile],
		delimiter=f"{' ':<26}", fmt='%0.3e')
	# ratefile.clear()

	# Fixes happen here
	metadata = constructors.get_bsolver_metadata(bsolver)

	'''print("\n\nMetadata\n\n")
	for k, v in metadata.items():
		print(f"{k} : {v}\n")'''


	metadata["comment"] = constructors.fix_comment_list(metadata["comment"])
	data = factory.DatabaseFactory(xsec_file, **metadata)
	metadata["comment"], metadata["threshold"] = data.fix_first_elements()
	metadata["comment"] = constructors.make_comment_dictionary(metadata["comment"])
	metadata["comment"] = data.fix_comment_dictionary(metadata["comment"])


	th = f"{'!Thresh(eV)':<35}{''.join([format(str(x),'<35') for x in metadata['threshold']])}\n"
	re = f"{'!Reaction':<35}{''.join([format(str(x),'<35') for x in metadata['comment']['PROCESS']])}\n"
	ty = f"{'!Type':<35}{''.join([format(str(x),'<35') for x in metadata['kind']])}\n"
	units = f"{'!<E> (eV)':<35}{''.join(format('Rate_Constant(m^3/s)','<35'))*len(processes)}\n"
	sep = f"{'!'}{''.join('-')*len(re)}\n"

	string = [th, re, ty, units, sep]

	header2 = f"{''.join(string)}"
	file2 = output_file

	b = constructors.WriteHeader(file2, header2)

def modify_xsec(xsec_list : list, percent_change : float, idx_to_change : int):
	"""Applied a flat error multiplier to a certain cross section

	:param xsec_list: List of cross section dictionaries
	:type xsec_list: list
	:param percent_change: Percent change of the cross section, ex. -5%
	:type percent_change: float
	:param idx_to_change: Index of the cross section to be changed
	:type idx_to_change: int
	:return: List
	:rtype: List of cross section dictionaries, now modified
	"""	

	raw_data = []
	for single_xsec in xsec_list:
		single_xsec_data = []
		for point in single_xsec['data']:
			single_xsec_data.append(point[1])
		raw_data.append(single_xsec_data)

	modified_xsec_list = copy.deepcopy(xsec_list)

	for ion_idx, single_modified_xsec in enumerate(modified_xsec_list):
		for eV_idx, point in enumerate(single_modified_xsec['data']):
			if (ion_idx == idx_to_change):
				point[1] = raw_data[ion_idx][eV_idx] * (1  + (percent_change / 100))

	return modified_xsec_list