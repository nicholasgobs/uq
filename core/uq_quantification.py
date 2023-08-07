# %%
from modify_cross_sections import read_xsec
import matplotlib.pyplot as plt
import numpy as np
import scipy

def quantify_uncertainty(plot : bool = False):
	"""Quantifies uncertainty from the cross section files in xsec_dict and plots them before and after interpolation

	:param plot: Boolean to plot cross sections, defaults to False
	:type plot: bool, optional
	:return: Tuple of dictionaries with the mean, standard deviations, and energy levels for each reaction process quantified
	:rtype: tuple
	"""	

	#Read the data from the given files in ../cross_section_files/ and assign names
	xsec_dict = {'Phelps (1985)'    : read_xsec('uq_phelps')
	      		,'Kawaguchi (2021)' : read_xsec('uq_kawaguchi')
				,'Itakawa (2006)'   : read_xsec('uq_itakawa')
				,'Cartwright (1977)': read_xsec('uq_cartwright')
				,'Winters (1966)'   : read_xsec('uq_winters')
				,'Zipf (1980)'	    : read_xsec('uq_zipf')
				,'Cosby (1993)'     : read_xsec('uq_cosby')
				,'Pitchford (1982)' : read_xsec('uq_pitchford')}

	#For compatibility, assign a product to the effective momentum transfer cross section
	xsec_dict['Phelps (1985)'   ][0]['product'] = 'Effective'
	xsec_dict['Kawaguchi (2021)'][0]['product'] = 'Effective'
	xsec_dict['Itakawa (2006)'  ][0]['product'] = 'Effective'
	xsec_dict['Pitchford (1982)'][0]['product'] = 'Effective'

	#Create a set of strings of all the products present across all cross section sources
	products = build_products_set(xsec_dict)
	
	all_avg = {}
	all_std_dev = {}
	all_energies = {}
	
	for product in products:
		#if product != "N2(E3)":
		#	continue
		
		#Plot cross section data for product before interpolation, extrapolation, and average/std.dev. calculation
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		use_log 	  = True	#use log scale in cross section plots
		use_geometric = True	#use geometric mean and std.dev, otherwise arithmetic 

		sources_data = fill_products_set(xsec_dict, product)
		
		if plot:
			plot_single_product(product, sources_data, "", use_log)
			f = plt.figure()
			plt.show()
			plt.close(f)
		
		#Find and plot extrapolated, interpolated values and average/std.dev		
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		all_x = build_all_x_set(sources_data)
		
		#Get cross section value at highest and lowest possible energies
		for source_values in sources_data.values():
			for point in source_values:
				if point[0] == max(all_x):
					y_at_max_x = point[1]
				if point[0] == min(all_x):
					y_at_min_x = point[1]
		
		full_data = {}
		for source_name, source_values in sources_data.items():
			#print(source_name)
			source_x = []
			source_y = []
			
			#Get source cross section data, ignoring duplicate energies or 0 cross section values
			for point in source_values:
				if (point[1] == 0 or point[0] in source_x):
					continue
				source_x.append(point[0])
				source_y.append(point[1])

			if len(source_values) == 0:
				continue
			
			#Get energy values which are to be interpolated onto
			x_in_source_range = [n for n in all_x if n <= max(source_x) and n >= min(source_x)]

			#Find interpolated and extrapolated data
			interpolated = interpolate(source_x, source_y, x_in_source_range)
			extrapolated_left, extrapolated_right = extrapolate(all_x, x_in_source_range, source_x, source_y, y_at_max_x, y_at_min_x)

			#Add all data into single list for single source in the product
			inter_data = [[x_in_source_range[i], interpolated[i]] for i in range(len(interpolated))]
			temp = []
			temp.extend(extrapolated_left)
			temp.extend(inter_data)
			temp.extend(extrapolated_right)

			#Add complete data for given source to dictionary to be plotted			
			full_data[source_name] = temp
		
		#Get cross section data in list at all available energies for mean, std.dev. calculation
		values_at_all_x = []
		for i in range(len(all_x)):
			values_at_all_x.append([])

		for source in full_data.values():
			for i, point in enumerate(source):
				values_at_all_x[i].append(point[1])
		
		#print(temp)

		#Calculate mean and standard deviation for product
		avg, std_dev = find_avg_std_dev(values_at_all_x, use_geometric)

		#Plot the interpolated data and mean with 1 standard deviation range on plot in black and gray
		if plot:
			f = plt.figure()
			plot_single_product(product, full_data, "Interpolated ", use_log)
			if use_geometric:
				plt.fill_between(all_x, np.multiply(avg, std_dev), np.divide(avg, std_dev), color = 'k', alpha = 0.3)
			else:
				plt.fill_between(all_x, np.add(avg, std_dev), np.subtract(avg, std_dev), color = 'k', alpha = 0.3)
			plt.plot(all_x, avg, 'k')
			plt.show()
			plt.close(f)
		
		#Add energies, averages, and standard deviations to dictionary with reaction product as key
		all_avg[product] = avg
		all_std_dev[product] = std_dev
		all_energies[product] = all_x

	return all_avg, all_std_dev, all_energies


def build_products_set(xsec_dict : dict):
	"""Create a set of all reaction products found in all sources

	:param xsec_dict: Dictionary of all sources and their data read from file
	:type xsec_dict: dict
	:return: List of strings of reaction products
	:rtype: list
	"""	
	products = set()
	for xsec_set in xsec_dict.values():
		for single_xsec in xsec_set:
			if not 'product' in single_xsec.keys():
				print("no product")
				continue
			products.update([single_xsec['product']])

	products = sorted(products)
	return products

def fill_products_set(xsec_dict : dict, product : str):
	"""Fill single dictionary with data from all sources from single product

	:param xsec_dict: Dictionary of all data read from all sources
	:type xsec_dict: dict
	:param product: String of reaction product to fill data for
	:type product: str
	:return: Dictionary of data, in which the key is the source and the value is a list of data
	:rtype: dict
	"""	
	data = {}
	for name, xsec_set in xsec_dict.items():
		for single_xsec in xsec_set:
			if not 'product' in single_xsec.keys():
				continue
			if single_xsec['product'] == product:
				temp = []
				for point in single_xsec['data']:
					if point[1] == 0 or point[0] == 0 or point[0] > 1e9:
						continue
					temp.append(point)
				data[name] = temp
	return data

def build_all_x_set(data : dict):
	"""Removes invalid data points and sorts x values (energies)

	:param data: Dictionary of data
	:type data: dict
	:return: List of x values (energies)
	:rtype: list
	"""	
	all_x = set()
	for values in data.values():
		for point in values:
			if point[1] != 0:
				all_x.update([point[0]])
	all_x = sorted(all_x)
	return all_x

def interpolate(source_x : list, source_y : list, x_in_source_range : list):
	"""Interpolates on the source data onto new energies

	:param source_x: Energies from source data
	:type source_x: list
	:param source_y: Cross section values from source data
	:type source_y: list
	:param x_in_source_range: Energy levels to extrapolate onto
	:type x_in_source_range: list
	:return: List of interpolated cross section values
	:rtype: list
	"""	
	spl = scipy.interpolate.interp1d(source_x, source_y)
	interpolated = spl(x_in_source_range)
	return interpolated

def extrapolate(all_x : list, cut_all_x : list, x : list, y : list , y_at_max_x, y_at_min_x):
	"""Extrapolates to higher and lower energy levels from all sources a single reaction product from one source using power function of the form y = b * x ^ m

	:param all_x: All energies from all sources for given reaction product
	:type all_x: list
	:param cut_all_x: Energy values in range of original source
	:type cut_all_x: list
	:param x: Energy values from source
	:type x: list
	:param y: Cross section values from source
	:type y: list
	:param y_at_max_x: Cross section value at highest energy of any source read for reaction product
	:type y_at_max_x: _type_
	:param y_at_min_x: _Cross section value at lower energy of any source read for reaction product, essentially threshold
	:type y_at_min_x: _type_
	:return: Tuple of lower energy extrapolated values and higher level extrapolates values
	:rtype: tuple
	"""	
	#Extrapolate to higher energies if possible
	if not max(all_x) in x:
		#Linear regression fit on log-log scale
		result = scipy.stats.linregress(np.log([x[-1], max(all_x)]), np.log([y[-1], y_at_max_x]))
		
		#Power function fit parameters from log-log linear fit
		m_right = result.slope
		b_right = np.exp(result.intercept)
	
	#Extrapolate to lower energies if possible using the same process
	if not min(all_x) in x:
		result = scipy.stats.linregress(np.log([min(all_x), x[0]]), np.log([y_at_min_x, y[0]]))
		m_left = result.slope
		b_left = np.exp(result.intercept)
	
	#Create extrapolated cross section values at necessary higher energy levels using power function fit
	extrapolated_right = []
	for extended_x_right in all_x[all_x.index(cut_all_x[-1]) + 1:]:
		extrapolated_right.append([extended_x_right, b_right * (extended_x_right ** m_right)])
	
	#Create extrapolated cross section values at necessary lower energy levels using power function fit
	extrapolated_left = []
	for extended_x_left in all_x[0:all_x.index(cut_all_x[0])]:
		value = b_left * (extended_x_left ** m_left)
		#Check for 0 or NaN values, if present, use cross section at lowest available energy level
		if value == 0 or np.isnan(value):
			extrapolated_left.append([extended_x_left, y_at_min_x])
		else:
			extrapolated_left.append([extended_x_left, value])
	
	# print(f"Left  0:{len(np.where(extrapolated_left == 0)[0])}")
	# print(f"Right 0:{len(np.where(extrapolated_right == 0)[0])}")

	# print(f"Left  n:{int(np.isnan(np.sum(extrapolated_left)))}")
	# print(f"Right n:{int(np.isnan(np.sum(extrapolated_right)))}")
	
	return extrapolated_left, extrapolated_right

def find_avg_std_dev(values : list, use_geometric : bool):
	"""Find the mean and standard deviation of the values at each energy level

	:param values: List of lists of cross sections at a certain energy
	:type values: list
	:param use_geometric: To use geometric mean and standard deviation or arithmetic
	:type use_geometric: bool
	:return: List of means and standard deviations at each energy level
	:rtype: list
	"""	
	avg = []
	std_dev = []
	for vals_at_x in values:
		if use_geometric:
			avg.append(scipy.stats.gmean(vals_at_x))	
			if len(vals_at_x) > 1:
				std_dev.append(scipy.stats.gstd(vals_at_x))
			else:
				std_dev.append(1)
		else:
			avg.append(np.average(vals_at_x))
			std_dev.append(np.std(vals_at_x))
	
	# print(f"avg   0:{len(np.where(avg == 0)[0])}")
	# print(f"std   0:{len(np.where(std_dev == 0)[0])}")

	# print(f"avg   n:{int(np.isnan(np.sum(avg)))}")
	# print(f"std   n:{int(np.isnan(np.sum(std_dev)))}")
	
	return avg, std_dev

def plot_single_product(product_name : str, data : dict, title : str, use_log : bool):	
	"""Plot one reaction cross section from all sources

	:param product_name: Name of reaction process
	:type product_name: str
	:param data: Dictionary of data corresponding to each source
	:type data: dict
	:param title: String to be added before the process name in the plot title
	:type title: str
	:param use_log: Boolean to use log for cross section (y) axis
	:type use_log: bool
	"""	
	legend_names = []
	for name, values in data.items():
		legend_names.append(name)
		x = []
		y = []
		for point in values:
			if (point[0] in x):
				continue
			x.append(point[0])
			y.append(point[1])
		plt.plot(x, y)
	plt.xscale('log')
	if use_log:
		#if product_name != 'Effective':
		#	plt.yscale('log')
		plt.yscale("log")
	plt.xlabel('Energy (eV)')
	plt.ylabel('Cross Section (m^2)')
	plt.grid()
	plt.legend(legend_names)
	plt.title(title + product_name)


if __name__ == "__main__":
	quantify_uncertainty(plot = True)