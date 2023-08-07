import multiprocessing as m
from rigid_beam_all_outputs import *
from modify_cross_sections import *
from uq_quantification import *
from copy import deepcopy
import time

def single_uq_run(args: tuple):    
    """Performs a single run of the UQ loop

    :param args: Tuple of data needed for the uq loop: (int n, list avgs, list std_devs, list energies)
    :type args: tuple
    """    
    n = args[0]
    avgs = args[1]
    std_devs = args[2]
    energies = args[3]
    xsec_data = deepcopy(args[4])
    print(f"Run #{n}")

    #Get the thread number to seed random number generation and avoid each thread creating identical numbers
    thread_number = int(str(m.current_process()).split('r-')[1].split('parent')[0][:-2])
    np.random.seed(int(1e6 * np.random.random() * thread_number))
    print(f"Thead number: {thread_number}.")

    #Apply the standard deviation randomly to the mean data and save in xsec_data list
    apply_mean_stddev(avgs, std_devs, energies, xsec_data)

    #Generate rate file to be used by this thread
    rate_file = f"../multi_test/rates/rates_{thread_number}.txt"
    find_and_write_rates("phelps", xsec_data, rate_file)
    
    #Configure and run simulationm
    output_dir = "../multi_test/"
    problem_config = configure_sim("", output_dir)
    problem_config['PhysicsModules']['PlasmaChemistry']['RateFile'] = rate_file
    sim = turbopy.Simulation(problem_config)
    try:
        sim.run()
    except:
        print(f"ERROR: {thread_number} ~~~~~~~~~~~~~~~~~~~~~~~")

    #Log the QOI for the simulation
    log_all_outputs(output_dir, use_file=False, sim=sim)
    log_outputs_over_time(output_dir, sim, n)

def apply_mean_stddev(avgs: dict, std_devs: dict, energies: dict, reference: list):
    """Loop through the data and apply the mean and standard deviation

    :param avgs: Dictionary of lists of the average xsec value for each reaction
    :type avgs: dict
    :param std_devs: Dictionary of lists of the 
    :type std_devs: dict
    :param energies: The common set of energies for each reaction
    :type energies: dict
    :param reference: A reference cross section set, such that its comments and process data can be used in rate file generation
    :type reference: list
    """    
    for single_xsec in reference:
        #Get product string
        product = single_xsec['product']
        
        #Set threshold energy as the minimum non-zero value from the average dataset
        single_xsec['threshold'] = min(energies[product])
        
        #Randomly sample
        z = np.random.normal()
        
        #Create list in rate-generating-compatible format of the data with the mean and std.dev applied to the data
        single_xsec['data'] = [[e, avgs[product][i] * (std_devs[product][i] ** z)] for i, e in enumerate(energies[product])]

def plot_qoi_distribution(output_dir: str):
    """Creates histograms of the quantities of interest saved to the given output directory

    :param output_dir: String of the name of the directory where the QOI text files are saved
    :type output_dir: str
    """
    for i, qoi in enumerate(Path(output_dir + "QOI/").glob("*")):
        qoi_file_name = str(qoi)[len(str(output_dir) + "QOI/"):]
        qoi_name = qoi_file_name.replace('.txt', '').replace('_', ' ')
        print(qoi_name)
        try: 
            plt.figure()
            plt.hist(np.loadtxt(qoi), bins = 20)
            plt.title(qoi_name)
            plt.show()
        except:
            continue
    for i, qoi in enumerate(Path(output_dir + "peak_densities/").glob("*")):
        qoi_file_name = str(qoi)[len(str(output_dir) + "peak_densities/"):]
        qoi_name = qoi_file_name.replace('.txt', '').replace('_', ' ')
        plt.figure()
        plt.hist(np.loadtxt(qoi), bins = 20)
        plt.title(f"Peak density: {qoi_name}")
        plt.show()

    for i, qoi in enumerate(Path(output_dir + "QOI_over_time/").glob("*")):
        qoi_file_name = str(qoi)[len(str(output_dir) + "QOI_over_time/"):]
        qoi_name = qoi_file_name.replace('.npy', '').replace('_', ' ')
        plt.figure()
        plt.grid()
        data = []
        for single_file in Path(qoi).glob("*"):
            nums = np.load(single_file)[:-1]
            #print(nums)
            data.append(list(nums))
        avg = np.average(np.array(data), axis = 0)
        std_dev = np.std(np.array(data), axis = 0)
        plt.plot(avg)
        plt.fill_between(range(len(data[0])), np.add(avg, 3*std_dev), np.subtract(avg, 3*std_dev), color = 'k', alpha = 0.2)
        plt.fill_between(range(len(data[0])), np.add(avg, 2*std_dev), np.subtract(avg, 2*std_dev), color = 'k', alpha = 0.2)
        plt.fill_between(range(len(data[0])), np.add(avg, std_dev), np.subtract(avg, std_dev), color = 'k', alpha = 0.2)
        plt.title(f"{qoi_name} over time")
        plt.xlabel("Time step")
        plt.show()

def log_outputs_over_time(output_dir : str, sim, n):    
    ds = sim.diagnostics[1]._traces
    ds_fields = sim.diagnostics[2]._traces
    
    output_dir = f"{output_dir}QOI_over_time"
    QOI_over_time_outdir = Path(output_dir)
    QOI_over_time_outdir.mkdir(parents=True, exist_ok=True)
    
    net_current_output_dir = f"{output_dir}/net_current"
    net_current_outdir = Path(net_current_output_dir)
    net_current_outdir.mkdir(parents=True, exist_ok=True)
    data = ds['synthetic_integral:SyntheticNetCurrent_J'].values
    np.save(f'{net_current_output_dir}/' + f'net_current{n}.npy', data, allow_pickle=True)

    line_density_output_dir = f"{output_dir}/line_density"
    line_density_outdir = Path(line_density_output_dir)
    line_density_outdir.mkdir(parents=True, exist_ok=True)
    data = ds['line_density'].values
    np.save(f'{line_density_output_dir}/' + f'line_density{n}.npy', data, allow_pickle=True)

    electron_energy_output_dir = f"{output_dir}/electron_energy"
    electron_energy_outdir = Path(electron_energy_output_dir)
    electron_energy_outdir.mkdir(parents=True, exist_ok=True)
    data = ds_fields['Fields:electron_energy'].values[:,0]
    np.save(f'{electron_energy_output_dir}/' + f'electron_energy{n}.npy', data, allow_pickle=True)

    

def main_loop(n: int, output_dir: str):
    """Runs the UQ loop in output_dir for n runs

    :param n: Number of runs to complete
    :type n: int
    :param output_dir: Directory to store output
    :type output_dir: str
    """
    #Create the output directory if it does not exits already
    outpath = Path(f"{output_dir}")
    outpath.mkdir(parents=True, exist_ok=True)

    #Create the rates directory if it does not exist already
    rates_dir = f"{output_dir}/rates/"
    outpath = Path(f"{rates_dir}")
    outpath.mkdir(parents=True, exist_ok=True)
    
    #Quantify uncertainty
    avg_all_products, std_dev_all_products, energies_all_products = quantify_uncertainty(plot = False)
    
    #Use phelps set as reference data
    phelps_data = read_xsec("uq_phelps")
    phelps_data[0]['product'] = "Effective"
    phelps_products = [single_xsec['product'] for single_xsec in phelps_data]

    #Get average, standard deviations, and energies for each product in the reference from the quantified uncertainty of all sources
    avgs = {product: avg_all_products[product] for product in phelps_products}
    std_devs = {product: std_dev_all_products[product] for product in phelps_products}
    energies = {product: energies_all_products[product] for product in phelps_products}

    #Parallelize UQ loop and run 
    pool = m.Pool()
    results = pool.map(single_uq_run, [(x, avgs, std_devs, energies, phelps_data) for x in range(n)])
    pool.close()

if __name__ == "__main__":
    n = 1000
    output_dir = f"../multi_test/"
    
    start = time.time()    
    
    #main_loop(n, output_dir)
    plot_qoi_distribution(output_dir)

    end_seconds = time.time() - start
    print(f"Time: {round(end_seconds)} s for {n} runs")

  

    