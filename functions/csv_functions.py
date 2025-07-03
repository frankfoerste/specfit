import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
        
def csv2spec_para(file_path, print_warning = False):
    """ 
    This function reads out the spectra of a .csv-file and reads out the
    detector parameters given in the .csv-file.
    !!!OBS!!! only .csv file format as returned by XRS-FP2 simulations supported
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .csv-file.
    
    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    file_name = file_path.split("/")[-1]
    ### define default values
    a0 = 0.0
    a1 = 0.01
    FWHM = 0.0792
    Fano = 0.113
    channels = 4096
    max_energy = a0 + a1 * (channels-1)
    gating_time = 3e-6
    life_time, real_time = 1, 1
    parameters = np.array([a0,a1,Fano,FWHM, life_time, max_energy, gating_time, real_time])
    start = False
    with open(file_path,"r") as infile:
        for line in infile:
            if "Files Created (Actual):" in line:
                samples = int(line.split(",")[-1])
                spectra = np.zeros((samples, 4096))
            elif "Channel #" in line:
                start = True
                continue
            if start:
                line = line.split(",")
                channel = int(line[0])-1
                spectra[:,channel] = np.array([int(intensity) for intensity in line[1:]])
    folder_path = "/".join(file_path.split("/")[:-1])
    try: os.mkdir(f"{folder_path}/data/")
    except: pass
    if os.path.exists(f"{folder_path}/data/data.h5"):
        with h5py.File(f"{folder_path}/data/data.h5", "r+") as tofile:
            del tofile[file_name]
    with h5py.File(f"{folder_path}/data/data.h5", "w") as tofile:
        tofile.create_dataset(f"{file_name}/spectra",
                              data = spectra,
                              compression = "gzip", compression_opts = 9)
        tofile.create_dataset(f"{file_name}/sum spec", data = np.sum(spectra, axis = 0))
        tofile.create_dataset(f"{file_name}/counts", data = [np.sum(spectra)])
        tofile.create_dataset(f"{file_name}/parameters", data = parameters)
        tofile.create_dataset(f"{file_name}/max pixel spec", data = spectra)
        tofile.create_dataset(f"{file_name}/position dimension", data = np.array([1,1,1]))
        tofile.create_dataset(f"{file_name}/tensor positions", data = np.array([[0,0,0]]))
        tofile.create_dataset(f"{file_name}/positions", data = np.array([[0,0,0]]))
    return spectra, parameters
      
def csv_position_dimension(file_path):
    """
    This function reads out the size of each dimension of the spectra in the 
    measurement-tensor.
    returns dimension = [nr_x,nr_y,nr_z]
    position_dimension = csv_position_dimension(file_path)
    """
    if file_path[-4:] == ".csv":
        with open(file_path, "r") as infile:
            for line in infile:
                if "Files Created (Actual):" in line: 
                    samples = int(line.split(",")[-1])
    return np.array([samples,1,1])

def csv_tensor_positions(file_path):
    """
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x,y,z]
    position = spx_tensor_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    """
    if file_path[-4:] == ".csv":
        with open(file_path, "r") as infile:
            for line in infile:
                if "Files Created (Actual):" in line: 
                    samples = int(line.split(",")[-1])
    tensor_positions = np.zeros((samples,3), dtype = int)
    tensor_positions[:,0] = np.arange(samples)
    return tensor_positions

def convert_string(string):
    string = string.replace(",",".")
    power = float(string[-2:])
    leading = float(string[:6])
    convert = 10**power*leading/1000
    return convert

def calc_sum_spec(spectrum): 
    """
    This function calculates the sum spectrum of all given spectra.
    """
    values = np.asarray(list(spectrum.values()))
    nr_arrays = len(values)
    sum_spec = np.divide(values.sum(axis = 0),nr_arrays)
    return sum_spec
