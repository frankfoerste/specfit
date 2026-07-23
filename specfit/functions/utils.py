import pickle
import numpy as np
import h5py

def norm2sec(spectrum, time):
    """
    This function normalizes the given spectrum to seconds based on life or real time.

    Parameters
    ----------
    spectrum : list
    time : float

    Returns
    -------
    list of the normalized spectrum
    """
    spectrum = [float(i) for i in spectrum]
    spectrum[:] = [i / time for i in spectrum]
    return spectrum

def sum_spec(spectrum):
    """
    This function calculates the sum spectrum of all given spectra.
    """
    values = np.asarray(list(spectrum.values()))
    sum_spec = values.sum(axis=0)
    return sum_spec

def get_file_and_folder_name(path):
    """
    This function returns the folderpath and the filename
    """
    file_name_length = len(path.split('/')[-1:][0])
    folder_path = path[:-file_name_length]
    file_name = path[-file_name_length:]
    return folder_path, file_name

def save_dict_pickle(save_path, dictionary):
    """
    This function saves the dictionary to the given path using pickle
    """
    with open(save_path, 'wb') as saveto:
        pickle.dump(dictionary, saveto, protocol=pickle.HIGHEST_PROTOCOL)

def open_dict_pickle(load_path):
    """
    This function loads a dictionary from a pickle-file
    """
    with open(load_path, 'rb') as loadfrom:
        return pickle.load(loadfrom)
    
def get_hdf5_write_operator(hdf5_file, file_name):
    """
    This function checks the hdf5 file for its content and determines if
    the write operator should overwrite or append
    """
    if hdf5_file.exists():
        with h5py.File(hdf5_file, "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
        write_operator = "r+"
    else:
        write_operator = "w"
    return write_operator
    
def create_hdf5_encoding(dataset):
    """
    This function creates a encoding dictionary for the compression
    of hdf5 files with xarray
    """
    # Encoding options for compression (same for all variables)
    encoding = {var: {"compression": "gzip", 
                      "compression_opts": 5} for var in dataset.data_vars}
    return encoding

def set_xarray_units(dataset):
    """
    Function to automatically set the correct units to
    the axes and variable of an xarray dataset utilised
    by SpecFit.
    The Dataset needs to have the coordinates
    spec_nr  - ['']
    dimension  - ['']
    energy  - ['keV']
    X  - ['mm']
    Y  - ['mm']
    Z  - ['mm']
    parameters
    with the DataArrays:
    counts [spec_nr] - 'counts per second'
    position dimension [dimension]  - ['x', 'y', 'z']
    tensor positions [spec_nr, dimension]  - ['mm', 'mm', 'mm']
    positions [spec_nr, dimension]  - ['mm', 'mm', 'mm']
    spectra [X, Y, Z, energy]  - 'counts per second'
    max pixel spec [energy]  - 'counts per second'
    sum spec [energy]  - 'counts per second'
    parameters [parameter]  - ['keV', 'keV', 'a.u.', 'keV', 's', 'keV', 's', 's']
 
    """    
    dataset.coords["spec_nr"].attrs["units"] = ""
    dataset.coords["parameter"].attrs["units"] = ["keV", "keV", "a.u.", 
                                                  "keV", "s", "keV", 
                                                  "s", "s"]
    dataset.coords["dimension"].attrs["units"] = ""
    dataset.coords["energy"].attrs["units"] = "keV"
    dataset.coords["X"].attrs["units"] = "mm"
    dataset.coords["Y"].attrs["units"] = "mm"
    dataset.coords["Z"].attrs["units"] = "mm"
    # set units to DataArrays
    dataset["counts"].attrs["units"] = "counts per second"
    dataset["position dimension"].attrs["units"] = ["x", "y", "z"]
    dataset["positions"].attrs["units"] = ["mm", "mm", "mm"]
    dataset["tensor positions"].attrs["units"] = ["x", "y", "z"]
    dataset["spectra"].attrs["units"] = "counts per second"
    dataset["max pixel spec"].attrs["units"] = "counts per second"
    dataset["sum spec"].attrs["units"] = "counts per second"
