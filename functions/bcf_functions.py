import gc
import h5py
import itertools
import numpy as np
from pathlib import Path
import dask.array as da
import hyperspy.api as hs

def bcf2spec_para(file_path, save_sum_spec=True, save_spectra=True,
                  save_counts=True, save_parameters=True,
                  save_any=True, verbose=False):
    """
    This function reads out the .bcf-file and reads out the
    detector parameters given in the .bcf-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.

    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    ### get the folder path sting
    folder_path = Path(file_path).parent
    file_name = Path(file_path).name
    ### create a data folder to store the data
    Path(folder_path/"data").mkdir(parents=True, exist_ok=True)
    ### open the bruker bcf file
    data = hs.load(file_path, lazy=True,
                   select_type="spectrum_image",
                   )
    ### the last entry contains the measurement data (the others images)
    ### read out parameter
    if verbose:
        print("### read out parameters ###")
    nr_spectra = int(data.data.shape[0] * data.data.shape[1])
    a0 = data.original_metadata["Spectrum"]["CalibAbs"] # "Null"energie in keV
    a1 = data.original_metadata["Spectrum"]["CalibLin"] # Kanalbreite in keV
    fwhm = 2*np.sqrt(2*np.log(2))*np.sqrt(data.original_metadata["Spectrum"]["SigmaAbs"])
    fano = data.original_metadata["Spectrum"]["SigmaLin"]/(3.85e-3)
    channels = data.original_metadata["Spectrum"]["ChannelCount"] # Anzahl Kanäle
    gating_time = 3e-6 # Zeit zum Auslesen Spektrum
    real_time = data.metadata["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["real_time"]
    real_time /= nr_spectra
    ### calculate the mean sum spectrum
    if verbose:
        print("### calculate the mean sum spectrum")
    sum_spec = data.sum()/nr_spectra
    ### calculate the maximum pixel spectrum
    max_pixel_spec = data.max()
    ### calculate the counts per spectrum
    if verbose:
        print("### calculate the counts per spectrum")
    counts = da.from_array(np.array(data.sum(axis=-1)))
    ### get the size of the array
    if verbose:
        print("### get the size of the array")
    size = [data.data.shape[0], data.data.shape[1],1]
    ### calculate a positions tensor from the size
    if verbose:
        print("### calculate a positions tensor from the size")
    row_indices, col_indices, depth_indices = np.indices(size, dtype=np.uint)
    tensor_positions = np.column_stack((row_indices.flatten(), col_indices.flatten(), depth_indices.flatten()))
    del row_indices, col_indices, depth_indices
    ### calculate the mean life time from the zero peaks
    if verbose:
        print("### calculate the mean life time from the zero peaks")
    life_time = data.data[...,75:116].sum()/nr_spectra*1e-4
    ### calculate lifetime for each spectrum
    if verbose:
        print("### calculate lifetime for each spectrum")
    life_times = data.data[...,75:116].sum(axis=-1)*1e-4
    ### normalize the spectra to the measurement life time
    if verbose:
        print("### normalize the spectra to the measurement life time")
    spectra = data.data/life_times[...,None]
    del life_times, data
    spectra = spectra.reshape(nr_spectra, channels).astype(np.float32)
    # spectra = spectra.rechunk((100, 100, channels))
    ### store the parameters
    if verbose:
        print("### store the parameters")
    parameters = np.array([a0, a1, fano, fwhm, life_time, a0 + a1*(channels), gating_time, real_time])
    ### now save everything to a data h5 file
    print("### now save spectra to a data h5 file")
    if (folder_path/"data/data.h5").exists():
        with h5py.File(f"{folder_path}/data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    spectra.to_hdf5(f"{folder_path}/data/data.h5", f"{file_name}/spectra", compression="gzip", shuffle=True)
    if verbose:
        print("### now save everything to a data h5 file")
    with h5py.File(f"{folder_path}/data/data.h5", "r+") as tofile:
        tofile.create_dataset(f"{file_name}/max pixel spec", data=max_pixel_spec, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/sum spec", data=sum_spec, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/counts", data=counts, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/parameters", data=parameters, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/positions", data=tensor_positions, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/tensor positions", data=tensor_positions, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/position dimension", data=size, compression="gzip",
                              shuffle=True)
    del counts
    return spectra, parameters, size, tensor_positions, np.array(sum_spec)

def many_bcf2spec_para(folder_path, signal=None ,worth_fit_threshold=200,
                       save_sum_spec=True, save_spectra=True,
                       save_counts=True, save_parameters=True,
                       save_any=True, print_warning=False,
                       save_spec_as_dict=True,
                       return_values=False, verbose=False):
    """
    This function reads out all the .bcf-files in the inputted folder_path and reads out the
    detector parameters given in the .bcf-files.

    Parameters
    ----------
    folder_path : str
        complete folder path of the folder storing the .bcf-files.

    Returns
    -------
    list containing the sum spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    folder_path = Path(folder_path)
    Path(folder_path/"data").mkdir(parents=True, exist_ok=True)
    
    bcf_file_list = folder_path.glob(folder_path/"*.bcf")   # creates a list with all .bcf-files stored inside the folder
    file_name = folder_path.name
    for file_nr, file_path in enumerate(bcf_file_list):          # iteration over all .bcf-files
        bcf_file = hs.load(file_path, lazy=True)   # loads one .bcf-file
        bcf_file = bcf_file[-1]
        if file_nr == 0:                                         # initialization step to predefine shape of variables
            number_of_channels = bcf_file.data.shape[2]
            nr_spectra = np.prod(bcf_file.data.shape[:2])
            spectra = {}
            overall_life_time = 0
            sum_spec = np.zeros(shape=(number_of_channels,))
            all_counts = np.zeros(shape=(nr_spectra,))
            parameters = np.zeros(shape=(8,))
        spectra_tmp = bcf_file.data
        sum_spec_temporary = np.sum(spectra_tmp, axis=(0,1))/nr_spectra  # sums up all intensities for all measure points for each channel (1D array with shape (4096,))
        a0 = bcf_file.original_metadata["Spectrum"]["CalibAbs"]  # "Zero"energy in keV
        a1 = bcf_file.original_metadata["Spectrum"]["CalibLin"]  # Channel width in keV
        fwhm = 2*np.sqrt(2*np.log(2))*np.sqrt(bcf_file.original_metadata["Spectrum"]["SigmaAbs"])
        fano = bcf_file.original_metadata["Spectrum"]["SigmaLin"]/(3.85e-3)
        channels = bcf_file.original_metadata["Spectrum"]["ChannelCount"]  # number of channels
        gating_time = 3e-6  # time to read out one spectrum
        real_time = bcf_file.metadata["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["real_time"]
        if verbose:
            print(f"real_time from bcf - {real_time} s")
        real_time /= nr_spectra
        # deletes unused variables and clears up some memory space
        gc.collect()
        counts = bcf_file.sum(axis=2)
        counts = np.asarray(counts).reshape(nr_spectra)
        if file_nr == 0:
            position_dimension = [spectra_tmp.shape[0], spectra_tmp.shape[1],1]
            tensor_positions = np.asarray(list(itertools.product(range(position_dimension[0]),
                                                                 range(position_dimension[1]))),
                                          dtype=np.uint)
            tensor_positions = np.hstack((tensor_positions,
                                          np.zeros((len(tensor_positions),1), dtype=np.uint)))
        spectra_temporary = {}
        life_time_sum = 0
        for i in range(nr_spectra):
            spectrum = spectra_tmp[tensor_positions[i][0]][tensor_positions[i][1]]
            life_time = sum(spectrum[75:116])*1e-4 # sum(spectrum[75:116]) is "Zeropeak", an indicator for life time
            life_time_sum += life_time
            spectra_temporary[f"{i}"] =  spectrum / life_time
        life_time_sum /= nr_spectra
        parameters_temporary = np.array([a0, a1, fano, fwhm, life_time_sum, a0 + a1*(channels), gating_time, real_time])
        del a0, a1, fwhm, fano, channels, gating_time, life_time
        del spectra_tmp, bcf_file
        gc.collect()
        # summation over all .bcf-files
        if file_nr == 0:
            spectra = spectra_temporary
            overall_real_time = real_time
            overall_life_time = life_time_sum
            sum_spec = sum_spec_temporary
            all_counts = counts
            parameters = parameters_temporary
        else:
            spectra = {k: spectra.get(k, 0) + spectra_temporary.get(k, 0) for k in spectra_temporary.keys()} # sums up all spectra over all .bcf-files
            overall_real_time += real_time # sums up real time over all .bcf-files
            overall_life_time += life_time_sum # sums up life time over all .bcf-files
            sum_spec += sum_spec_temporary
            all_counts += counts
            parameters += parameters_temporary
    # deletes unused variables and clears up some memory space
        del counts, parameters_temporary, sum_spec_temporary, life_time_sum, real_time
        del spectrum, spectra_temporary
        gc.collect()
    # normalization over the number of bcf-files
    for spec in spectra.keys():
        spectra[spec] /= len(bcf_file_list)
    sum_spec = sum_spec/len(bcf_file_list)
    max_pixel_spec = np.max(np.array(list(spectra.values())), axis=0)
    parameters = parameters/len(bcf_file_list)
    if verbose:
        print(f"overall life_time:\t {overall_life_time} s")
        print(f"overall real_time:\t {overall_real_time} s")
    if (folder_path/"data/data.h5").exists():
        with h5py.File(f"{folder_path}/data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    with h5py.File(f"{folder_path}/data/data.h5", "w") as tofile:
        tofile.create_dataset(f"{file_name}/spectra", data=np.array(list(spectra.values())),
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/max pixel spec", data=max_pixel_spec,
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/sum spec", data=sum_spec,
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/counts", data=all_counts,
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/parameters", data=parameters,
                              compression="gzip")
    return spectra, parameters, position_dimension, tensor_positions, sum_spec

def norm2sec(spectrum, time):
    """
    This function normalizes the given spectrum to seconds based on life or real time.

    Parameters
    ----------
    spectrum : list
    time : float

    Returns
    -------
    list of the normed spectrum
    """
    spectrum[:] = [i / time for i in spectrum]
    return spectrum

def log_file_type(log_file):
    """
    This function defines the type of the given .log-file by reading the
    header and scanning it for a keyphrase
    """
    with open(log_file, "r") as log_file:
        log_file_type = None
        for line in log_file:
            if "Scan started" in line:
                log_file_type = "Louvre"
                break
    return log_file_type

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
    sum_spec = np.divide(values.sum(axis=0),nr_arrays)
    return sum_spec

def bcf_tensor_position(file_path):
    """
    This function returns the tensor_positions and position_dim
    """
    bcf_file = hs.load(file_path, lazy=True)
    if len(bcf_file) <= 5:
        data = bcf_file[4]
        spectra_tmp = data.data
    else:
        data = bcf_file
        spectra_tmp = data.data
    position_dim = [spectra_tmp.shape[0], spectra_tmp.shape[1],1]
    tensor_position = np.asarray(list(itertools.product(range(position_dim[0]), # Indizies von Messpunkten
                                                 range(position_dim[1]))),
                            dtype=np.uint)
    tensor_position = np.hstack((tensor_position,
                                  np.zeros((len(tensor_position),1), dtype=np.uint)))
    return tensor_position, position_dim
