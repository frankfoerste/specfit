import gc
import h5py
import itertools
import psutil
import numpy as np
from pathlib import Path
import dask.array as da
import hyperspy.api as hs
import xarray as xr
import utils

def bcf2spec_para(file_path, return_values=False, verbose=False):
    """
    This function reads out the .bcf-file and reads out the
    detector parameters given in the .bcf-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.
    return_values : bool
        if True the spectra and parameter will be returned, else None
    verbose: bool
        if True verbose mode is activated

    """
    # get the folder path sting
    file_path = Path(file_path)
    folder_path = file_path.parent
    file_name = file_path.name
    write_operator = utils.get_hdf5_write_operator(hdf5_file=folder_path/"data/data.h5",
                                                   file_name=file_name)
    ds = xr.Dataset()
    # create a data folder to store the data
    (folder_path / "data").mkdir(parents=True, exist_ok=True)
    # open the bruker bcf file
    data = hs.load(file_path, lazy=True,
                   select_type="spectrum_image",
                   signal_type="EDS_SEM"
                   )
    # the last entry contains the measurement data (the others images)
    # read out parameter
    if verbose:
        print("# read out parameters ###")
    nr_spectra = int(data.data.shape[0] * data.data.shape[1])
    dx = int(data.original_metadata.Microscope.DX)/1000
    dy = int(data.original_metadata.Microscope.DY)/1000
    X0 = np.round(data.original_metadata.Stage.X, 3)
    Y0 = np.round(data.original_metadata.Stage.Y, 3)
    Z0 = np.round(data.original_metadata.Stage.Z, 3)
    a0 = data.original_metadata["Spectrum"]["CalibAbs"] # "Null"energie in keV
    a1 = data.original_metadata["Spectrum"]["CalibLin"] # Kanalbreite in keV
    fwhm = 2*np.sqrt(2*np.log(2))*np.sqrt(data.original_metadata["Spectrum"]["SigmaAbs"])
    fano = data.original_metadata["Spectrum"]["SigmaLin"]/(3.85e-3)
    channels = data.original_metadata["Spectrum"]["ChannelCount"] # Anzahl Kanäle
    gating_time = 3e-6 # Zeit zum Auslesen Spektrum
    real_time = data.metadata["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["real_time"]
    real_time /= nr_spectra
    # get the size of the array
    if verbose:
        print("# get the size of the array")
    size = [data.data.shape[0], data.data.shape[1],1]
    ds["position dimension"] = xr.DataArray(data=size,
                                            dims=("dimension"))
    # calculate a positions tensor from the size
    if verbose:
        print("# calculate a positions tensor from the size")
    row_indices, col_indices, depth_indices = np.indices(size, dtype=np.uint)
    ds["tensor positions"] = xr.DataArray(data=np.column_stack((row_indices.flatten(), col_indices.flatten(), depth_indices.flatten())),
                                   dims=("spec_nr", "dimension")
                                   )
    ds["positions"] = xr.DataArray(data=ds["tensor positions"]*[dx,dy,1]+[X0, Y0, Z0],
                                   dims=("spec_nr", "dimension"),
                                   coords={"dimension": ["x", "y", "z"]},
                                   attrs={"units": ["mm", "mm", "mm"]})
    
    del row_indices, col_indices, depth_indices
    # calculate the mean life time from the zero peaks
    if verbose:
        print("# calculate the mean life time from the zero peaks")
    life_time = data.data[...,75:116].sum()/nr_spectra*1e-4
    # calculate lifetime for each spectrum
    if verbose:
        print("# calculate lifetime for each spectrum")
    life_times = data.data[...,75:116].sum(axis=-1)*1e-4
    # normalize the spectra to the measurement life time
    if verbose:
        print("# normalize the spectra to the measurement life time")
    spectra = data.data/life_times[...,np.newaxis]
    spectra = spectra.reshape([data.data.shape[0], data.data.shape[1], 1, channels])
    ds["spectra"] = xr.DataArray(spectra,
                                 dims=("X", "Y", "Z", "energy"),
                                 coords={"energy": np.arange(a0, a0 + a1*(channels), a1),
                                         "X": np.arange(X0, X0+dx*data.data.shape[0]-dx, dx),
                                         "Y": np.arange(Y0, Y0+dy*data.data.shape[1]-dy, dy),
                                         "Z": np.array([Z0])},
                                 attrs={"units": "counts per second"})
    ds["counts"] = xr.DataArray(data=np.ravel(ds["spectra"].sum(axis=-1)),
                                dims=("spec_nr"),
                                coords={"spec_nr": np.arange(len(data))},
                                attrs={"units": "counts per second"})
    ds["max pixel spec"] = ds["spectra"].max(axis=(0, 1, 2))
    ds["sum spec"] = ds["spectra"].mean(axis=(0, 1, 2))
    
    # store the parameters
    if verbose:
        print("# store the parameters")
    ds["parameters"] = xr.DataArray(data=[a0, a1, fano, fwhm, life_time, 
                                          a0 + a1*(channels), gating_time,
                                          real_time],
                                    dims=("parameter"),
                                    coords={"parameter": ["a0", "a1", "Fano", "FWHM",
                                                     "life_time", "max_energy",
                                                     "gating_time", "real_time"]},
                                    attrs={"units": ["keV", "keV", "a.u.", "keV",
                                                     "s", "keV", "s", "s"]})

    # now save everything to a data h5 file
    if (folder_path/"data/data.h5").exists():
        with h5py.File(folder_path/"data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    utils.set_xarray_units(ds)
    ds.to_netcdf(
        path=folder_path/"data/data.h5",
        group=file_name,
        mode=write_operator,
        engine="h5netcdf",
        encoding=utils.create_hdf5_encoding(dataset=ds),
    )
    if verbose:
        print("# now save everything to a data h5 file")
    if return_values:
        return ds["spectra"], ds["parameters"], ds["position dimension"], ds["tensor positions"], ds["sum spec"]

def bcf2spec_para_dask(folder_path, verbose=True):
    """
    This is a copy of the bcf2spec_para function with dask implementation
    """
    # get the folder path sting
    file_path = Path(file_path)
    folder_path = file_path.parent
    file_name = file_path.name
    if (folder_path / "data/data.h5").exists():
        with h5py.File(folder_path / "data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
        write_operator = "r+"
    else:
        write_operator = "w"
    # create a data folder to store the data
    (folder_path / "data").mkdir(parents=True, exist_ok=True)
    # open the bruker bcf file
    data = hs.load(file_path, lazy=True,
                   select_type="spectrum_image",
                   signal_type="EDS_SEM"
                   )
    # the last entry contains the measurement data (the others images)
    # read out parameter
    if verbose:
        print("# read out parameters ###")
    nr_spectra = int(data.data.shape[0] * data.data.shape[1])
    dx = int(data.original_metadata.Microscope.DX)/1000
    dy = int(data.original_metadata.Microscope.DY)/1000
    a0 = data.original_metadata["Spectrum"]["CalibAbs"] # "Null"energie in keV
    a1 = data.original_metadata["Spectrum"]["CalibLin"] # Kanalbreite in keV
    fwhm = 2*np.sqrt(2*np.log(2))*np.sqrt(data.original_metadata["Spectrum"]["SigmaAbs"])
    fano = data.original_metadata["Spectrum"]["SigmaLin"]/(3.85e-3)
    channels = data.original_metadata["Spectrum"]["ChannelCount"] # Anzahl Kanäle
    gating_time = 3e-6 # Zeit zum Auslesen Spektrum
    real_time = data.metadata["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["real_time"]
    real_time /= nr_spectra
    # calculate the mean sum spectrum
    if verbose:
        print("# calculate the mean sum spectrum")
    sum_spec = data.sum()/nr_spectra
    # calculate the maximum pixel spectrum
    max_pixel_spec = data.max()
    # calculate the counts per spectrum
    if verbose:
        print("# calculate the counts per spectrum")
    counts = da.from_array(np.array(data.sum(axis=-1)),
                           chunks=(data.get_chunk_size()))
    # get the size of the array
    if verbose:
        print("# get the size of the array")
    size = [data.data.shape[0], data.data.shape[1],1]
   # calculate a positions tensor from the size
    if verbose:
        print("# calculate a positions tensor from the size")
    row_indices, col_indices, depth_indices = np.indices(size, dtype=np.uint)
    tensor_positions = da.from_array(np.column_stack((row_indices.flatten(), col_indices.flatten(), depth_indices.flatten())),
                                     chunks=(10000,3))
    del row_indices, col_indices, depth_indices
    # calculate the mean life time from the zero peaks
    if verbose:
        print("# calculate the mean life time from the zero peaks")
    life_time = data.data[...,75:116].sum()/nr_spectra*1e-4
    # calculate lifetime for each spectrum
    if verbose:
        print("# calculate lifetime for each spectrum")
    life_times = data.data[...,75:116].sum(axis=-1)*1e-4
    # normalize the spectra to the measurement life time
    if verbose:
        print("# normalize the spectra to the measurement life time")
    spectra = data.data/life_times[...,None]
    spectra = spectra.reshape([data.data.shape[0], data.data.shape[1], 1, channels])
    spectra = da.asarray(spectra.reshape(nr_spectra, channels).astype(np.float64),
                         ).rechunk(chunks=(5000, channels))
    # store the parameters
    if verbose:
        print("# store the parameters")
    parameters = da.from_array(np.array([[a0, a1, fano, fwhm, life_time, a0 + a1*(channels), gating_time, real_time]]),
                               chunks=(10000,8))
    # now save everything to a data h5 file
    print("# now save spectra to a data h5 file")
    if verbose:
        print("# now save everything to a data h5 file")
    with h5py.File(folder_path/"data/data.h5", "r+") as tofile:
        tofile.create_dataset(f"{file_name}/max pixel spec", data=max_pixel_spec, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/sum spec", data=sum_spec, compression="gzip",
                              shuffle=True)
        tofile.create_dataset(f"{file_name}/position dimension", data=size, compression="gzip",
                              shuffle=True)
    return spectra, parameters, position_dimension, tensor_positions, sum_spec

def many_bcf2spec_para(folder_path, return_values=False, verbose=False):
    """
    This function reads out all the .bcf-files in the inputted folder_path and reads out the
    detector parameters given in the .bcf-files.

    Parameters
    ----------
    folder_path : str
        absolute folder path of the folder storing the .bcf-files.

    Returns
    -------
    list containing the sum spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    folder_path = Path(folder_path)
    Path(folder_path / "data").mkdir(parents=True, exist_ok=True)
    bcf_file_list = [file for file in folder_path.glob("*.bcf")]   # creates a list with all .bcf-files stored inside the folder
    nr_bcf_files = len(bcf_file_list)
    file_name = folder_path.name
    write_operator = utils.get_hdf5_write_operator(hdf5_file=folder_path/"data/data.h5",
                                                   file_name=file_name)
    # create xarray Dataset
    ds = xr.Dataset()
    for file_nr, file_path in enumerate(bcf_file_list):          # iteration over all .bcf-files
        data = hs.load(file_path, lazy=True,
                       select_type="spectrum_image",
                       signal_type="EDS_SEM")   # loads one .bcf-file
        if file_nr == 0:                                         # initialization step to predefine shape of variables
            channels = data.original_metadata["Spectrum"]["ChannelCount"]  # number of channels
            nr_spectra = np.prod(data.data.shape[:-1])
            spectra = np.zeros(shape=data.data.shape[:-1]+(nr_bcf_files,)+(data.data.shape[-1],))
            overall_life_time = 0
            parameters = np.zeros(shape=(8,))
            X0 = np.round(data.original_metadata.Stage.X, 3)
            Y0 = np.round(data.original_metadata.Stage.Y, 3)
            Z0 = np.round(data.original_metadata.Stage.Z, 3)
            dx = int(data.original_metadata.Microscope.DX)/1000
            dy = int(data.original_metadata.Microscope.DY)/1000
            dz = 1
        if file_nr == 1:
            dz = Z0 - np.round(data.original_metadata.Stage.Z, 3)
        a0 = data.original_metadata["Spectrum"]["CalibAbs"]  # "Zero"energy in keV
        a1 = data.original_metadata["Spectrum"]["CalibLin"]  # Channel width in keV
        fwhm = 2*np.sqrt(2*np.log(2))*np.sqrt(data.original_metadata["Spectrum"]["SigmaAbs"])
        fano = data.original_metadata["Spectrum"]["SigmaLin"]/(3.85e-3)
        gating_time = 3e-6  # time to read out one spectrum
        real_time = data.metadata["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["real_time"]
        if verbose:
            print(f"real_time from bcf - {real_time} s")
        real_time /= nr_spectra
        # deletes unused variables and clears up some memory space
        gc.collect()
        if file_nr == 0:
            ds["position dimension"] = xr.DataArray(data=[data.data.shape[0], data.data.shape[1], nr_bcf_files],
                                                    dims=("dimension"))
            ds["tensor positions"] = xr.DataArray(data=np.asarray(list(itertools.product(np.arange(ds["position dimension"][0]),
                                                                 np.arange(ds["position dimension"][1]),
                                                                 np.arange(ds["position dimension"][2]))),dtype=np.uint),
                                                  dims=("spec_nr", "dimension"),
                                                  coords={"spec_nr": np.arange(np.prod(ds["position dimension"]))})
        life_times = data.data[...,75:116].sum(axis=-1)*1e-4
        life_time_sum = life_times.mean()
        spectra[...,file_nr,:] = data.data/life_times[..., np.newaxis]
        parameters_temporary = np.array([a0, a1, fano, fwhm, life_time_sum, a0 + a1*(channels), gating_time, real_time])
        del a0, a1, fwhm, fano, channels, gating_time, data
        
        gc.collect()
        # summation over all .bcf-files
        if file_nr == 0:
            overall_real_time = real_time
            overall_life_time = life_time_sum
            parameters = parameters_temporary
        else:
            overall_real_time += real_time # sums up real time over all .bcf-files
            overall_life_time += life_time_sum # sums up life time over all .bcf-files
            parameters += parameters_temporary
    # deletes unused variables and clears up some memory space
        del parameters_temporary, life_time_sum, real_time
        gc.collect()
    # normalization over the number of bcf-files
    ds["parameters"] = xr.DataArray(data=parameters/len(bcf_file_list),
                                    dims=("parameter"),
                                    coords={"parameter": ["a0", "a1", "Fano", "FWHM",
                                                     "life_time", "max_energy",
                                                     "gating_time", "real_time"]},
                                    attrs={"units": ["keV", "keV", "a.u.", "keV",
                                                     "s", "keV", "s", "s"]})
    ds["spectra"] = xr.DataArray(data=spectra,
                                 dims=("X", "Y", "Z", "energy"),
                                 coords={"energy": np.arange(ds["parameters"][0],
                                                             ds["parameters"][5],
                                                             ds["parameters"][1]),
                                         "X": np.arange(X0, X0+dx*ds["position dimension"][0]-dx, dx),
                                         "Y": np.arange(Y0, Y0+dy*ds["position dimension"][1]-dy, dy),
                                         "Z": np.arange(Z0, Z0+dz*nr_bcf_files, dz),},
                                 attrs={"units": "counts per second"})
    ds["counts"] = xr.DataArray(data=np.ravel(ds["spectra"].sum(axis=-1)),
                                dims=("spec_nr"),
                                attrs={"units": "counts per second"})
    ds["max pixel spec"] = ds["spectra"].max(axis=(0, 1, 2))
    ds["sum spec"] = ds["spectra"].mean(axis=(0, 1, 2))
    ds["positions"] = xr.DataArray(data=ds["tensor positions"] * [dx, dy, dz] + [X0, Y0, Z0],
                                   dims=("spec_nr", "dimension"),
                                   coords={"dimension": ["x", "y", "z"]},
                                   attrs={"units": ["mm", "mm", "mm"]})

    if verbose:
        print(f"overall life_time:\t {overall_life_time} s")
        print(f"overall real_time:\t {overall_real_time} s")
    if (folder_path/"data/data.h5").exists():
        with h5py.File(folder_path/"data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
    utils.set_xarray_units(dataset=ds)
    ds.to_netcdf(
        path=folder_path/"data/data.h5",
        group=file_name,
        mode=write_operator,
        engine="h5netcdf",
        encoding=utils.create_hdf5_encoding(dataset=ds),
    )
    if return_values:
        return ds["spectra"], ds["parameters"], ds["position dimension"], ds["tensor positions"], ds["sum spec"]

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
    print("bcv_tensor_position")
    bcf_file = hs.load(file_path, lazy=True,
                       select_type="spectrum_image")
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

if __name__ == "__main__":
    folder = "C:\\Doktorarbeit\\development\\specfit\\example_measurements\\bcf\\"
    bcf2spec_para(file_path=folder+"spider.bcf")
    # many_bcf2spec_para(folder_path=folder)
