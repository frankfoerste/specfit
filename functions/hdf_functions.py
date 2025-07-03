import os
import h5py
import itertools
import numpy as np
import dask.array as da
from pathlib import Path

def hdf2spec_para(file_path, verbose=False):
    """ 
    This function reads out the spectrum of a .hdf5-file and reads out the
    detector parameters given in the .hdf5-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .hdf5-file.
    
    Returns
    -------
    array containing the spectrum
    array containing the detector parameters [a0, a1, FANO, FWHM]
    """
    file_type = file_path.split(".")[-1]
    file_name = file_path.split("/")[-1]
    save_path = "/".join(file_path.split("/")[:-1])
    Path(f"{save_path}/data/").mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "r") as hdf5_file:
        if file_type == "h5":
            if "concentrations" in hdf5_file.keys():
                spectra = da.from_array(hdf5_file["spectra"][()])
                sum_spec = np.sum(hdf5_file["spectra"], axis=(0, 1))
                counts = np.sum(spectra, axis=-1)
                max_pixel_spec = np.max(spectra, axis=(0, 1))
                a0 = 2.
                a1 = 0.01
                Fano = 0.112
                FWHM = 0.045
                gating_time = 1e4  # zeropeakperiod
                life_time = np.ones(spectra.shape[:-1])
                real_time = life_time
                channels = spectra.shape[-1]
                spectra = np.divide(spectra, life_time[..., np.newaxis])
            else:
                try:
                    try:
                        sdd_name = "BRQM1:mca03"  # Quad detector
                        pixels = sorted(hdf5_file[f"c1/main/{sdd_name}chan1"].keys(), key=int)
                    except KeyError:
                        sdd_name = "BRQM1:mca08"
                        pixels = sorted(hdf5_file[f"c1/main/{sdd_name}chan1"].keys(), key=int)
                    spectra = np.zeros((len(pixels), 4096))
                    for i, pixel in enumerate(pixels):
                        spectra[i, :] = hdf5_file[f"c1/main/{sdd_name}chan1/{pixel}"][:, 0]
                    spectra = da.from_array(spectra)
                    sum_spec = np.sum(spectra, axis=0)
                    max_pixel_spec = np.max(spectra, axis=0)
                    counts = np.sum(spectra, axis=1)
                    a0 = -0.2316
                    a1 = 0.0024827
                    Fano = 0.109353
                    FWHM = 0.0317839
                    gating_time = 1e3
                    try:
                        life_time = hdf5_file[f"c1/main/{sdd_name}.R0"][f"{sdd_name}.R0"] / (gating_time * 4)
                    except KeyError:
                        # otherwise find at .ELTM
                        life_time = hdf5_file[f"c1/main/{sdd_name}.ELTM"][f"{sdd_name}.ELTM"] / (gating_time * 4)
                    real_time = hdf5_file[f"c1/main/{sdd_name}.ERTM"][f"{sdd_name}.ERTM"]

                    channels = int(len(spectra[0, :]))
                    ringcurrent = hdf5_file["c1/main/bIICurrent:Mnt1chan1"]["bIICurrent:Mnt1chan1"]
                    spectra = spectra / life_time[:spectra.shape[0], None]
                    sum_spec = np.divide(sum_spec, life_time.sum())
                    if verbose:
                        print("pixel", pixels)
                        print(
                            f"Life time: {life_time}, Real time: {real_time} \nDeadtime %: {1 - (life_time / real_time)}")
                except KeyError:
                    spectra = hdf5_file["Raw"][()]
                    shape = spectra.shape
                    if len(shape) == 3:
                        spectra = np.reshape(spectra, (shape[0] * shape[1], shape[2]))
                    else:
                        spectra = np.reshape(spectra, (shape[0] * shape[1] * shape[2], shape[3]))
                    sum_spec = hdf5_file["Spectrum"][()]
                    counts = np.sum(spectra, axis=-1)
                    max_pixel_spec = np.max(spectra, axis=0)
                    parameter = hdf5_file["Header"][()]
                    h, m, s = parameter[0][2].decode("ascii").split(":")

                    a0 = parameter[0][12] / 1000
                    a1 = parameter[0][13] / 1000
                    Fano = parameter[0][15]
                    FWHM = 0.14
                    life_time = int(h) * 3600 + int(m) * 60 + int(s)
                    real_time = int(h) * 3600 + int(m) * 60 + int(s)
                    spectra = np.divide(spectra, life_time)  # norm the measurement to life_time
                    sum_spec = np.divide(sum_spec, life_time)
                    channels = parameter[0][19]
                    gating_time = 3e-6
        elif file_type == "hdf5":
            spectra = da.from_array(hdf5_file["SDD/measurement/detector"])
            sum_spec = np.squeeze(hdf5_file["SDD/sum spectra/sumS"], axis=1)
            counts = np.sum(spectra, axis=-1)
            max_pixel_spec = hdf5_file["SDD/maximum pixel spectrum/mps"][()]
            a0 = -0.235825
            a1 = 0.002515
            Fano = 0.112
            FWHM = 0.045
            gating_time = 1e-6  # zeropeakperiod
            life_time = np.sum(spectra[:, :125], axis=-1) / (gating_time * 4)
            real_time = 1
            channels = int(len(spectra[0]))
            spectra = np.divide(spectra, life_time[:, None])
        parameters = np.array([a0, a1, Fano, FWHM, np.mean(life_time), 
                               a0 + a1 * (channels - 1), gating_time, 
                               np.mean(real_time)])
        if verbose:
            print("parameters", parameters.shape)
        if os.path.exists(f"{save_path}/data/data.h5"):
            with h5py.File(f"{save_path}/data/data.h5", "r+") as tofile:
                if file_name in tofile.keys():
                    del tofile[file_name]
            write_operator = "r+"
        else:
            write_operator = "w"
        with h5py.File(f"{save_path}/data/data.h5", write_operator) as tofile:
            tofile.create_dataset(f"{file_name}/max pixel spec", data=max_pixel_spec, 
                                  compression="gzip")
            tofile.create_dataset(f"{file_name}/sum spec", data=sum_spec, 
                                  compression="gzip")
            tofile.create_dataset(f"{file_name}/parameters", data=parameters, 
                                  compression="gzip")
        da.to_hdf5(f"{save_path}/data/data.h5", {f"{file_name}/counts": counts}, 
                   compression="gzip")
        da.to_hdf5(f"{save_path}/data/data.h5", {f"{file_name}/spectra": spectra}, 
                   compression="gzip")
    return spectra, parameters, sum_spec, channels

def hdf_tensor_positions(file_path):
    """
    This function reads out the specific position of the spectrum in the 
    measurement-tensor.
    returns position = [x, y, z]
    position = spx_tensor_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    """
    file_type = file_path.split(".")[-1]
    file_name = file_path.split("/")[-1]
    save_path = "/".join(file_path.split("/")[:-1])
    hdf5_file = h5py.File(file_path, "r")
    if file_type == "hdf5": # For AnImaX mit python ansteuerung
        line_breaks = hdf5_file["SDD/scan index log/line breaks"][()]
        positions = [line_breaks[-2, 2] + 1, line_breaks[1, 1], 1]
        tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                             range(positions[1]))), 
                                      dtype=np.uint32)
        tensor_positions = np.hstack((tensor_positions, 
                                      np.zeros((len(tensor_positions), 1), dtype=np.uint32)))
    elif file_type == "h5":
        if "c1/main" in hdf5_file:
            for M in ("NEXAFS", "SmarM", "PI", "CounterSpec"):
                if M == "NEXAFS":
                    try:
                        axis1 = hdf5_file["c1/main/Energ:io0200Energy"].shape[0] / 2  # NEXAFS Energies
                        axis2 = 1
                        break
                    except KeyError:
                        continue
                if M == "SmarM":
                    try:
                        axis2 = hdf5_file["c1/main/SmarM:smaract0800001"].shape[0]  # PR_Y
                    except KeyError:
                        axis2 = 1
                    try:
                        axis1 = hdf5_file["c1/main/SmarM:smaract0800000"].shape[0]  # PR_X
                        break
                    except KeyError:
                        continue
                if M == "PI":
                    try:
                        axis2 = hdf5_file["c1/main/PISMC:animax0104002"].shape[0]  # PR_Y
                    except KeyError:
                        axis2 = 1
                    try:
                        axis1 = hdf5_file["c1/main/PISMC:animax0104000"].shape[0]  # PR_X
                        break
                    except KeyError:
                        continue
                if M == "CounterSpec":
                    try:
                        axis1 = hdf5_file["c1/main/Counter-mot"].shape[0]  # counter dummy
                    except KeyError:
                        axis1 = 1
                    axis2 = 1
            if axis1 % axis2 == 0:
                width = axis2
                height = axis1 // axis2
            elif axis2 % axis1 == 0:
                width = axis2 // axis1
                height = axis1
            else:  # scan stopped prematurely, treat as line scan
                width = max(axis1, axis2)
                height = 1
            positions = [int(height), int(width), 1]
            tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                                 range(positions[1]))), 
                                          dtype=np.uint32)
            tensor_positions = np.hstack((tensor_positions, 
                                          np.zeros((len(tensor_positions), 1), dtype=np.uint32)))
        elif "concentrations" in hdf5_file:
            shape = hdf5_file["spectra"].shape
            if len(shape) == 3:
                positions = [shape[0], shape[1], 1]
                tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                                     range(positions[1]), 
                                                                     range(positions[2]))), 
                                              dtype=np.uint32)
            else:
                positions = list(shape[:-1])
                tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                                     range(positions[1]), 
                                                                     range(positions[2]))), 
                                              dtype=np.uint32)
        else:
            shape = hdf5_file["Raw"].shape
            if len(shape) == 3:
                positions = [shape[0], shape[1], 1]
                tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                                     range(positions[1]))), 
                                              dtype=np.uint32)
                tensor_positions = np.hstack((tensor_positions, 
                                              np.zeros((len(tensor_positions), 1), 
                                                       dtype=np.uint32)))
            else:
                positions = list(shape[:-1])
                tensor_positions = np.asarray(list(itertools.product(range(positions[0]), 
                                                                     range(positions[1]), 
                                                                     range(positions[2]))), 
                                              dtype=np.uint32)
    with h5py.File(f"{save_path}/data/data.h5", "r+") as tofile:
        tofile.create_dataset(f"{file_name}/tensor positions", data=tensor_positions, 
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/positions", data=tensor_positions, 
                              compression="gzip")
        tofile.create_dataset(f"{file_name}/position dimension", data=positions, 
                              compression="gzip")
    hdf5_file.close()
    return positions, tensor_positions
