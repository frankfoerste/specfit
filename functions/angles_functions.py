import sys
import collections
import os
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.ioff()

def _cast_to_int(x):
    try:
        new_int = int(_cast_to_float(x))
    except:
        print("some error with casting of Data - maybe empty String?")
        sys.exit()
    return new_int

def _cast_to_float(x):
    x= x.strip()
    if "e" in str(x):
        new_float = float(x.split("e")[0])*10**int((float(x.split("e")[1])))
    else:
        new_float = float(x)
    return new_float

def txt2spec_para(file_path):
    """
    This function reads out the spectrum of a .txt-file and reads out the
    detector parameters given in the .txt-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .txt-file.

    Returns
    -------
    dict containing the spectrum
    list containing hard coded detector parameters
    """
    file_name = file_path.split("/")[-1]
    tfile = open(file_path, "r")
    file_content = tfile.readlines()
    angles = []
    spectra_tmp = []
    for i, anglespec in enumerate(file_content):
        anglespec = anglespec.strip('\n').strip().split(' ')
        cur_angle = f'{_cast_to_float(anglespec[0])}'
        angles.append(cur_angle)
        help_anglespec = (anglespec[1:])  # remove angle
        spectra_tmp.append([])
        for x in help_anglespec:
            spectra_tmp[i].append(_cast_to_float(x))
    tfile.close()
    # split spectra into portions
    spectra = collections.OrderedDict()
    for i, angle in enumerate(angles):
        spectra[angle] = np.array(spectra_tmp[i])
    parameters = [0.04369487, 0.0090333623, 0.4927366167856802, 0.074025388749522497, 1, 0,280e-9, 1]
    parameters[5] = np.add(np.multiply(len(spectra_tmp[i])-1,parameters[1]),parameters[0])
    print(np.array(list(spectra.values())).shape)
    sum_spec = np.sum(np.array(list(spectra.values())), axis=0)
    positions = txt2positions(file_path=file_path)
    x = len(np.unique(positions[:, 0]))
    y = len(np.unique(positions[:, 1]))
    z = len(np.unique(positions[:, 2]))
    pos_dim = np.array([x, y, z])
    tensor_positions = np.zeros(positions.shape, dtype=int)
    tensor_positions[:, 2] = np.arange(z, dtype=int)
    folder_path = "/".join(file_path.split("/")[:-1])
    Path(f"{folder_path}/data/").mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"{folder_path}/data/data.h5"):
        with h5py.File(f"{folder_path}/data/data.h5", "r+") as tofile:
            if file_name in tofile.keys():
                del tofile[file_name]
        write_operator = "r+"
    else:
        write_operator = "w"
    with h5py.File(f"{folder_path}/data/data.h5", write_operator) as tofile:
        tofile.create_dataset(f"{file_name}/spectra",
                                data=np.array(list(spectra.values())),
                                compression="gzip", compression_opts=9)
        tofile.create_dataset(f"{file_name}/sum spec", data=sum_spec,
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/counts", data=[np.sum(np.array(list(spectra.values())), axis=-1)],
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/parameters", data=parameters,
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/max pixel spec", data=np.max(np.array(list(spectra.values())), axis=0),
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/position dimension", data=pos_dim,
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/tensor positions", data=tensor_positions,
                                compression="gzip")
        tofile.create_dataset(f"{file_name}/positions", data=positions,
                                compression="gzip")
    return spectra, parameters, tensor_positions, pos_dim, sum_spec

def txt2channels(file_path):
    """
    This function reads out the number of channels of a .txt-file .
    """
    tfile = open(file_path,"r")
    channels = len(tfile.readline().split()) -1  # remove angle information
    tfile.close()
    return channels

def txt2positions(file_path):
    tfile = open(file_path, "r")
    ticks = []
    for line in tfile:
        string_tick = line.split(" ")[0]
        ticks.append(_cast_to_float(string_tick))
    return np.array(np.meshgrid(1,1,ticks)).T.reshape(-1,3)

def txt2num_spec(file_path):
    tfile = open(file_path, "r")
    lines = tfile.readlines()
    return len(lines)

def plot_spectra_for_angle(energy,spectra,bg,fit,angle,save_folder_path):
    """
    plot and save an image with spectra,background and spectra-fit
    """
    angle = _cast_to_float(angle)
    if not os.path.exists(save_folder_path +"/ang_images"):
        os.mkdir(save_folder_path +"/ang_images")
    savepath = save_folder_path +"/ang_images/fit_plot_{0:0.4f}.png".format(angle)
    fit_with_bg = np.add(fit,bg)
    fig,ax =plt.subplots()
    try:
        plt.yscale("log")
        ax.plot(energy,spectra, "b.")
        ax.plot(energy,bg,"g-")
        ax.plot(energy,fit_with_bg,"r-")
    except:
        plt.yscale("linear")
        ax.plot(energy,spectra, "b.")
        ax.plot(energy,bg,"g-")
        ax.plot(energy,fit_with_bg,"r-")
    plt.xlabel = "Energy in keV"
    plt.ylabel = "intenity/arb. un."
    plt.savefig(savepath)
    plt.close(fig)

def save_numpy_arrays(energy,spectra,bg,fit,angle,save_folder_path):
    angle = _cast_to_float(angle)
    fit_with_bg = np.add(fit,bg)
    if not os.path.exists(save_folder_path +"/ang_npz"):
        os.mkdir(save_folder_path +"/ang_npz")
    outfile = save_folder_path +"/ang_npz/en_sp_bg_fit_{0:0.4f}.png".format(angle)
    np.savez(outfile,energy,spectra,bg,fit_with_bg)

def plot_angle_line_intensities(results,angles,savepath,n=0):
    """
    plot for every result-line the angles on x-axis and the intenity on x axis
    cut n angles on both sides -cause they look ugly
    Needs:
        results - list of all results dim: numberlines x numberspecptra
        angles -list of all angles
        savepath - path of allsaved.txts is used to save the images
        n - cut off the n first and last angles
    """
    # do not cut more than you have
    if n >= len(angles)/2:
        print(f"ERROR, you are cutting 2*{n} angles in the plot! Thats more than you have... n is set zu 0.")
        n = None
    else:
        # cut the first and the last
        angles_cut = angles[n:-n]
    # cast all to floast
    angles_cut = np.vectorize(_cast_to_float)(angles_cut)
    angles = np.vectorize(_cast_to_float)(angles)
    for r,sp in zip(results,savepath):
        r = np.array(r)
        # plot and save it!
        try:
            fig,ax =plt.subplots()
            ax.plot(angles,r,linestyle = "None",marker = ".")
            plt.xlabel = "emission angle /°"
            plt.ylabel = "intenity/(photons x sr^(-1))"
            plt.savefig(sp+".png")
            plt.close()
        except:  # most likely an dimension error
            print("ERROR")
            print(angles,r)
            print(len(angles), len(r))
