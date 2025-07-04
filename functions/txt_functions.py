
import psutil
import pickle
import natsort as ns
from pathlib import Path
import time as t
import numpy as np
import specfit_GUI_functions as sfunc

def txt2spec_para(file_path):
    """
    This function reads out the txt_files set up by 3D-measurements. It returns
    the spectrum and the parameters:
        spectrum, parameters = spx_txt2spec_para(file_path)
    """
    spectrum = []
    a0 = 0.
    a1 = 0.001
    fano = 0.115
    fwhm = 0.2
    life_time = 1
    gating_time = 3e-6
    start = 100
    fwhm = None
    with open(file_path, "r") as infile:
        for i, line in enumerate(infile):
            if "lin." in line:
                a1 = float(line.split()[-1])/1000
            elif "abs." in line:
                a0 = float(line.split()[-1])/1000
            elif "FWHM" in line:
                try:
                    fwhm = float(line.split()[1]) / 1000.0
                except:
                    fwhm = float(line.split()[-1]) / 1000.0
            elif "Fano" in line:
                try:
                    fano = float(line.split()[1])
                except:
                    fano = float(line.split()[-1])
            elif "Life time:" in line:
                life_time = float(line.split()[-1].strip()) / 1000.0
            elif "Life-Zeit" in line:
                life_time = float(line.split()[1].strip()) / 1000.0
            elif "Channels:" in line or( "Kan" in line and "le:" in line):
                channels = int(line.split()[1].strip())
            elif "Energy" in line or "Energie" in line:
                start = int(i)
            elif i == 0:
                try: float(line.split()[0])
                except: continue
                line = line.split()
                try: spectrum.append(int(line[1]))
                except: spectrum.append(int(float(line[0])))
                try:
                    line[1]
                    a0 = float(line[0])
                except: pass
                start = 0
            elif i > start:
                line = line.split()
                try: spectrum.append(int(line[1]))
                except: spectrum.append(int(float(line[0])))
                if i == 1:
                    try:
                        line[1]
                        a1 = np.round(float(line[0])-a0,3)
                    except: pass
    if fwhm is None:
        parameters = [a0, a1, 0.115, 0.2, 1, a0 + a1 * i, gating_time]
    else:
        parameters = [a0,a1,fano,fwhm, life_time, a0 + a1 * (channels-1), gating_time]
    return np.asarray(spectrum), parameters

def many_txt2spec_para(folder_path, signal):
    """
    This function reads out the spectrum of a .spx-file and reads out the
    detector parameters given in the .spx-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.
    index : int
        the number of the spectrum in the measurement set

    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    folder_path = Path(folder_path)
    file_name = folder_path.name
    Path(folder_path /"data").mkdir(parents=True, exist_ok=True)
    counts = []
    parameters = []
    spectra = {}
    file_nr = 0
    signal_progress = signal
    folder_size = 0
    machine_memory = psutil.virtual_memory().total * 1E-9
    start_time = t.time()
    sorted_folder = ns.natsorted(folder_path.glob("*.txt"))
    with open(folder_path/"data/sorted_SpecFit.dat", "w+", encoding="ISO-8859-1") as f:
        for i, _ in enumerate(sorted_folder):
            f.write(f"({i}) {sorted_folder[i]} \n")
    folder_size = sum(f.stat().st_size for f in sorted_folder if f.is_file()) * 1E-9
    if (folder_size / machine_memory) < 0.12:                              ### for machines with big memory
        print("machine memory big enough. creating spectra dict")
        for txt_file in ns.natsorted(folder_path.glob("*.txt")):
            spectrum_tmp = []
            start = 100
            with open(txt_file,"r", encoding="ISO-8859-1") as infile:
                for i, line in enumerate(infile):
                    if "lin." in line:
                        a1 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif "abs." in line:
                        a0 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif "FWHM" in line:
                        try:
                            fwhm = float(line.split()[-1]) / 1000.0
                        except:
                            fwhm = float(line.split()[1]) / 1000.0
                    elif "Fano" in line:
                        try:
                            fano = float(line.split()[-1])
                        except:
                            fano = float(line.split()[1])
                    elif "Energy" in line or "Energie" in line:
                        start = i
                    elif "Life time:" in line:
                        life_time = float(line.split()[-1].strip()) / 1000.0
                    elif "Life-Zeit" in line:
                        life_time = float(line.split()[1].strip()) / 1000.0
                    elif "Channels:" in line or ("Kan" in line and "le:" in line): #annoying problems with ä
                        channels = int(float(line.split()[1].strip()))
                    elif i > start:
                        line = line.split()
                        spectrum_tmp.append(int(float(line[1])))
            gating_time = 3e-6
            parameters_tmp = [a0, a1, fano, fwhm, life_time, a0 + a1 * (channels-1), gating_time]
            counts.append(sum(spectrum_tmp))
            spectra["%i"%file_nr] = np.divide(spectrum_tmp,life_time)
            parameters.append(parameters_tmp)
            file_nr += 1
            signal_progress.emit(file_nr)
        pickle.dump(spectra, open(f"{folder_path}/data/spectra.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        sum_spec = sfunc.sum_spec(spectra)
        max_pixel_spec = np.max(np.array(list(spectra.values())), axis=0)
        np.save(folder_path/"data/max_pixel_spec", max_pixel_spec)
        np.save(folder_path/"data/sum_spec", sum_spec)
        del spectra
    else:                                                                       #for machines with low memory
        print("machine memory to small. creating single spectra")
        Path(folder_path /"single_spectra").mkdir(parents=True, exist_ok=True)
        for spx_file in ns.natsorted(folder_path.glob("*.txt")):
            spectrum = []
            start = 100
            with open(spx_file,"r", encoding="ISO-8859-1") as infile:
                for i, line in enumerate(infile):
                    if "lin." in line:
                        a1 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif "abs." in line:
                        a0 = float(line.split()[-1])/1000 ### /1000 because standart-Unit is keV
                    elif "FWHM" in line:
                        try:
                            fwhm = float(line.split()[-1]) / 1000.0
                        except:
                            fwhm = float(line.split()[1]) / 1000.0
                    elif "Fano" in line:
                        try:
                            fano = float(line.split()[-1])
                        except:
                            fano = float(line.split()[1])
                    elif "Energy" in line or "Energie" in line:
                        start = i
                    elif "Life time:" in line or "Life-Zeit" in line:
                        life_time = float(line.split()[1].strip()) / 1000.0
                    elif "Channels:" in line or "Kanäle:" in line:
                        channels = int(float(line.split()[1].strip()))
                    elif i > start:
                        line = line.split()
                        spectrum.append(int(line[1]))
            gating_time = 3e-6
            parameters_tmp = [a0, a1, fano, fwhm, life_time, a0 + a1*(channels-1), gating_time]
            spectrum = np.divide(spectrum, life_time)
            counts.append(sum(spectrum))
            np.save(folder_path/f"single_spectra/spectrum_{file_nr}", spectrum)
            parameters.append(parameters_tmp)
            file_nr += 1
            signal_progress.emit(file_nr)
        sum_spec = sum_from_single_files(folder_path)
        np.save(folder_path/"data/sum_spec", sum_spec)
    np.save(folder_path/"data/counts", counts)
    np.save(folder_path/"data/parameters", parameters)
    print(f"txt loadingtime - {t.time() - start_time}")

def txt2energy(file_path):
    """
    This function reads out the txt_files set up by 3D-measurements. It returns
    the energy:
        energy = spx_txt2energy(file_path)
    """
    energy = []
    start = 100
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for i, line in enumerate(infile):
            if "Energ" in line:
                start = i
            elif i > start:
                line = line.split()
                energy.append(float(line[0]))
    return energy

def txt2channels(file_path):
    """
    This function reads out the txt_files set up by 3D-measurements. It returns
    the channels:
        channels = spx_txt2energy(file_path)
    """
    channels = None
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "Kan" in line:
                channels = int(line.split()[1])
                break
            elif "Channels" in line:
                channels = int(line.split()[1])
                break
    if channels is None:
        channels = 4096
    return channels

def txt2life_time(file_path):
    """
    This function reads out the txt_files set up by 3D-measurements. It returns
    the channels:
        channels = spx_txt2energy(file_path)
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "Life time:" in line or "Life-Zeit" in line:
                life_time = float(line.split()[1].strip()) / 1000.0
                break
    print( "returned life-time", life_time)
    return life_time

def spx_tensor_position(file_path):
    """
    This function reads out the specific position of the spectrum in the
    measurement-tensor.
    returns position = [x,y,z]
    position = spx_tensor_position(file_path)
    """
    position = file_path.split("(")[-1].replace(").txt","")
    position = position.split(",")
    return position

def sum_from_single_files(folder_path):
    folder_path = Path(folder_path)
    assert (folder_path/"single_spectra").exists()
    first_spec = True
    for single_spec_file in folder_path.glob("single_spectra/*.npy"):
        if first_spec is True:
            sum_spec = np.load(single_spec_file)
            first_spec = False
        else:
            sum_spec += np.load(single_spec_file)
    return sum_spec
