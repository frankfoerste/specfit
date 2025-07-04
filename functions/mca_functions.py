import os
import h5py
import psutil  # Module to determine system memory properties
import numpy as np
import time as t
from glob import iglob
from glob import glob
import matplotlib.pyplot as plt
plt.ion()

def mca2spec_para(file_path, XANES=True, print_warning=False):
    """
    This function reads out the spectrum and detector parameters given in the
    .mca-file depending on which kind of .mca file is loaded (see function
    check_mca_type).

    Parameters
    ----------
    file_path : str
        complete folder path of the .mca-file.

    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, fwhm]
    """
    # define file independent default values
    gating_time = 3e-6
    spectrum = []
    data_start = False
    # check the mca_file type
    mca_type = check_mca_type(file_path)
    if mca_type == "XPS-FP2":
        a0 = 0.5
        a1 = 0.01
        fwhm = 0.08
        fano = 0.112
        channels = 4096
        with open(file_path, "r") as infile:
            for line in infile:
                if "<<END>>" in line:
                    # end of data
                    break
                elif data_start:
                    # read out spectrum
                    spectrum.append(int(line.replace("\n", "")))
                elif "<<DATA>>" in line:
                    # data starts
                    data_start = True
                elif "LIVE_TIME" in line:
                    life_time = float(line.split("-")[-1])
                    real_time = float(line.split("-")[-1])
    elif mca_type == "measurement":
        a0 = 0.0
        a1 = 0.0092
        fwhm = 0.045
        fano = 0.148
        with open(file_path, "r", encoding="ISO-8859-1") as infile:
            for line in infile:
                if data_start:
                    # read out spectrum
                    for data in line.replace("\\", "").split():
                        spectrum.append(int(data))
                elif "@A" in line:
                    # start of the spectrum
                    for data in line.replace("@A", "").replace("\\", "").split():
                        spectrum.append(int(data))
                    # set the start of the data to true
                    data_start = True
                elif "@CTIME" in line:
                    # read out the measurement time
                    life_time = float(line.split()[2])
                    real_time = float(line.split()[-1])
                elif "@CHANN" in line:
                    # read out the channels
                    channels = int(line.split()[1])
    life_time_mca = life_time
    # determine life_time via ROI-methode. Sum over spectrum_tmp[75:116] and
    # divide by zero peak frequency (10000 per second for M4)
    # check if detector is set in 10keV, 20keV or 40keV and manipulate the
    # spectrum in order to represent a 40keV spectrum
    max_energy = a0 + a1 * (channels-1)
    parameters = [a0, a1, fano, fwhm, life_time, max_energy, gating_time, real_time]
    spectrum = np.divide(spectrum, life_time)
    if print_warning:
        if np.abs(1-life_time_mca / life_time) > 0.04:
            print(f"high difference in life_time calculation in file:\t {file_path}")
            print(f"difference:\t {np.abs(1-life_time_mca / life_time)*100:.3f}%")
            print(f"mca life-time:\t {life_time_mca:.3f} s")
            print(f"ROI life-time:\t {life_time} s")
    return np.array(spectrum), np.array(parameters)

def many_mca2spec_para(folder_path, XANES=False, signal=None ,
                       worth_fit_threshold=200,
                       save_sum_spec=True, save_spectra=True,
                       save_counts=True, save_parameters=True,
                       save_any=True, print_warning=False,
                       save_spec_as_dict=True,
                       return_values=False):
    """
    This function reads out the spectrum of a .mca-file and reads out the
    detector parameters given in the .mca-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .mca-file.
    index : int
        the number of the spectrum in the measurement set

    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, fwhm]
    """
    folder_name = folder_path.split("/")[-1]
    worth_fit = []
    counts = []
    parameters = []
    signal_progress = signal
    folder_size = 0
    a0 = 0.0
    a1 = 0.0102
    fwhm = 0.06
    fano = 0.120
    gating_time = 3e-6
    machine_memory = psutil.virtual_memory().total * 1E-9
    start = t.time()
    # derive the position of the mca-file
    if save_spec_as_dict is False:
        positions, len_scans = mca_tensor_positions(folder_path, XANES=XANES)
        x = np.unique(positions[:, 0])
        z = np.unique(positions[:, 1])
        monoE = np.unique(positions[:, 2])
        # calculate the stepsizes in every direction
        try: x_steps = x[1] - x[0]
        except: x_steps = 1
        try: z_steps = z[1] - z[0]
        except: z_steps = 1
        try: monoE_steps = monoE[1] - monoE[0]
        except: monoE_steps = 1
        # calculate the tensor positions by dividing positions by steps
        tensor_positions = np.copy(positions)
        # now subtract to 0
        tensor_positions[:, 0] -= x[0]
        tensor_positions[:, 1] -= z[0]
        tensor_positions[:, 2] -= monoE[0]
        tensor_positions /= [x_steps, z_steps, monoE_steps]
        tensor_positions = np.array(tensor_positions, dtype=int)
    sorted_folder = np.sort(glob(f"{folder_path}/*.mca"))
    try:
        os.mkdir(f"{folder_path}/data/")
    except:
        pass
    # read out ionization current from .spec file
    spec_file_path = glob("/".join(folder_path.split("/")[:-1]) + "/*.spec")[0]
    nr_scans, len_scans = read_nr_len_scans_from_spec(spec_file_path)
    scan_0, _ = [int(tmp.replace(".mca", "")) for tmp in sorted_folder[0].split("/")[-1].split("_")[-2:]]
    ionization_current = read_ionization_spec(spec_file_path, nr_scans,
                                              len_scans=len_scans,
                                              scan_0=scan_0)
    # read out spectra
    folder_size = sum(os.path.getsize(f) for f in sorted_folder if os.path.isfile(f)) * 1E-9
    life_time = False
    spectra = {}
    scan_tmp, scan_offset = 0, 0
    iterator = 0
    scans = []  #: check the number of scans in the mca path
    if (folder_size / machine_memory) < 0.7:  # for machines with big memory
        print("machine memory big enough. creating spectra dict")
        for file_nr, mca_file in enumerate(sorted_folder):
            # determine number of scan which the file belongs to
            scan, point = [int(tmp.replace(".mca", "")) for tmp in mca_file.split("/")[-1].split("_")[-2:]]
            if len_scans[f"#S {scan}"] == 0:
                continue
            file_nr = point
            for i in np.arange(scan_0, scan - 1 - scan_offset):
                file_nr += (len_scans["#S {}" % (i)] + 1)
            spectrum = []
            data_start = False
            with open(mca_file, "r", encoding="ISO-8859-1") as infile:
                for line in infile:
                    if data_start:
                        for data in line.replace("\\", "").split():
                            spectrum.append(int(data))
                    elif "@A" in line:
                        for data in line.replace("@A", "").replace("\\", "").split():
                            spectrum.append(int(data))
                        data_start = True
                    elif "@CTIME" in line:
                        life_time = float(line.split()[2])
                        real_time = float(line.split()[-1])
                    elif "@CHANN" in line:
                        channels = int(line.split()[1])
            if len(spectrum) == 0:
                print(f"mca_file {mca_file} corrupted")
                continue
            if np.abs(np.subtract(scan, scan_tmp)) > 1:
                scan_offset += 1
                scan -= 1
            scan_tmp = int(scan)
            if scan not in scans:
                scans.append(scan)
            spectrum = np.asarray(spectrum)
            max_energy = a0 + a1 * (channels - 1)
            parameters.append([a0, a1, fano, fwhm, life_time, max_energy, gating_time, real_time])
            spectrum = np.divide(spectrum, life_time)
            # norm spectrum to ionization current
            spectrum = np.divide(spectrum, ionization_current[str(scan-1)][point])
            sum_spec_tmp = sum(spectrum)
            counts.append(sum_spec_tmp)
            if sum_spec_tmp > worth_fit_threshold:
                worth_fit.append(True)
            else:
                worth_fit.append(False)
            # now we try to rebuild specfit load in routine to support array
            # spectra
            if save_spec_as_dict:
                spectra[f"{iterator}"] = spectrum
            else:
                if file_nr == 0:
                    spectra = np.empty((len(x), len(z), len(monoE), 4096))
                x_pos, z_pos, monoE_pos = tensor_positions[file_nr]
                spectra[x_pos][z_pos][monoE_pos] = spectrum
            if signal_progress is not None:
                signal_progress.emit(file_nr)
            iterator += 1
        with h5py.File(f"{folder_path}/data/data.h5", "w") as tofile:
            tofile.create_dataset(f"{folder_name}/spectra",
                                  data=np.array(list(spectra.values())),
                                  compression="gzip", compression_opts=9)
        max_pixel_spec = np.max(np.array(list(spectra.values())), axis=0)
        sum_spec = calc_sum_spec(spectra)
    else:  # for machines with low memory
        print("machine memory to small. get a better machine ASAP!")
    with h5py.File(f"{folder_path}/data/data.h5", "a") as tofile:
        tofile.create_dataset(f"{folder_name}/sum spec", data=sum_spec,
                              compression="gzip")
        tofile.create_dataset(f"{folder_name}/counts", data=counts,
                              compression="gzip")
        tofile.create_dataset(f"{folder_name}/parameters", data=parameters,
                              compression="gzip")
        tofile.create_dataset(f"{folder_name}/max pixel spec", data=max_pixel_spec,
                              compression="gzip")
    print(f"mca loadingtime - {t.time() - start}")
    if return_values:
        if save_spec_as_dict is False:
            return spectra, parameters, positions, tensor_positions
        else:
            return spectra, parameters

def read_ionization_spec(spec_file_path, nr_scans, len_scans, scan_0=1):
    ionization_current = {}
    for la, _ in enumerate(len_scans):
        ionization_current[str(la)]=[]
    read_line = False
    with open(spec_file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "#S" in line:
                # sometimes the scan has no values stored, blind value
                scan = " ".join(line.split()[:2])
            if "#L" in line:
                try: where = line.split().index("AS_IC")
                except: where = line.split().index("ionch1")
                nr_variables = len(line.split())-1
                read_line = True
                continue
            elif "#C" in line:
                read_line = False
                continue
            elif line == "\n":
                read_line = False
                continue
            if read_line:
                if len(line.split()) != nr_variables: continue
                if len_scans[scan] != 0:
                    ## edit: AS_IC is in units of 10e-10, nomalize here
                    ionization_current[str(int(scan.split()[-1])-1)].append(float(line.split()[where-1])/10E-10)
    for scan in ionization_current:
        if scan == []:
            ionization_current.remove([])
            nr_scans -= 1
    return ionization_current #norm_ion_cur  ## edit: nicht auf 1 normieren, sondern auf 10e-10, s. Z 271

def read_nr_len_scans_from_spec(spec_file_path):
    nr_scans = 0
    len_scans = {}
    current_scan_nr = ""
    with open(spec_file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "#S" in line:
                line = line.split()
                current_scan_nr = " ".join(line[:2])
                if line[2] == "acquire":
                    len_scans[current_scan_nr]=1
                if line[2] == "eigerloopscan":
                    len_scans[current_scan_nr]=1
                else:
                    len_scans[current_scan_nr] = int(line[-2])
                nr_scans += 1
                continue
            elif "aborted after" in line:
                len_scan = int(line.split()[-2])
                len_scans[current_scan_nr] = len_scan
                if len_scan == 0:
                   nr_scans -= 1
    return nr_scans, len_scans

def sum_from_single_files(folder_path, save_sum_spec=True):
    try:
        os.mkdir(f"{folder_path}/data/")
    except:
        pass
    first_spec = True
    for single_spec_file in iglob(f"{folder_path}/single_spectra/*.npy"):
        if first_spec is True:
            sum_spec = np.load(single_spec_file)
            first_spec = False
        else:
            sum_spec += np.load(single_spec_file)
    if save_sum_spec:
        np.save(f"{folder_path}/data/sum_spec", sum_spec)
    return sum_spec

def mca2life_time(file_path):
    """
    This function reads out the life-time in seconds of a .mca-file.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<LifeTime>" in line:
                life_time = int(line.replace("<LifeTime>", "").replace("</LifeTime>", "").replace("\r\n", "").replace("    ", ""))
                life_time /= 1000.0
                break
            if "<RealTime>" in line:
                life_time = int(line.replace("<RealTime>", "").replace("</RealTime>", "").replace("\r\n", "").replace("    ", ""))
                life_time /= 1000.0
                break
    return life_time

def mca_metadata(file_path):
    """
    Function to read out the meta data in a .mca file

    Parameters
    ----------
    file_path : str
        absolute path ot the .mca file.

    Returns
    -------
    dictionary with meta-data field

    """
    start = False
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "#O0" in line:
                start = True
                metadata_names = []
            if start is True:
                if "#O" in line:
                    for _ in line.replace("\n", "").split()[1:]:
                        metadata_names.append(_)
                else:
                    start = False
                    break
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "#P0" in line:
                start = True
                metadata = []
            if start is True:
                if "#P" in line:
                    for _ in line.replace("\n", "").split()[1:]:
                        metadata.append(float(_))
                else:
                    start = False
                    break
    return dict(zip(metadata_names, metadata))

def mca2real_time(file_path):
    """
    This function reads out the real-time in seconds of a .mca-file.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<RealTime>" in line:
                real_time = int(line.replace("<RealTime>", "").replace("</RealTime>", "").replace("\r\n", "").replace("    ", ""))
                real_time /= 1000.0
                break
    return real_time

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

def mca2channels(file_path):
    """
    This function reads out the channe numbers.
    """
    mca_type = check_mca_type(file_path)
    if mca_type == "XPS-FP2":
        return 4096
    elif mca_type == "measurement":
        with open(file_path, "r", encoding="ISO-8859-1") as infile:
            for line in infile:
                if "@CHANN" in line:
                    channels = int(line.split()[1])
                    break
        return channels

def mca_tensor_position(file_path, XANES=False):
    """
    This function reads out the specific position of the spectrum in the
    measurement-tensor.
    returns position = [x, y, z]
    position = mca_tensor_position(file_path, XANES = bool)
    It determines whether it is a line scan or a 3D-Scan.
    If XANES = True, the position is read out from the monoE parameter
    """
    file_name = file_path.split("/")[-1].split("_")
    point = int(file_name[-1].replace(".mca", ""))
    if file_path[-4:] == ".mca":
        metadata = mca_metadata(file_path)
        with open(file_path, "r", encoding="ISO-8859-1") as infile:
            for line in infile:
                if "#S" in line:
                    lines = line.replace("\n", "").split()
                    # read out the mode of measurement and the coressponding motor and steps
                    mode = lines[2]
                    motors = lines[3:-1][::4]
                    print(f"mode {mode} motors {motors}")
                    ## either #S 1  eigermesh  m1 61.8 64.44 33  mz 22.59 22.91 4  10
                    ## or #S 1 ascan m1 15 17 20 1
                    ## ascan can be dscan also, Motorname is the one after
                    if mode == "eigermesh":
                        x = metadata[motors[0]]
                        y = metadata[motors[1]]
                        z = 1
                    elif mode in ["ascan", "dscan"]:
                        x = metadata[motors[0]]
                        y = 1
                        z = 1
                    elif mode in ["acquire"]:
                        x = 1
                        y = 1
                        z = 1
                    return [x, y, z]

def mca_tensor_positions(folder_path, file_type=".mca", XANES=False):
    """
    This function reads out the tensor position of all mca/txt-files in the
    given folder_path

    Parameters
    ----------
    folder_path: str
        path of the folder containing the mca or txt files
    file_type : str
        type of the file
    XANES : bool
        if true, monoE is used as z coordinate

    Returns
    -------
    positions = np.array([x0, y0, z0], [x0, y1, z0], ..., [xn, ym, zk])
    """
    if isinstance(folder_path, str):
        files = sorted(glob(folder_path + "*" + file_type))
    elif isinstance(folder_path, list):
        files = folder_path
    nr_scans = np.unique([int(file.split("/")[-1].split("_")[-2]) for file in files])
    len_scans = np.zeros(len(nr_scans), dtype = int)
    for file in files:
        len_scans[int(file.split("/")[-1].split("_")[-2])-nr_scans[0]] += 1
    positions = []
    for file in files:
        positions.append(mca_tensor_position(file, XANES=XANES))
    return positions, len_scans

def convert_string(string):
    string = string.replace(",", ".")
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
    sum_spec = np.divide(values.sum(axis=0), nr_arrays)
    return sum_spec

def check_mca_type(mca_file_path):
    """
    Function to decide what kind of .mca file is provided. Up to date only
    measurement files from beamlines and simulation files generated by the
    software XPS-FP2 are supported and destinguished.

    Parameters
    ----------
    mca_file_path : str
        Absolute path to the .mca file.

    Returns
    -------
    str
        description of the file, either 'XPS-FP2' as coming from this software
        or 'measurment' coming from beamlines
    """
    with open(mca_file_path, "r") as mca_file:
        for line in mca_file:
            if "PMCA SPECTRUM" in line:
                return "XPS-FP2"
            else:
                return "measurement"
