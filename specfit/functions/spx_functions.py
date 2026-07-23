import numpy as np
import time as t
import xarray as xr
from glob import iglob
import codecs
import natsort as ns
from pathlib import Path
from specfit.functions import utils


def spx2spec_para(
        file_path,
        return_values=False
        ):
    """
    This function reads out the spectrum of a .spx-file and reads out the
    detector parameters given in the .spx-file.

    Parameters
    ----------
    file_path : str
        complete folder path of the .spx-file.

    Returns
    -------
    list containing the spectrum
    list containing the detector parameters [a0, a1, FANO, FWHM]
    """
    file_path = Path(file_path)
    folder_path = file_path.parent
    Path(folder_path/"data").mkdir(parents=True, exist_ok=True)
    file_name = file_path.name
    write_operator = utils.get_hdf5_write_operator(
        hdf5_file=folder_path/"data/data.h5",
        file_name=file_name)

    # define default values
    a0 = -0.96
    a1 = 0.01
    FWHM = 0.0792
    Fano = 0.113
    channels = False
    gating_time = 3e-6
    X0, Y0, Z0 = spx_position(file_path=file_path)
    ds = xr.Dataset()
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<Channels>" in line:
                spectrum = line.split(">")[1].split("<")[0].split(",")
                break
            elif "<CalibAbs>" in line:
                a0 = float(line.split(">")[1].split("<")[0])
            elif "<CalibLin>" in line:
                a1 = float(line.split(">")[1].split("<")[0])
            elif "<ZeroPeakPosition>" in line:
                zero_peak_position = int(line.split(">")[1].split("<")[0])
            elif "<ZeroPeakFrequency>" in line:
                zero_peak_frequency = int(line.split(">")[1].split("<")[0])
            elif "<SigmaAbs>" in line:
                FWHM = float(line.split(">")[1].split("<")[0])
                FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(FWHM)
            elif "<SigmaLin>" in line:
                # 3.85eV is the mean energy required to create an electon-
                # hole-pair
                Fano = float(line.split(">")[1].split("<")[0]) / (3.85e-3)
            elif "<LifeTime>" in line:
                life_time = int(line.split(">")[1].split("<")[0]) / 1000.0
            elif "<ZeroPeakFrequency>" in line:
                zero_peak_frequency = int(line.split(">")[1].split("<")[0])
            elif "<RealTime>" in line:
                real_time = int(line.split(">")[1].split("<")[0]) / 1000.0
            elif "<ChannelCount>" in line:
                channels = int(line.split(">")[1].split("<")[0])
            elif "<PulsePairResTimeCount>" in line:
                gating_time = int(line.split(">")[1].split("<")[0]) * 1e-6
                if gating_time == 0.0:
                    gating_time = 3e-6
    if channels is False:
        channels = len(spectrum)
    try:
        spectrum = [int(intensity) for intensity in spectrum]
    except ValueError:
        print("spectrum modified, containing float numbers!")
        spectrum = [float(intensity) for intensity in spectrum]
    # determine life_time via ROI-methode. Sum over spectrum_tmp[75:116] and
    # divide by zero peak frequency (10000 per second for M4)
    # check if detector is set in 10keV, 20keV or 40keV and manipulate the
    # spectrum in order to represent a 40keV spectrum
    max_energy = a0 + a1 * (channels-1)
    factor = (1/a1)/100
    life_time = np.sum(
        spectrum[zero_peak_position - int(20*factor):
                 zero_peak_position + int(20*factor)]) / zero_peak_frequency
    if life_time == 0:
        print(f"error in {file_path}")
        print("life_time is ZERO")
        print("setting life_time to 1")
        life_time = 1
    spectrum = np.divide(spectrum, life_time)
    ds["position dimension"] = xr.DataArray(
        [1, 1, 1],
        dims=("dimension"))
    ds["tensor positions"] = xr.DataArray(
        [[0, 0, 0]],
        dims=("spec_nr", "dimension"),)
    ds["positions"] = xr.DataArray(
        data=[[X0, Y0, Z0]],
        dims=("spec_nr", "dimension"),
        coords={"dimension": ["x", "y", "z"]},
        attrs={"units": ["mm", "mm", "mm"]})
    ds["parameters"] = xr.DataArray(
        data=[[
            a0,
            a1,
            Fano,
            FWHM,
            life_time,
            max_energy,
            gating_time,
            real_time]],
        dims=("spec_nr", "parameter"),
        coords={"parameter": [
            "a0",
            "a1",
            "Fano",
            "FWHM",
            "life_time",
            "max_energy",
            "gating_time",
            "real_time"]},
        attrs={"units": [
            "keV",
            "keV",
            "a.u.",
            "keV",
            "s",
            "keV",
            "s",
            "s"]})
    ds["spectra"] = xr.DataArray(
        data=spectrum.reshape((1, 1, 1, channels)),
        dims=("X", "Y", "Z", "energy"),
        coords={
            "energy": np.arange(
                ds["parameters"][0][0],
                ds["parameters"][0][5] + ds["parameters"][0][1],
                ds["parameters"][0][1]),
            "X": np.arange(X0, X0 + 0.1),
            "Y": np.arange(Y0, Y0 + 0.1),
            "Z": np.arange(Z0, Z0 + 0.1)},
        attrs={"units": "counts per second"})
    ds["counts"] = xr.DataArray(
        ds["spectra"].sum(axis=-1),
        dims=("X", "Y", "Z"),
        attrs={"units": "counts per second"})
    ds["max pixel spec"] = ds["spectra"].max(axis=(0, 1, 2))
    ds["sum spec"] = ds["spectra"].mean(axis=(0, 1, 2))

    # set units to xarray Dataset
    utils.set_xarray_units(dataset=ds)

    # save the hdf5
    ds.to_netcdf(
        path=folder_path/"data/data.h5",
        group=file_name,
        mode=write_operator,
        engine="h5netcdf",
        encoding=utils.create_hdf5_encoding(dataset=ds),
    )
    if return_values:
        return (
            ds["spectra"],
            ds["parameters"]
        )


def many_spx2spec_para(
        folder_path,
        signal=None,
        save_spec_as_dict=True,
        return_values=False
        ):
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
    Path(folder_path / "data").mkdir(parents=True, exist_ok=True)
    write_operator = utils.get_hdf5_write_operator(
        hdf5_file=folder_path / "data" / "data.h5",
        file_name=file_name)

    # assign default values
    zero_peak_frequency = 1e4
    signal_progress = signal
    zero_peak_position = 96
    zero_peak_frequency = 10000

    # create an empty xarray Dataset to fill in read out data
    ds = xr.Dataset()
    start = t.time()
    # derive the position of the spx-file
    if save_spec_as_dict is False:
        positions = spx_positions(folder_path)

        # read out the x, y, z axes
        x = np.unique(positions[:, 0])
        y = np.unique(positions[:, 1])
        z = np.unique(positions[:, 2])

        # calculate the stepsizes in every direction
        try:
            x_steps = x[1] - x[0]
        except IndexError:
            x_steps = 1
        try:
            y_steps = y[1] - y[0]
        except IndexError:
            y_steps = 1
        try:
            z_steps = z[1] - z[0]
        except IndexError:
            z_steps = 1
        # calculate the tensor positions by dividing positions by steps
        tensor_positions = np.copy(positions)
        # now subtract to 0
        tensor_positions[:, 0] -= x[0]
        tensor_positions[:, 1] -= y[0]
        tensor_positions[:, 2] -= z[0]
        tensor_positions /= [x_steps, y_steps, z_steps]
        tensor_positions = np.array(tensor_positions, dtype=int)
    sorted_folder = ns.natsorted(folder_path.glob("*.spx"))
    ds["parameters"] = xr.DataArray(
        np.zeros((len(sorted_folder), 8)),
        dims=("spec_nr", "parameter"),
        coords={"parameter": [
            "a0",
            "a1",
            "Fano",
            "FWHM",
            "life_time",
            "max_energy",
            "gating_time",
            "real_time"]},
        attrs={"units": [
            "keV",
            "keV",
            "a.u.",
            "keV",
            "s",
            "keV",
            "s",
            "s"]})

    # read out the positions of the measurement folder
    ds["positions"] = xr.DataArray(
        spx_positions(folder_path=(folder_path/"")),
        dims=("spec_nr", "dimension"),
        coords={"dimension": ["x", "y", "z"]},
        attrs={"units": ["mm", "mm", "mm"]})

    x = len(np.unique(ds["positions"][:, 0]))
    y = len(np.unique(ds["positions"][:, 1]))
    z = len(np.unique(ds["positions"][:, 2]))
    ds["position dimension"] = xr.DataArray(
        [x, y, z],
        dims=("dimension"))
    ds["tensor positions"] = ds["positions"].copy()
    for i in range(3):
        ds["tensor positions"][:, i] = (
            ds["tensor positions"][:, i]-ds["tensor positions"][:, i].min()
        )
        if ds["position dimension"][i] > 1:
            _unique = np.unique(ds["tensor positions"][:, i])
            ds["tensor positions"][:, i] /= _unique[1]
    ds["tensor positions"] = ds["tensor positions"].astype(int)

    # read out the files
    life_time = False
    for file_nr, spx_file in enumerate(sorted_folder):
        with open(spx_file, "r", encoding="ISO-8859-1") as infile:
            for line in infile:
                if "<LifeTime>" in line:
                    life_time = int(line.split(">")[1].split("<")[0]) / 1000.0
                elif "<RealTime>" in line:
                    real_time = int(line.split(">")[1].split("<")[0]) / 1000.0
                elif "<ZeroPeakPosition>" in line:
                    zero_peak_position = int(line.split(">")[1].split("<")[0])
                elif "<ZeroPeakFrequency>" in line:
                    zero_peak_frequency = int(line.split(">")[1].split("<")[0])
                elif "<PulsePairResTimeCount>" in line:
                    gating_time = int(line.split(">")[1].split("<")[0]) * 1e-6
                    if gating_time == 0.0:
                        gating_time = 3e-6
                elif "<ChannelCount>" in line:
                    channels = int(line.split(">")[1].split("<")[0])
                elif "<CalibAbs>" in line:
                    a0 = float(line.split(">")[1].split("<")[0])
                elif "<CalibLin>" in line:
                    a1 = float(line.split(">")[1].split("<")[0])
                elif "<SigmaAbs>" in line:
                    FWHM = float(line.split(">")[1].split("<")[0])
                    FWHM = 2*np.sqrt(2*np.log(2))*np.sqrt(FWHM)
                elif "<SigmaLin>" in line:
                    # 3.85eV is the mean energy required to create an electon-
                    # hole-pair
                    Fano = float(line.split(">")[1].split("<")[0]) / (3.85e-3)
                elif "<Channels>" in line:
                    spectrum = line.split(">")[1].split("<")[0].split(",")
                    break
        spectrum = [float(intensity) for intensity in spectrum]

        # calculate the life time from the zero peak
        # detector is set in 10keV, 20keV or 40keV and maipulate the
        # spectrum in order to represent an 40keV spectrum
        factor = (1/a1)/100
        if not (94 < zero_peak_frequency < 99):
            zero_peak_position = int(zero_peak_position/factor)
        a1 *= factor
        max_energy = a0 + a1 * (channels-1)
        life_time = np.sum(spectrum[
            zero_peak_position-int(20/factor):
            zero_peak_position+int(20/factor)]) / zero_peak_frequency * factor
        if life_time == 0:
            print("factor:\t", factor)
            print(
                "zero peak position and frequency:\t",
                zero_peak_position,
                zero_peak_frequency)
            print(
                f"error in {spx_file}\n",
                "life_time is ZERO\n",
                "setting life_time to 1")
            life_time = 1
        ds["parameters"][file_nr] = np.array([
            a0,
            a1,
            Fano,
            FWHM,
            life_time,
            max_energy,
            gating_time,
            real_time])
        spectrum = np.divide(spectrum, life_time)
        # sometimes the number of channels are corrupted, add or remove
        # channels to comply with 4096
        if channels < 4096:
            spectrum = np.r_[spectrum, np.zeros(4096-channels)]
        elif channels > 4096:
            spectrum = spectrum[:4096]
        # now we try to rebuild specfit load in routine to support array
        # spectra
        if save_spec_as_dict:
            if file_nr == 0:
                spectra = np.zeros((len(sorted_folder), len(spectrum)))
            spectra[file_nr] = spectrum
        else:
            if file_nr == 0:
                spectra = np.empty((len(x), len(y), len(z), 4096))
            x_pos, y_pos, z_pos = tensor_positions[file_nr]
            spectra[x_pos][y_pos][z_pos] = spectrum
        if signal_progress is not None:
            signal_progress.emit(file_nr)

    # reshape the spectra to the shape of the measurement
    ds["spectra"] = xr.DataArray(
        data=spectra.reshape([x, y, z, channels]),
        dims=("X", "Y", "Z", "energy"),
        coords={
            "energy": np.arange(
                ds["parameters"][file_nr][0],
                ds["parameters"][file_nr][5]+ds["parameters"][file_nr][1],
                ds["parameters"][file_nr][1]),
            "X": np.arange(x),
            "Y": np.arange(y),
            "Z": np.arange(z)},
        attrs={"units": "counts per second"})
    ds["counts"] = xr.DataArray(
        ds["spectra"].sum(axis=-1),
        dims=("X", "Y", "Z"),
        attrs={"units": "counts per second"})
    ds["max pixel spec"] = ds["spectra"].max(axis=(0, 1, 2))
    ds["sum spec"] = ds["spectra"].mean(axis=(0, 1, 2))
    # set units to xarray Dataset
    utils.set_xarray_units(dataset=ds)
    # save the hdf5
    ds.to_netcdf(
        path=folder_path/"data/data.h5",
        group=file_name,
        mode=write_operator,
        engine="h5netcdf",
        encoding=utils.create_hdf5_encoding(dataset=ds),
    )
    print(f"spx loadingtime - {t.time()-start:.2f} s")
    if return_values:
        if save_spec_as_dict is False:
            return (
                ds["spectra"],
                ds["parameters"],
                ds["positions"],
                ds["tensor_positions"]
            )
        else:
            return (
                ds["spectra"],
                ds["parameters"]
            )


def sum_from_single_files(
        folder_path,
        save_sum_spec=True
        ):
    Path(f"{folder_path}/data/").mkdir(parents=True, exist_ok=True)
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


def spx2life_time(file_path):
    """
    This function reads out the life-time in seconds of a .spx-file.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<LifeTime>" in line:
                life_time = int(line.split(">")[1].split("<")[0])
                life_time /= 1000.0
                break
            if "<RealTime>" in line:
                life_time = int(line.split(">")[1].split("<")[0])
                life_time /= 1000.0
                break
    return life_time


def spx2real_time(file_path):
    """
    This function reads out the real-time in seconds of a .spx-file.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<RealTime>" in line:
                real_time = int(line.split(">")[1].split("<")[0])
                real_time /= 1000.0
                break
    return real_time


def norm2sec(spectrum, time):
    """
    This function normalizes the given spectrum to seconds based on life or
    real time.

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


def spx2channels(file_path):
    """
    This function reads out the channe numbers.
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        for line in infile:
            if "<ChannelCount>" in line:
                channels = int(line.split(">")[1].split("<")[0])
                break
    return channels


def spx_log_content(file_path):
    """
    This function reads out all the parameters of the scan saved in the
    .log_file.

    Parameters
    ----------
    folder_path: str - path of the folder

    Returns
    -------
    [[scan_width], [start], [end], [positions]] = spx_log_content(file_path)
    """
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        k = 0
        _l = 0
        content = [[], [], [], []]
        for line in infile:
            if k == 2:
                content[0 + _l].append(float(line.split()[1]))
                _l += 1
                k = 0
            else:
                content[0 + _l].append(float(line.split()[1]))
                k += 1
        content[3] = [int(content[3][i]) for i in range(3)]
    return content


def spx_position(file_path):
    """
    This function reads out the specific position of the spectrum in the
    measurement-tensor.
    returns position = [x, y, z]
    position = spx_position(file_path)
    It determines whether it is a line scan or a 3D-Scan.
    """
    file_path = Path(file_path)
    data = None
    if file_path.suffix == ".spx":
        with open(file_path, "r", encoding="ISO-8859-1") as infile:
            for line in infile:
                if "<Data" in line:
                    encoded = line.split(">")[1].split("<")[0]
                    data = codecs.decode(
                        encoded.encode("ascii"),
                        "base64")
                    if len(data) == 162:
                        return (
                            np.frombuffer(
                                data[1:-1],
                                dtype=np.float64)[15:18]
                        )
                    elif len(data) == 161:
                        return (
                            np.frombuffer(
                                data[1:],
                                dtype=np.float64)[15:18]
                        )
                    else:
                        return (
                            np.frombuffer(
                                data[1:],
                                dtype=np.float64)[15:18]
                        )

        # for older .spx files no <Data for stage position is stored,
        # falling back on position retrieval from data naming
        if data is None:
            position = file_path.name.replace(
                file_path.suffix, '').split('(')[-1].replace(')', '')
            return [float(pos) for pos in position.split(",")]


def spx_positions(folder_path, file_type=".spx"):
    """
    This function reads out the tensor position of all spx/txt-files in the
    given folder_path

    Parameters
    ----------
    folder_path: str - path of the folder containing the spx or txt files

    Returns
    -------
    positions = np.array([x0, y0, z0], [x0, y1, z0], ..., [xn, ym, zk])
    """
    folder_path = Path(folder_path)
    files = folder_path.glob(f"*{file_type}")
    positions = []
    for file in files:
        position = spx_position(file)
        positions.append(position)
    return np.array(positions)


def log_file_type(log_file):
    """
    This function defines the type of the given .log-file by reading the
    header and scanning it for a keyphrase
    """
    with open(log_file, "r", encoding="ISO-8859-1") as log_file:
        log_file_type = None
        for line in log_file:
            if "Scan started" in line:
                log_file_type = "Louvre"
                break
    return log_file_type


def Louvre_log_file_content(log_file):
    """
    This function reads out the width, start and end position of the
    measurement as stated in the .log-file
    """
    with open(log_file, "r", encoding="ISO-8859-1") as log_file:
        #: ["width", "start", "end"]
        parameters = [2, 0, 1]
        width_start_end = []
        axes_parameters = []
        for line in log_file:
            if "Start" in line:
                line = line.split()
                line = [
                    convert_string(line[2]),
                    convert_string(line[4]),
                    convert_string(line[6])]
                axes_parameters.append(line)
                if len(axes_parameters) == 3:
                    log_file.close()
                    break
        for i in range(3):
            width_start_end.append(
                [axes_parameters[j][parameters[i]] for j in range(3)])
        width_start_end.append(
            [int(np.round(abs((width_start_end[2][i] - width_start_end[1][i]) /
                              width_start_end[0][i]) + 1))for i in range(3)])
    return width_start_end


def Louvre_tensor_position(log_file):
    tensor_position = {}
    temp = 0
    with open(log_file, "r+", encoding="ISO-8859-1") as log_file:
        for line in log_file:
            if "X  Y  Z" in line:
                temp = 1
            elif temp == 1:
                positions = line.split()
                positions = [
                    float(positions[i])/1000 for i, _ in enumerate(positions)]
                temp = 2
            elif temp == 2:
                tensor_position[int(line.replace(
                    "corresponding to spectrum No", ""))] = positions
                temp = False
    return tensor_position


def convert_string(string):
    string = string.replace(", ", ".")
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
