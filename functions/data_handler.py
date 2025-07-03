import os
import sys
import pathlib
import numpy as np
import h5py
from PyQt6 import QtWidgets, QtCore
from glob import glob
import time
import spx_functions as spx  # Module to read out .spx-files
import spe_functions as spe  # Module to read out .spx-files
import mca_functions as mca  # Module to read out .spx-files
import msa_functions as msa  # Module to read out .msa-files
import txt_functions as txt  # Module to read out .txt-files
import hdf_functions as hdf  # Module to read out .hdf-files
import bcf_functions as bcf  # Module to read out .bcf-files
import csv_functions as csv  # Module to read out .csv-files
import angles_functions as angles  # Module to interpret angle-measurements
import specfit_GUI_functions as sfunc  # Module used for the specfit_GUI

file_dir = os.path.dirname(os.path.abspath(__file__))
if "win" in sys.platform:
    parent_dir = "\\".join(file_dir.split("\\")[:-1])
    elements_path = parent_dir+"\\Data\\elements.dat"
else:
    parent_dir = "/".join(file_dir.split("/")[:-1])
    elements_path = parent_dir+"/Data/elements.dat"

z_elements = {}
with open(parent_dir+"/Data/elements.dat", "r") as element_file:
    for line in element_file:
        line = line.replace("\n", "").replace(" ", "").split("\t")
        z_elements[line[1]] = int(line[0])
        if line[0] == "98":
            break

class MSAThread(QtCore.QThread):
    def __init__(self, file_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.file_path = file_path

    def run(self):
        msa.msa2spec_sum_para(self.file_path,
                              self.signals.progress,
                              self.signals.sum_spec)
        self.signals.finished.emit("Done")

class SPXThread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.folder_path = folder_path

    def run(self):
        spx.many_spx2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit("Done")

class BCFThread(QtCore.QThread):
    def __init__(self, folderpath, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.folder_path = folderpath

    def run(self):
        bcf.many_bcf2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit("Done")


class SPEThread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.folder_path = folder_path

    def run(self):
        spe.many_spe2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit("Done")


class MCAThread(QtCore.QThread):
    def __init__(self, folder_path, XANES=False, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.folder_path = folder_path
        self.XANES = XANES

    def run(self):
        mca.many_mca2spec_para(self.folder_path,
                               XANES=self.XANES,
                               signal=self.signals.progress)
        self.signals.finished.emit("Done")


class TXTThread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = ThreadSignals()
        self.folder_path = folder_path

    def run(self):
        txt.many_txt2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit("Done")


class ThreadSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    sum_spec = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)


class DataHandler():
    """tool to handle the spectra Data, including parameters"""

    def __init__(self, parent=None):
        self.reset_2_default()
        self.parent = parent

    def reset_2_default(self):
        # if this file is in the same folder as spectra file/dir it is automatically loaded
        self.default_param_file_name = "param_save.txt"
        self.file_path = ""  # full path to spectra file
        self.folder_path = ""  # full path to dir that stores spec dict
        # 'angle_file' or 'file' or 'msa_file' or 'folder' or 'hdf5_file' or 'bcf_file'
        self.loadtype = ""
        self.file_type = ""  # .spx or .txt or .msa or .bcf
        self.save_folder_path = ""
        self.save_data_folder_path = ""
        self.SpecFit_MainWindow = None
        self.life_time = 0  # given in .msa or spx file
        self.spectra = {}  # dict {1:spectra1, 2:spectra2, ...}
        self.sum_spec = np.empty(1)  # stores sum_spec
        self.len_spectrum = 0  # stores the len of the sum spec
        self.len_scans = 1  # number of scans in a measurement
        # stores all parameters parameters that are stored in Data = [a0,a1,Fano, FWHM_0, lifetime, max_energy, gating_time] a0+a1*x
        self.parameters = None
        try:
            # do not change parameters if already defined
            self.parameters_user[0]
        except:
            # stores parameters defined by User in GUI,
            self.parameters_user = np.zeros(8)
        self.use_parameters = False  # if True use parameters_user else parameter_data
        self.bg_zero = False  # do not calculate background
        self.strip_cycles = 15
        self.strip_width = 60
        self.smooth_cycles = 0
        self.smooth_width = 1
        self.roi_start = 0.1  # investigated region start in keV
        self.roi_end = 15.  # investigated region end in keV
        self.calc_minima = True  # prior smoothing calc of minima in spectrum
        self.calc_minima_order = 15  # order of minima algorithm
        self.plot_style = "lin"  # lin or log
        # stores tensor positions [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]
        self.tensor_positions = np.empty(3)
        # stores number of steps in each direction for example [1,10,300]
        self.position_dimension = []
        # stores the short Form of all elements in List ['Ni','Cu']
        self.selected_elements = []
        self.selected_lines = []  # stores the selected Lines in a Nested List [[Ka,Kb],[Ka]]
        # stores dict needed by specfit in Form {'Cu': (True, ['K-line', 'L1'], 29)}
        self.specfit_addlines = {}
        self.nl_fit = False  # if True parameters are changed
        self.threshold = 1e-3
        self.user_defined_lines = []  # nested list with [[name,start,end],[name,start,end]]
        self.log_file_type = None  # 'spx or MSA','Louvre','None'
        # stores width,start, end as defined in log file [[scan_width],[start],[end],[position_dim]] = spx_log_content(file_path)
        self.log_content = []
        self.file_dialog = None
        self.main_thread = None
        self.file_list = []  # list of all files of type
        self.mincount = 0  # spec with less counts in ROI are not evaluated
        # label from the main GUI that can display the progress of the loading process
        self.label_loading_progress = None
        self.use_lib = "xraylib"

    def _get_file_path(self, angle_file):
        """ 
        read out file_path utilizing a QFileDialog and determine the loadtype
        as either angle_file or file
        """
        if angle_file:
            loadtype = "angle_file"
        else:
            loadtype = "file"
        try:
            self.file_path = self.file_dialog.getOpenFileName(
                filter="(*.spx *.MSA *.txt *.spe *.mca *.dat *.hdf5 *.h5 *.bcf *.csv)")[0]
            self.label_loading_progress.showMessage(self.file_path)
            assert os.path.exists(self.file_path)
        except:
            self.label_loading_progress.showMessage("file does not exist")
            self.file_path = "not_found"
        return self.file_path, loadtype

    def _get_folder_path(self):
        """
        read out the folderpath utilizing a QFileDialog
        """
        try:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()  # function for a folder-GUI
            self.label_loading_progress.showMessage(folderpath)
            assert os.path.isdir(folderpath)
        except:
            self.label_loading_progress.showMessage("Dir not found")
            folderpath = "not_found"
        return folderpath
    
    def _get_file_name(self, file_path):
        """
        return the file name
        """
        return file_path.split("/")[-1]

    def update_channels(self, use_user_parameter=False):
        """ 
        calculate the new energy axis and adjust it to the lengths of the
        loaded sum spectrum
        """

        if use_user_parameter:
            self.energies = np.arange(self.parameters_user[0], self.parameters_user[5] +
                                      self.parameters_user[1] * self.len_spectrum,
                                      self.parameters_user[1])  # max energy
        else:
            self.energies = np.arange(
                self.parameters[0][0], self.parameters[0][0] + self.parameters[0][1] * self.len_spectrum,
                self.parameters[0][1])  # max energy
            
        if len(self.energies) != len(self.sum_spec):
            self.energies.resize(len(self.sum_spec))

        # now try to set the parmeters max energy value to the highest energy
        # value or set it to 40 as default
        try:
            self.parameters_user[5] = self.energies[-1]
        except:
            self.parameters_user[5] = 40.
    
    def update_data_from_h5file(self, h5_path):
        """
        function to update parameters in the data class from a data.h5 provided

        Parameters
        ----------
        h5_path : str
            absolute path to the h5 file which should be used to update the 
            parameters in the data handler class.
        """
        with h5py.File(h5_path, "r") as f:
            if "tensor positions" in f.keys():
                file_key=""
            else:
                file_key=f"{self.file_name}/"
            self.tensor_positions = f[f"{file_key}tensor positions"][()]
            self.positions = f[f"{file_key}positions"][()]
            self.position_dimension = f[f"{file_key}position dimension"][()]
            self.sum_spec = f[f"{file_key}sum spec"][()]
            self.parameters = f[f"{file_key}parameters"][()]
            self.max_pixel_spec = f[f"{file_key}max pixel spec"][()]
            self.len_spectrum = len(self.sum_spec)
            self.spectra = f[f"{file_key}spectra"][()]
        if self.parameters.ndim == 1:
            self.parameters = np.expand_dims(self.parameters, axis=0)

    def get_roi_indicees(self):
        """
        Return ROI indices as tuple
        """
        idx_min = (np.abs(np.subtract(self.energies, self.roi_start))).argmin()
        idx_max = (np.abs(np.subtract(self.energies, self.roi_end))).argmin()
        return idx_min, idx_max

    def open_data_file(self, angle_file=False):
        self.file_path, self.loadtype = self._get_file_path(angle_file)
        self.file_name = self._get_file_name(self.file_path)
        self.folder_path = self.get_folder_of_path()
        self.save_data_folder_path = f"{self.folder_path}/data"
        self.file_type = self.get_file_type()
        reload = QtWidgets.QMessageBox.Yes
        if self.file_type == ".MSA":
            self.save_data_folder_path = os.path.join(self.folder_path, "data")
            self.loadtype = "msa_file"
            self.file_list = list(
                np.sort(glob(f"{self.folder_path}/*{self.file_path}")))
            self.life_time = msa.msa2life_time(f"{self.file_path}")
            self.len_spectrum = msa.msa2channels(self.file_path)

            if os.path.isfile(f"{self.save_data_folder_path}/data.h5"):
                with h5py.File(f"{self.save_data_folder_path}/data.h5", "r") as f:
                    if self.file_name in f.keys(): reload = True
                if reload is True:  
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                            "Do you want to reload the measurement?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.run_msa_worker()  # stores parameters
                    assert self.load_stored_spec_and_param(
                    ), "something is wrong with .MSA worker, parameters are not stored correctly"
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.sum_spec, self.parameters = msa.msa2spec_sum_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.energies = np.arange(self.parameters[0][0],
                                          self.parameters[0][0] +
                                          self.parameters[0][1] * self.len_spectrum,
                                          self.parameters[0][1])
                assert self.load_stored_spec_and_param(
                ), "something is wrong with .MSA worker, parameters are not stored correctly"

        elif self.file_type == ".bcf":
            self.loadtype = "bcf_file"
            if os.path.isfile(f"{self.save_data_folder_path}/data.h5"):
                with h5py.File(f"{self.save_data_folder_path}/data.h5", "r") as f:
                    if self.file_name in f.keys(): reload = True
                if reload is True:  
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                            "Do you want to reload the measurement?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.bcf2spec_para(
                        self.file_path)
                    if self.parameters.ndim == 1:
                        self.parameters = np.expand_dims(self.parameters, axis=0) 
                    self.life_time = self.parameters[0][4]
                    self.real_time = self.parameters[0][7]
                    self.parameters_user = np.copy(self.parameters[0])
                    self.energies = np.arange(
                        self.parameters[0][0], self.parameters[0][5], self.parameters[0][1])
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.bcf2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.parameters_user = np.copy(self.parameters[0])
                self.energies = np.arange(
                    self.parameters[0][0], self.parameters[0][5], self.parameters[0][1])
            self.len_spectrum = len(self.sum_spec)
                
        elif self.file_type == ".csv":
            self.loadtype = "csv_file"
            if os.path.isfile(f"{self.save_data_folder_path}/data.h5"):
                with h5py.File(f"{self.save_data_folder_path}/data.h5", "r") as f:
                    if self.file_name in f.keys(): reload = True
                if reload is True:  
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                            "Do you want to reload the measurement?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters = csv.csv2spec_para(self.file_path)
                    if self.parameters.ndim == 1:
                        self.parameters = np.expand_dims(self.parameters, axis=0) 
                    self.tensor_positions = csv.csv_tensor_positions(self.file_path)
                    self.positions = self.tensor_positions
                    self.position_dimension = csv.csv_position_dimension(self.file_path)
                    self.sum_spec = np.sum(self.spectra, axis=0)
                    self.life_time = self.parameters[0][4]
                    self.real_time = self.parameters[0][7]
                    self.parameters_user = np.copy(self.parameters[0])
                    self.energies = np.arange(
                        self.parameters[0][0], self.parameters[0][5], self.parameters[0][1])
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters = csv.csv2spec_para(self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.tensor_positions = csv.csv_tensor_positions(self.file_path)
                self.positions = self.tensor_positions
                self.position_dimension = csv.csv_position_dimension(self.file_path)
                self.sum_spec = np.sum(self.spectra, axis=0)
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.parameters_user = np.copy(self.parameters)
                self.energies = np.arange(
                    self.parameters[0][0], self.parameters[0][5], self.parameters[0][1])
            self.len_spectrum = len(self.sum_spec)

        elif self.file_type == ".hdf5" or self.file_type == ".h5":
            self.loadtype = "hdf5_file"
            if os.path.isfile(f"{self.save_data_folder_path}/data.h5"):
                with h5py.File(f"{self.save_data_folder_path}/data.h5", "r") as f:
                    if self.file_name in f.keys():
                        reload = True
                    else:
                        reload = QtWidgets.QMessageBox.Yes
                if reload is True:
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                            "Do you want to reload the measurement?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters, self.sum_spec, self.len_spectrum = hdf.hdf2spec_para(
                        self.file_path)
                    if self.parameters.ndim == 1:
                        self.parameters = np.expand_dims(self.parameters, axis=0) 
                    self.life_time = self.parameters[0][4]
                    self.real_time = self.parameters[0][7]
                    self.position_dimension, self.tensor_positions = hdf.hdf_tensor_positions(
                        self.file_path)
                    
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters, self.sum_spec, self.len_spectrum = hdf.hdf2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.position_dimension,
                self.tensor_positions = hdf.hdf_tensor_positions(self.file_path)
            if self.file_type == ".hdf5":
                self.roi_start = 0.0
                self.roi_end = 2
            elif self.file_type == ".h5":
                self.roi_start = 0.1
                self.roi_end = 15

        else:
            if self.file_type == ".spx":
                self.spectra["0"], self.parameters = spx.spx2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.len_spectrum = spx.spx2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra["0"]
                self.roi_start = 0.5
                self.roi_end = 15

            elif self.file_type == ".spe":
                self.spectra["0"], self.parameters = spe.spe2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.len_spectrum = spe.spe2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra["0"]
                self.roi_start = 0.5
                self.roi_end = 25

            elif self.file_type == ".mca":
                self.spectra["0"], self.parameters = mca.mca2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.len_spectrum = mca.mca2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra["0"]
                self.roi_start = 0.5
                self.roi_end = 15

            elif self.file_type == ".txt" and (not angle_file):
                self.spectra["0"], self.parameters = txt.txt2spec_para(
                    self.file_path)
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.life_time = self.parameters[0][4]
                self.real_time = self.parameters[0][7]
                self.len_spectrum = txt.txt2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra["0"]
                self.roi_start = 0.5
                self.roi_end = 15
                
            elif angle_file:
                if os.path.isdir(self.save_data_folder_path):
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                            "Do you want to reload the measurement?",
                                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                            QtWidgets.QMessageBox.No)
                    if reload == QtWidgets.QMessageBox.Yes:
                        self.spectra, self.parameters, self.tensor_positions, \
                        self.position_dimension, self.sum_spec = angles.txt2spec_para(self.file_path)
                        self.len_spectrum = len(self.sum_spec)
                    elif reload == QtWidgets.QMessageBox.No:
                        self.load_stored_spec_and_param()
                else:
                    self.spectra, self.parameters, self.tensor_positions, \
                    self.position_dimension, self.sum_spec = angles.txt2spec_para(self.file_path)
                    self.len_spectrum = len(self.sum_spec)
        self.parameters = np.array(self.parameters)
        if self.parameters.ndim == 1:
            self.parameters = np.expand_dims(self.parameters, axis=0) 
        param_path = os.path.join(self.folder_path, self.default_param_file_name)
        self.max_pixel_spec = np.max(self.spectra, axis=0)
        self.check_param_file(param_path)
        self.update_channels()
        self.create_save_folder()
        self.parent.entry_a0.setText(f"{self.parameters[0][0]}")
        self.parent.entry_a1.setText(f"{self.parameters[0][1]}")
        self.parent.entry_fano.setText(f"{self.parameters[0][2]}")
        self.parent.entry_FWHM.setText(f"{self.parameters[0][3]}")

    def open_data_folder(self):
        """
        This function reads out the given folder, decides wether its .spx, or .txt 
        and loads the spectra, and parameters. 
        """
        self.loadtype = "folder"
        self.folder_path = self._get_folder_path()
        self.file_name = self._get_file_name(self.folder_path)
        if self.folder_path != "not_found":
            self.save_data_folder_path = os.path.join(self.folder_path, "data")
            # if any spx in folder -> filetype spx
            self.file_type = self.determine_folder_file_type()
            # create a list of all file_type-files contained in the selected folder
            self.file_list = list(
                np.sort(glob(f"{self.folder_path}/*{self.file_type}")))
            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, "?",
                                                        "Do you want to reload the measurement?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    if self.file_type == ".spx":
                        self.run_spx_worker()
                        self.roi_start = 0.5
                    elif self.file_type == ".spe":
                        self.run_spe_worker()
                        self.roi_start = 0.5
                        self.roi_end = 25
                    elif self.file_type == ".bcf":
                        self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.many_bcf2spec_para(
                            self.folder_path)
                        if self.parameters.ndim == 1:
                            self.parameters = np.expand_dims(self.parameters, axis=0) 
                        self.roi_start = 0.5
                    elif self.file_type == ".mca":
                        self.XANES = False
                        self.run_mca_worker(XANES=self.XANES)
                        self.roi_start = 0.5
                    elif self.file_type == ".txt":
                        self.run_txt_worker()
                        self.roi_start = 0.5
                    self.find_and_save_log_file()
                    if self.file_type == ".bcf":
                        self.positions = np.copy(self.tensor_positions)
                    else:  # for all different data types
                        self.tensor_positions, self.position_dimension, self.positions = self.tensor_positions_log_file(
                            self.file_list)
                    with h5py.File(f"{self.save_data_folder_path}/data.h5", "a") as tofile:
                        if isinstance(self.tensor_positions, object):
                            tofile.create_dataset(f"{self.file_name}/tensor positions",
                                                  data=np.concatenate(self.tensor_positions, axis=0))
                            tofile.create_dataset(f"{self.file_name}/positions",
                                                  data=np.concatenate(self.positions, axis=0))
                        else:
                            tofile.create_dataset(f"{self.file_name}/tensor positions", data=self.tensor_positions)
                            tofile.create_dataset(f"{self.file_name}/positions", data=self.positions)
                        tofile.create_dataset(f"{self.file_name}/position dimension", data=self.position_dimension)

                elif reload == QtWidgets.QMessageBox.No:
                    if self.load_stored_spec_and_param(folder=True):
                        pass
            else:
                if self.file_type == ".spx":
                    self.run_spx_worker()
                    self.roi_start = 0.5
                elif self.file_type == ".bcf":
                    self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.many_bcf2spec_para(
                        self.folder_path)
                    if self.parameters.ndim == 1:
                        self.parameters = np.expand_dims(self.parameters, axis=0) 
                    self.roi_start = 0.5
                elif self.file_type == ".spe":
                    self.run_spe_worker()
                    self.roi_start = 0.5
                    self.roi_end = 25
                elif self.file_type == ".mca":
                    self.XANES = False
                    self.run_mca_worker(XANES=self.XANES)
                    self.roi_start = 0.5
                elif self.file_type == ".txt":
                    self.run_txt_worker()
                    self.roi_start = 0.5
                self.find_and_save_log_file()
                if self.file_type == ".bcf":
                    self.positions = np.copy(self.tensor_positions)
                else:  # for all different data types
                    self.tensor_positions, self.position_dimension, self.positions = self.tensor_positions_log_file(
                        self.file_list)
                with h5py.File(f"{self.save_data_folder_path}/data.h5", "a") as tofile:
                    if isinstance(self.tensor_positions, object):
                        tofile.create_dataset(f"{self.file_name}/tensor positions",
                                              data=np.concatenate(self.tensor_positions, axis=0))
                        tofile.create_dataset(f"{self.file_name}/positions",
                                              data=np.concatenate(self.positions, axis=0))
                    else:
                        tofile.create_dataset(f"{self.file_name}/tensor positions", data=self.tensor_positions)
                        tofile.create_dataset(f"{self.file_name}/positions", data=self.positions)
                    tofile.create_dataset(f"{self.file_name}/position dimension", data=self.position_dimension)
                
        assert self.load_stored_spec_and_param(folder=True)
        if self.file_type == ".bcf":
            self.life_time = self.parameters[0][4]
            self.real_time = self.parameters[0][7]
            self.create_save_folder()
            self.check_param_file(os.path.join(
                self.folder_path, self.default_param_file_name))
            self.update_channels()
            self.parent.entry_a0.setText(f"{self.parameters[0][0]}")
            self.parent.entry_a1.setText(f"{self.parameters[0][1]}")
            self.parent.entry_fano.setText(f"{self.parameters[0][2]}")
            self.parent.entry_FWHM.setText(f"{self.parameters[0][3]}")
        else:  # for all other data types
            self.life_time = self.parameters[0][4]
            self.real_time = self.parameters[0][7]
            self.create_save_folder()
            self.check_param_file(os.path.join(
                self.folder_path, self.default_param_file_name))
            self.update_channels()
            self.parent.entry_a0.setText(f"{self.parameters[0][0]}")
            self.parent.entry_a1.setText(f"{self.parameters[0][1]}")
            self.parent.entry_fano.setText(f"{self.parameters[0][2]}")
            self.parent.entry_FWHM.setText(f"{self.parameters[0][3]}")
        self.max_pixel_spec = np.max(self.spectra, axis=0)
        

    def load_stored_spec_and_param(self, folder=False, stored_data=False):
        """
        try to load previously stored data, return a boolean if this was 
        succesful or not
        """
        if os.path.exists(f"{self.save_data_folder_path}/data.h5"):
            self.label_loading_progress.showMessage("loading stored data")
            stored_data = True
            with h5py.File(f"{self.save_data_folder_path}/data.h5", "r") as infile:
                if "tensor positions" in infile.keys():
                    file_key=""
                else:
                    file_key=f"{self.file_name}/"
                self.tensor_positions = infile[f"{file_key}tensor positions"][()]
                self.positions = infile[f"{file_key}positions"][()]
                self.position_dimension = infile[f"{file_key}position dimension"][()]
                self.sum_spec = infile[f"{file_key}sum spec"][()]
                self.parameters = infile[f"{file_key}parameters"][()]
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.max_pixel_spec = infile[f"{file_key}max pixel spec"][()]
                self.len_spectrum = len(self.sum_spec)
                self.spectra = infile[f"{file_key}spectra"][()]
            if self.file_type == ".mca":  #TODO
                self.len_scans = np.prod(self.position_dimension, axis = 1)
                self.sum_len_scans = [np.arange(0, self.len_scans[0])]
                for i, index in enumerate(self.len_scans[1:]):
                    self.sum_len_scans.append(np.arange(self.len_scans[i], index))
                if len(self.len_scans) > 1:
                    tensor_0 = np.array([self.tensor_positions[0:self.len_scans[0]]])
                    tensor = [self.tensor_positions[self.len_scans[i]:index+1] for i, index in
                                                   enumerate(self.len_scans[1:])]
                    tensor_positions = [tensor_0, tensor]
                    positions = np.array([self.positions[0:self.len_scans[0]]])
                    positions = np.vstack(([self.positions[self.len_scans[i]:index+1] for i, index in
                                            enumerate(self.len_scans[1:])]))
                    self.tensor_positions = tensor_positions
                    self.positions = positions
                else:
                    if self.tensor_positions.ndim != 3:
                        self.tensor_positions = np.expand_dims(self.tensor_positions, axis=0)
            return stored_data
        if os.path.exists(f"{self.save_data_folder_path}/parameters.npy") and os.path.exists(
                f"{self.save_data_folder_path}/tensor_positions.npy") and os.path.exists(
                f"{self.save_data_folder_path}/sum_spec.npy") and os.path.exists(
                f"{self.save_data_folder_path}/position_dimension.npy"):
            self.label_loading_progress.showMessage("loading stored data")
            if os.path.exists(f"{self.save_data_folder_path}/spectra.pickle"):
                stored_data = True
                if not folder:
                    self.spectra = sfunc.open_dict_pickle(
                        f"{self.save_data_folder_path}/spectra.pickle")
                self.tensor_positions = np.load(
                    f"{self.save_data_folder_path}/tensor_positions.npy", allow_pickle=True)
                self.position_dimension = np.load(
                    f"{self.save_data_folder_path}/position_dimension.npy")
                self.max_pixel_spec = np.load(
                    f"{self.save_data_folder_path}/max_pixel_spec.npy")
                self.parameters = np.load(
                    f"{self.save_data_folder_path}/parameters.npy")
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.sum_spec = np.load(f"{self.save_data_folder_path}/sum_spec.npy")
                self.len_spectrum = len(self.sum_spec)
            elif os.path.isdir(f"{self.folder_path}/single_spectra"):
                stored_data = True
                # spectra and tensor positions are not loaded here ...
                self.parameters = np.load(
                    f"{self.save_data_folder_path}/parameters.npy")
                if self.parameters.ndim == 1:
                    self.parameters = np.expand_dims(self.parameters, axis=0) 
                self.sum_spec = np.load(f"{self.save_data_folder_path}/sum_spec.npy" %
                                        ())
                self.len_spectrum = len(self.sum_spec)
                self.position_dimension = np.load(
                    f"{self.save_data_folder_path}/position_dimension.npy")
        return stored_data

    def build_tensor_position(self, positions):
        """
        needs list/arraywit [xmax,ymax,z_max]returns an array of Form [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]
        """
        pos_arr_1 = np.arange(positions[0])
        pos_arr_2 = np.arange(positions[1])
        pos_arr_3 = np.arange(positions[2])
        return np.array(np.meshgrid(pos_arr_1, pos_arr_2, pos_arr_3)).T.reshape(-1, 3)

    def run_msa_worker(self):
        msa_worker = MSAThread(self.file_path)
        msa_worker.start()
        msa_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_spx_worker(self):
        spx_worker = SPXThread(self.folder_path)
        spx_worker.signals.progress.connect(self.run_spx_progress)
        spx_worker.start()
        spx_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_bcf_worker(self):
        bcf_worker = BCFThread(self.folder_path)
        bcf_worker.signals.progress.connect(self.run_bcf_progress)
        bcf_worker.start()
        bcf_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_spe_worker(self):
        spe_worker = SPEThread(self.folder_path)
        spe_worker.signals.progress.connect(self.run_spe_progress)
        spe_worker.start()
        spe_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_mca_worker(self, XANES=False):
        mca_worker = MCAThread(self.folder_path, XANES=XANES)
        mca_worker.signals.progress.connect(self.run_mca_progress)
        mca_worker.start()
        mca_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_txt_worker(self):
        txt_worker = TXTThread(self.folder_path)
        txt_worker.signals.progress.connect(self.run_txt_progress)
        txt_worker.start()
        txt_worker.wait()
        self.label_loading_progress.showMessage("loading done")

    def run_spx_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "spx-progress - {file_nr} files")

    def run_bcf_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "bcf-progress - {file_nr} files")

    def run_spe_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "spe-progress - {file_nr} files")

    def run_mca_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "mca-progress - {file_nr} files")

    def run_msa_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "msa-progress - {file_nr} files")

    def run_txt_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            "txt-progress - {file_nr} files")

    def check_param_file(self, param_path):
        """
        check if a file param_save.txt exists in the folder and load its content into specfit
        """
        if os.path.exists(param_path):
            self.selected_elements = []
            self.selected_lines = []
            self.user_defined_lines = []
            paramfile = open(param_path, "r")
            lines = paramfile.readlines()
            for line in lines:
                key, value = line.split(":")
                if key == "a0":
                    self.parameters_user[0] = float(value)
                elif key == "a1":
                    self.parameters_user[1] = float(value)
                elif key == "fano":
                    self.parameters_user[2] = float(value)
                elif key == "FWHM":
                    self.parameters_user[3] = float(value)
                elif key == "use_parameters":
                    self.use_parameters = value.strip() == "True"
                elif key == "background=0":
                    self.bg_zero = value.strip() == "True"
                elif key == "strip_cycles":
                    self.strip_cycles = int(float(value.strip()))
                elif key == "strip_width":
                    self.strip_width = int(float(value.strip()))
                elif key == "smooth_cycles":
                    self.smooth_cycles = int(float(value.strip()))
                elif key == "smooth_width":
                    self.smooth_width = int(float(value.strip()))
                elif key == "roi_start":
                    self.roi_start = float(value.strip())
                elif key == "roi_end":
                    self.roi_end = float(value.strip())
                elif key == "log":
                    if value.strip() == "True":
                        self.plot_style = "log"
                elif key == "nl-Fit":
                    self.nl_fit = value.strip() == "True"
                elif key == "calc_minima":
                    self.calc_minima = value.strip() == "True"
                elif key == "calc_minima_order":
                    self.calc_minima_order = int(value.strip())
                elif key == "threshold":
                    self.threshold = value.strip()
                elif key == "use_lib":
                    self.use_lib = value.strip()
                elif key == "element":
                    line = value.strip().strip("\n").split(",")
                    elementname = line[0]
                    linelist = line[1:]
                    self.selected_elements.append(elementname)
                    self.selected_lines.append(linelist)
                elif key == "user_defined_line":
                    label = value.strip().split(",")[0]
                    start = value.split(",")[1].strip()
                    end = value.split(",")[2].strip()
                    self.user_defined_lines.append([label, start, end])
                elif key == "calc_PU":
                    if eval(value):
                        self.parent.sfs.check_PU_Lines.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_PU_Lines.setCheckState(QtCore.Qt.CheckState.Unchecked)
                elif key == "calc_Escape":
                    if eval(value):
                        self.parent.sfs.check_Escape_Lines.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_Escape_Lines.setCheckState(QtCore.Qt.CheckState.Unchecked)
                elif key == "PU_threshold":
                    self.parent.sfs.entry_PU_threshold.setText(value)
                elif key == "PU_factor":
                    self.parent.sfs.entry_PU_factor.setText(value)
                elif key == "Escape_threshold":
                    self.parent.sfs.entry_Escape_threshold.setText(value)
                elif key == "Escape_factor":
                    self.parent.sfs.entry_Escape_factor.setText(value)
                elif key == "fit_in_batches":
                    if eval(value):
                        self.parent.sfs.check_batch_fitting.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_batch_fitting.setCheckState(QtCore.Qt.CheckState.Unchecked)
                elif key == "save_background":
                    if eval(value):
                        self.parent.sfs.check_save_background.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_save_background.setCheckState(QtCore.Qt.CheckState.Unchecked)
                elif key == "save_fitted_spectrum":
                    if eval(value):
                        self.parent.sfs.check_save_fitted_spectrum.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_save_fitted_spectrum.setCheckState(QtCore.Qt.CheckState.Unchecked)
                elif key == "save_storage":
                    if eval(value):
                        self.parent.sfs.check_save_storage.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        self.parent.sfs.check_save_storage.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            if self.loadtype == "folder" and type(self.parameters[0] == list):
                self.parameters_user = self.parameters[0]

            else:
                if np.all(np.equal(self.parameters_user, np.zeros(8))):
                    self.parameters_user = self.parameters[0]
        # now set PU and Escape parameter in GUI
        self.parent.sfs.set_PU_Escape_parameters()

    def save_param_file(self, filename):
        """
        this function opens a dalog to determine a save location and saves settings and parameters
        """
        # check if ending of param file is .dat or .txt
        if pathlib.Path(filename).suffix not in [".txt", ".dat"]:
            filename += ".dat"
        # determine check state
        batch_fitting = self.parent.sfs.check_batch_fitting.checkState().value
        if batch_fitting == 0:
            batch_fitting = False
        else:
            batch_fitting = True
        save_background = self.parent.sfs.check_save_background.checkState().value
        if save_background == 0:
            save_background = False
        else:
            save_background = True
        save_fitted_spectrum = self.parent.sfs.check_save_fitted_spectrum.checkState().value
        if save_fitted_spectrum == 0:
            save_fitted_spectrum = False
        else:
            save_fitted_spectrum = True
        save_storage = self.parent.sfs.check_save_storage.checkState().value
        if save_storage == 0:
            save_storage = False
        else:
            save_storage = True
        # save parameter to file
        try:
            with open(filename, "w") as f:
                f.write(f"a0:{self.parameters_user[0]}\n")
                f.write(f"a1:{self.parameters_user[1]}\n")
                f.write(f"fano:{self.parameters_user[2]}\n")
                f.write(f"FWHM:{self.parameters_user[3]}\n")
                f.write(f"use_parameters:{self.use_parameters}\n")
                f.write(f"background=0:{self.bg_zero}\n")
                f.write(f"save_background:{save_background}\n")
                f.write(f"save_fitted_spectrum:{save_fitted_spectrum}\n")
                f.write(f"save_storage:{save_storage}\n")
                f.write(f"strip_cycles:{self.strip_cycles}\n")
                f.write(f"strip_width:{self.strip_width}\n")
                f.write(f"smooth_cycles:{self.smooth_cycles}\n")
                f.write(f"smooth_width:{self.smooth_width}\n")
                f.write(f"roi_start:{self.roi_start}\n")
                f.write(f"roi_end:{self.roi_end}\n")
                f.write(f'log:{self.plot_style == "log"}\n')
                f.write(f"nl-Fit:{self.nl_fit}\n")
                f.write(f"calc_minima:{self.calc_minima}\n")
                f.write(f"calc_minima_order:{self.calc_minima_order}\n")
                f.write(f"threshold:{self.threshold}\n")
                f.write(f"use_lib:{self.use_lib}\n")
                f.write(f"fit_in_batches:{batch_fitting}\n")
                f.write(f"calc_PU:{self.parent.s.calc_PU}\n")
                f.write(f"PU_threshold:{self.parent.s.PU_threshold}\n")
                f.write(f"PU_factor:{self.parent.s.PU_factor}\n")
                f.write(f"calc_Escape:{self.parent.s.calc_Escape}\n")
                f.write(f"Escape_threshold:{self.parent.s.Escape_threshold}\n")
                f.write(f"Escape_factor:{self.parent.s.Escape_factor}")
                for sel_element, sel_lines in zip(self.selected_elements, self.selected_lines):
                    linestring = ",".join(sel_lines)
                    f.write(f"\nelement: {sel_element},{linestring}")
                for label, start, end in self.user_defined_lines:
                    f.write(f"\nuser_defined_line: {label},{start},{end}")
                outstring = f"saved parameters in {filename}"
        except:
            outstring = "could not save parameters"

        return outstring

    def save_npy_probs(self):
        """
        save data in folder_path/data.
        """
        if not os.path.exists(self.save_data_folder_path):
            os.mkdir(self.save_data_folder_path)
        with h5py.File(f"{self.save_data_folder_path}/data.h5", "a") as tofile:
            if "counts" in list(tofile.keys()):
                tofile["counts"][()] = self.counts
                tofile["parameters"][()] = self.parameters
                tofile["sum spec"][()] = self.sum_spec
                tofile["max pixel spec"][()] = self.max_pixel_spec
                tofile["spectra"][()] = self.spectra
            else:
                tofile.create_dataset(f"{self.file_name}/counts", data=self.counts)
                tofile.create_dataset(f"{self.file_name}/parameters", data=self.parameters)
                tofile.create_dataset(f"{self.file_name}/sum spec", data=self.sum_spec)
                tofile.create_dataset(f"{self.file_name}/max pixel spec", data=self.max_pixel_spec)
                tofile.create_dataset(f"{self.file_name}/spectra", data=self.spectra)

    def remove_element(self, label):
        self.selected_elements = np.array(self.selected_elements)
        try:
            indicee = np.where(self.selected_elements == label)[0][0]
        except IndexError:
            self.label_loading_progress.showMessage(
                "element is not in selected_element_list")
        else:
            del self.selected_elements[indicee]
            del self.selected_lines[indicee]

    def get_label_working_folder(self):
        if (self.loadtype == "angle_file") or (self.loadtype == "file"):  # or msa_file or folder??
            working_folder = ""
        return working_folder

    def get_file_type(self):
        return pathlib.Path(self.file_path).suffix

    def determine_folder_file_type(self):
        file_type = ""
        if len(list(np.sort(glob(f"{self.folder_path}/*.spx")))) != 0:
            file_type = ".spx"
        elif len(list(np.sort(glob(f"{self.folder_path}/*.spe")))) != 0:
            file_type = ".spe"
        elif len(list(np.sort(glob(f"{self.folder_path}/*.mca")))) != 0:
            file_type = ".mca"
        elif len(list(np.sort(glob(f"{self.folder_path}/*.txt")))) != 0:
            file_type = ".txt"
        elif len(list(np.sort(glob(f"{self.folder_path}/*.bcf")))) != 0:
            file_type = ".bcf"
        elif len(list(np.sort(glob(f"{self.folder_path}/*.csv")))) != 0:
            file_type = ".csv"
        return file_type

    def get_lifetime(self):
        return self.life_time

    def get_folder_of_path(self):
        return os.path.dirname(self.file_path)

    def find_and_save_log_file(self):
        try:
            log_file = glob(f"{self.folder_path}/*.log")[0]  # get the .log-file
        except:
            self.log_file_type = None
            return  # if none .log is provided, the results array is 1D and tried to be sorted by name
        else:
            try:  # check wether the .log file is the standart or the Louvre type
                self.log_content = spx.spx_log_content(log_file)
                self.log_file_type = "spx or MSA"
            except:
                self.log_content = spx.Louvre_log_file_content(log_file)
                self.log_file_type = "Louvre"

    def create_save_folder(self):
        if self.loadtype in ["file", "msa_file", "angle_file", "hdf5_file", "bcf_file",
                             "csv_file"]:
            self.save_folder_path = f"{self.folder_path}/results"
        else:
            self.save_folder_path = f"{self.folder_path}/results"
        pathlib.Path(self.save_folder_path).mkdir(parents=True, exist_ok=True)

    def build_specfit_addlines(self):
        # builds dict{'element':(True/False, ['linename',...], Z), '':...,}
        # example: {'Cu': (True, ['K-line', 'L1'], 29)}
        self.specfit_addlines = {}
        for element, lines in zip(self.selected_elements, self.selected_lines):
            self.specfit_addlines[f"{element}"] = (
                True, lines, z_elements[element])

    def tensor_positions_log_file(self, file_list):  # TODO
        """
        return tensor_positions and position_dim  depending on log file
        """
        if self.log_file_type is None:
            if self.file_type == ".spx":
                positions = spx.spx_tensor_position(file_list[0])
                for i in range(len(file_list)-1):
                    position = spx.spx_tensor_position(file_list[i+1])
                    positions = np.vstack((positions, position))
                x = np.sort(np.unique(positions[:, 0]))
                y = np.sort(np.unique(positions[:, 1]))
                z = np.sort(np.unique(positions[:, 2]))
                start_position = [x[0], y[0], z[0]]
                position_dim = [len(x), len(y), len(z)]
                self.step_width = [0., 0., 0.]
                if len(x) != 1:
                    self.step_width[0] = x[1]-x[0]
                if len(y) != 1:
                    self.step_width[1] = y[1]-y[0]
                if len(z) != 1:
                    self.step_width[2] = z[1]-z[0]
                self.step_width = np.round(self.step_width, 3)
                tensor_position = (positions-start_position)/self.step_width
                tensor_position[tensor_position == np.inf] = 0
                # this if a value is NaN
                tensor_position[tensor_position != tensor_position] = 0
                tensor_position = np.around(tensor_position, 0).astype(int)
            elif self.file_type == ".spe":
                positions = spe.spe_tensor_position(file_list[0])
                for i in range(len(file_list)-1):
                    position = spe.spe_tensor_position(file_list[i+1])
                    positions = np.vstack((positions, position))
                x = np.sort(np.unique(positions[:, 0]))
                y = np.sort(np.unique(positions[:, 1]))
                z = np.sort(np.unique(positions[:, 2]))
                start_position = [x[0], y[0], z[0]]
                position_dim = [len(x), len(y), len(z)]
                self.step_width = [0., 0., 0.]
                if len(x) != 1:
                    self.step_width[0] = x[1]-x[0]
                if len(y) != 1:
                    self.step_width[1] = y[1]-y[0]
                if len(z) != 1:
                    self.step_width[2] = z[1]-z[0]
                self.step_width = np.round(self.step_width, 3)
                tensor_position = (positions-start_position)/self.step_width
                tensor_position[tensor_position == np.inf] = 0
                # this if a value is NaN
                tensor_position[tensor_position != tensor_position] = 0
                tensor_position = np.around(tensor_position, 0).astype(int)
            elif self.file_type == ".mca":
                positions_tmp, self.len_scans = mca.mca_tensor_positions(
                    file_list, XANES=self.XANES)
                nr_scans = len(self.len_scans)
                self.sum_len_scans = []
                for i, length in enumerate(self.len_scans):
                    self.sum_len_scans.append(
                        np.arange(sum(self.len_scans[:i]), sum(self.len_scans[:i+1])))
                x = [[] for i in range(nr_scans)]
                z = [[] for i in range(nr_scans)]
                monoE = [[] for i in range(nr_scans)]
                start_position = [[] for i in range(nr_scans)]
                position_dim = [[] for i in range(nr_scans)]
                self.step_width = [[] for i in range(nr_scans)]
                tensor_position = [[] for i in range(nr_scans)]
                positions = [[] for i in range(nr_scans)]
                empty_scans = []
                for scan in range(nr_scans):
                    if scan == 0:
                        positions[scan] = np.array(
                            positions_tmp[:self.len_scans[scan]])
                    elif scan == (nr_scans-1):
                        self.positions_tmp = positions_tmp
                        positions[scan] = np.array(
                            positions_tmp[sum(self.len_scans[:scan]):])
                    else:
                        positions[scan] = np.array(
                            positions_tmp[sum(self.len_scans[:scan]):sum(self.len_scans[:scan+1])])
                    if positions[scan].size == 0:
                        empty_scans.append(scan)
                        continue
                    if np.ndim(positions[scan]) != 1:
                        x[scan] = np.unique(positions[scan][:, 0])
                        z[scan] = np.unique(positions[scan][:, 1])
                        monoE[scan] = np.unique(positions[scan][:, 2])
                        start_position[scan] = [
                            x[scan][0], z[scan][0], monoE[scan][0]]
                        position_dim[scan] = [
                            len(x[scan]), len(z[scan]), len(monoE[scan])]
                        self.step_width[scan] = [0., 0., 0.]
                        if len(x[scan]) != 1:
                            self.step_width[scan][0] = x[scan][1]-x[scan][0]
                        if len(z[scan]) != 1:
                            self.step_width[scan][1] = z[scan][1]-z[scan][0]
                        if len(monoE[scan]) != 1:
                            self.step_width[scan][2] = monoE[scan][1] - \
                                monoE[scan][0]
                        self.step_width[scan] = self.step_width[scan]
                        tensor_position[scan] = (positions[scan]-start_position[scan])/self.step_width[scan]
                        tensor_position[scan][tensor_position[scan]
                                              == np.inf] = 0
                        # this if a value is NaN
                        tensor_position[scan][tensor_position[scan]
                                              != tensor_position[scan]] = 0
                        tensor_position[scan] = np.around(
                            tensor_position[scan], 0).astype(int)
                    else:
                        # if only one point in the measurement
                        x[scan] = np.unique(positions[scan][0])
                        z[scan] = np.unique(positions[scan][1])
                        monoE[scan] = np.unique(positions[scan][2])
                        tensor_position[scan] = np.array([0, 0, 0])
                        position_dim[scan] = np.array([1, 1, 1])
                for scan in empty_scans:
                    x.pop(scan)
                    z.pop(scan)
                    monoE.pop(scan)
                    start_position.pop(scan)
                    position_dim.pop(scan)
                    self.step_width.pop(scan)
                    tensor_position.pop(scan)
                    self.len_scans = np.delete(self.len_scans, scan)
                    self.sum_len_scans.pop(scan)
                    positions.pop(scan)
                self.len_scans = np.asarray(self.len_scans)
                self.sum_len_scans = np.asarray(self.sum_len_scans)
            elif self.file_type == "txt":
                positions = [1, 1, len(file_list)]
                tensor_position = self.build_tensor_position(positions)
                position_dim = [1, 1, len(file_list)]
            return tensor_position, position_dim, positions
        else:
            log_file = glob(f"{self.folder_path}/*.log")[0]
            self.step_width = self.log_content[0]
            start_position = self.log_content[1]

            if self.log_file_type == "Louvre":
                tensor_position = spx.Louvre_tensor_position(
                    log_file)  # TODO test
                positions = np.asarray(list(tensor_position.values()))
                tensor_position = np.copy(positions)
                self.step_width = spx.Louvre_log_file_content(log_file)[0]
                if self.step_width[0] < 0:
                    x = np.unique(positions[:, 0])[::-1]
                else:
                    x = np.unique(positions[:, 0])
                if self.step_width[1] < 0:
                    y = np.unique(positions[:, 1])[::-1]
                else:
                    y = np.unique(positions[:, 1])
                if self.step_width[2] < 0:
                    z = np.unique(positions[:, 2])[::-1]
                else:
                    z = np.unique(positions[:, 2])
                position_dim = [len(x), len(y), len(z)]
                tensor_position[:, 0] -= x[0]
                tensor_position[:, 1] -= y[0]
                tensor_position[:, 2] -= z[0]
                tensor_position[:, 0] /= self.step_width[0]
                tensor_position[:, 1] /= self.step_width[1]
                tensor_position[:, 2] /= self.step_width[2]
                tensor_position = np.nan_to_num(tensor_position)
                positions = np.nan_to_num(positions)
                return np.around(tensor_position, 0).astype(np.int), position_dim, positions
            elif self.log_file_type == "spx or MSA":  # for .spx files recorded with M4
                tensor_position = np.empty((len(file_list), 3))
                for i in range(len(file_list)):
                    start_file_time = time.time()
                    tensor_position[i] = spx.spx_tensor_position(file_list[i])
                    end_file_time = time.time()
                    self.label_loading_progress.showMessage(
                        f"loading : {i + 1}/{len(file_list)} - {i / float(len(file_list)) * 100:.1f} %% - time to go: {time.strftime('%H:%M:%S', time.gmtime((end_file_time - start_file_time) * (len(file_list) - i)))}")
                    QtWidgets.QApplication.processEvents()

                # now the tensor_position will be normed to matrix-positions
                tensor_position = (
                    tensor_position-start_position)/self.step_width
                tensor_position[tensor_position == np.inf] = 0
                tensor_position = np.around(tensor_position, 0).astype(int)
                position_dim = self.log_content[3]
                return tensor_position, position_dim

    def npy2bin(self):
        """
        This function converts numpy .npy-files to .bin-files.
        """
        npy_filename = QtWidgets.QFileDialog.getOpenFileName()[0]
        npy_file = np.load(npy_filename)
        # now transform the npy file to bin file
        npy_file.tofile(npy_filename.replace(".npy", ".bin"))
