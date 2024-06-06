#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:53:06 2018

@author: gesa
"""
###############################################################################
import os, sys
import numpy as np
import h5py
from PyQt5 import QtWidgets, QtCore
from glob import glob
import time
###############################################################################
import spx_functions as spx    # Module to read out .spx-files
import spe_functions as spe    # Module to read out .spx-files
import mca_functions as mca    # Module to read out .spx-files
import msa_functions as msa    # Module to read out .msa-files
import txt_functions as txt    # Module to read out .txt-files
import hdf_functions as hdf    # Module to read out .hdf-files
import bcf_functions as bcf    # Module to read out .bcf-files
import csv_functions as csv    # Module to read out .csv-files
# Module to interpret angle-measurements
import angles_functions as angles
import specfit_GUI_functions as sfunc      # Module used for the specfit_GUI
###############################################################################
file_dir = os.path.dirname(os.path.abspath(__file__))
if "win" in sys.platform:
    parent_dir = "\\".join(file_dir.split("\\")[:-1])
    elements_path = parent_dir+'\\Data\\elements.dat'
else:
    parent_dir = "/".join(file_dir.split("/")[:-1])
    elements_path = parent_dir+'/Data/elements.dat'   
    
z_elements = {}
with open(elements_path, 'r') as element_file:
    for line in element_file:
        line = line.replace('\n', '').replace(' ', '').split('\t')
        z_elements[line[1]] = int(line[0])
        if line[0] == '98':
            break


class msa_thread(QtCore.QThread):
    def __init__(self, file_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.file_path = file_path

    def run(self):
        msa.msa2spec_sum_para(self.file_path,
                              self.signals.progress,
                              self.signals.sum_spec)
        self.signals.finished.emit('Done')


class spx_thread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.folder_path = folder_path

    def run(self):
        spx.many_spx2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit('Done')


class bcf_thread(QtCore.QThread):
    def __init__(self, folderpath, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.folderpath = folderpath

    def run(self):
        bcf.many_bcf2spec_para(self.folderpath,
                               self.signals.progress)
        self.signals.finished.emit('Done')


class spe_thread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.folder_path = folder_path

    def run(self):
        spe.many_spe2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit('Done')


class mca_thread(QtCore.QThread):
    def __init__(self, folder_path, XANES=False, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.folder_path = folder_path
        self.XANES = XANES

    def run(self):
        mca.many_mca2spec_para(self.folder_path,
                               XANES=self.XANES,
                               signal=self.signals.progress)
        self.signals.finished.emit('Done')


class txt_thread(QtCore.QThread):
    def __init__(self, folder_path, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.signals = thread_signals()
        self.folder_path = folder_path

    def run(self):
        txt.many_txt2spec_para(self.folder_path,
                               self.signals.progress)
        self.signals.finished.emit('Done')


class thread_signals(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    sum_spec = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)


class data_handler():
    '''tool to handle the spectra Data, including parameters'''

    def __init__(self, parent=None):
        self.reset_2_default()
        self.parent = parent

    def reset_2_default(self):
        # if this file is in the same folder as spectra file/dir it is automatically loaded
        self.default_param_file_name = 'param_save.txt'
        self.file_path = ''  # full path to spectra file
        self.folderpath = ''  # full path to dir that stores spec dict
        # 'angle_file' or 'file' or 'msa_file' or 'folder' or 'hdf5_file' or 'bcf_file
        self.loadtype = ''
        self.file_type = ''  # .spx or .txt or .msa or .bcf
        self.save_folder_path = ''
        self.save_data_folder_path = ''
        self.SpecFit_MainWindow = None
        self.life_time = 0  # given in .msa or spx file
        self.spectra = {}  # dict {1:spectra1, 2:spectra2, ...}
        self.sum_spec = np.empty(1)  # stores sum_spec
        self.len_spectrum = 0  # stores the len of the sum spec
        self.len_scans = 1 # number of scans in a measurement
        # self.energies =[] #stores the x-axis of the spec
        # stores all parameters parameters that are stored in Data = [a0,a1,Fano, FWHM_0, lifetime, max_energy, gating_time] a0+a1*x
        self.parameters = None
        try:
            # do not change parameters if already defined
            self.parameters_user[0]
        except:
            # stores parameters defined by User in GUI,
            self.parameters_user = np.zeros(8)
        # todo: mit  Frank absprechen, was wird gespeichert?
        self.use_parameters = False  # if True use parameters_user else parameter_data
        self.bg_zero = False  # do not calculate background
        self.strip_cycles = 15
        self.strip_width = 60
        self.smooth_cycles = 0
        self.smooth_width = 1
        self.ROI_start = 0.1  # investigated region start in keV
        self.ROI_end = 15.  # investigated region end in keV
        self.calc_minima = True  # prior smoothing calc of minima in spectrum
        self.calc_minima_order = 15  # order of minima algorithm
        self.plot_style = 'lin'  # lin or log
        # stores tensor positions [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]'
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
        self.use_lib = 'xraylib'  # or TUB

    def _get_file_path(self, angle_file):
        ''' 
        read out file_path utilizing a QFileDialog and determine the loadtype
        as either angle_file or file
        '''
        if angle_file:
            loadtype = 'angle_file'
        else:
            loadtype = 'file'
        try:
            # Öffnet Fenster zum anklicken von Datein. [0] gibt nur den String.
            self.file_path = self.file_dialog.getOpenFileName(
                filter='(*.spx *.MSA *.txt *.spe *.mca *.dat *.hdf5 *.h5 *.bcf *.csv)')[0]
            self.label_loading_progress.showMessage(self.file_path)
            assert os.path.exists(self.file_path)
        except:
            self.label_loading_progress.showMessage('file does not exist')
            self.file_path = 'not_found'
        return self.file_path, loadtype

    def _get_folderpath(self):
        '''
        read out the folderpath utilizing a QFileDialog
        '''
        try:
            folderpath = QtWidgets.QFileDialog.getExistingDirectory()  # function for a folder-GUI
            self.label_loading_progress.showMessage(folderpath)
            assert os.path.isdir(folderpath)
        except:
            self.label_loading_progress.showMessage('Dir not found')
            folderpath = 'not_found'
        return folderpath

    def update_channels(self):
        ''' 
        calculate the new energy axis and adjust it to the lengths of the
        loaded sum spectrum
        '''

        if self.file_type == '.bcf':
            # new variable parameter_user introduced to work with
            self.parameters_user = np.copy(self.parameters)

        try:
            self.energies = np.arange(
                self.parameters[0], self.parameters[0]+self.parameters[1]*self.len_spectrum, self.parameters[1])  # max energy
        except:
            self.energies = np.arange(self.parameters_user[0], self.parameters_user[5] +
                                      self.parameters_user[1]*self.len_spectrum, self.parameters_user[1])  # max energy
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
        with h5py.File(h5_path, 'r') as f:
            self.tensor_positions = f['tensor positions'][()]
            self.positions = f['positions'][()]
            self.position_dimension = f['position dimension'][()]
            self.sum_spec = f['sum spec'][()]
            self.parameters = f['parameters'][()]
            self.max_pixel_spec = f['max pixel spec'][()]
            self.len_spectrum = len(self.sum_spec)
            self.spectra = f['spectra'][()]

    def get_ROI_indicees(self):
        '''return Roi indices as tuple '''
        idx_min = (np.abs(np.subtract(self.energies, self.ROI_start))).argmin()
        idx_max = (np.abs(np.subtract(self.energies, self.ROI_end))).argmin()
        return idx_min, idx_max

    def open_data_file(self, angle_file=False):
        self.file_path, self.loadtype = self._get_file_path(angle_file)
        self.folderpath = self.get_folder_of_path()
        self.save_data_folder_path = f'{self.folderpath}/data'
        self.file_type = self.get_file_type()
        if self.file_type == '.MSA':
            self.save_data_folder_path = os.path.join(self.folderpath, 'data')
            self.loadtype = 'msa_file'
            # Sucht in folderpath nach alles .MSA files
            self.file_list = list(
                np.sort(glob('%s/*%s' % (self.folderpath, self.file_type))))
            self.life_time = msa.msa2life_time('%s' % self.file_path)
            self.len_spectrum = msa.msa2channels(self.file_path)

            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                        'Do you want to reload the measurement?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.run_msa_worker()  # stores parameters
                    assert self.load_stored_spec_and_param(
                    ), 'something is wrong with .MSA worker, parameters are not stored correctly'
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.sum_spec, self.parameters = msa.msa2spec_sum_para(
                    self.file_path)
                self.energies = np.arange(self.parameters[0],
                                          self.parameters[0] +
                                          self.parameters[1]*self.len_spectrum,
                                          self.parameters[1])
                assert self.load_stored_spec_and_param(
                ), 'something is wrong with .MSA worker, parameters are not stored correctly'

        elif self.file_type == '.bcf':
            self.loadtype = 'bcf_file'
            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                        'Do you want to reload the measurement?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.bcf2spec_para(
                        self.file_path)
                    self.life_time = self.parameters[4]
                    self.real_time = self.parameters[7]
                    self.parameters_user = np.copy(self.parameters)
                    self.energies = np.arange(
                        self.parameters[0], self.parameters[5], self.parameters[1])
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.bcf2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.parameters_user = np.copy(self.parameters)
                self.energies = np.arange(
                    self.parameters[0], self.parameters[5], self.parameters[1])
            self.len_spectrum = len(self.sum_spec)
                
        elif self.file_type == '.csv':
            self.loadtype = 'csv_file'
            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                        'Do you want to reload the measurement?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters = csv.csv2spec_para(self.file_path)
                    self.tensor_positions = csv.csv_tensor_positions(self.file_path)
                    self.positions = self.tensor_positions
                    self.position_dimension = csv.csv_position_dimension(self.file_path)
                    self.sum_spec = np.sum(self.spectra, axis = 0)
                    self.life_time = self.parameters[4]
                    self.real_time = self.parameters[7]
                    self.parameters_user = np.copy(self.parameters)
                    self.energies = np.arange(
                        self.parameters[0], self.parameters[5], self.parameters[1])
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters = csv.csv2spec_para(self.file_path)
                self.tensor_positions = csv.csv_tensor_positions(self.file_path)
                self.positions = self.tensor_positions
                self.position_dimension = csv.csv_position_dimension(self.file_path)
                self.sum_spec = np.sum(self.spectra, axis = 0)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.parameters_user = np.copy(self.parameters)
                self.energies = np.arange(
                    self.parameters[0], self.parameters[5], self.parameters[1])
            self.len_spectrum = len(self.sum_spec)

        elif self.file_type == '.hdf5' or self.file_type == '.h5':
            self.loadtype = 'hdf5_file'
            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                        'Do you want to reload the measurement?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    self.spectra, self.parameters, self.sum_spec, self.len_spectrum = hdf.hdf2spec_para(
                        self.file_path)
                    self.life_time = self.parameters[4]
                    self.real_time = self.parameters[7]
                    self.position_dimension, self.tensor_positions = hdf.hdf_tensor_positions(
                        self.file_path)
                    
                elif reload == QtWidgets.QMessageBox.No:
                    self.load_stored_spec_and_param()
            else:
                self.spectra, self.parameters, self.sum_spec, self.len_spectrum = hdf.hdf2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.position_dimension, 
                self.tensor_positions = hdf.hdf_tensor_positions(self.file_path)
            if self.file_type == '.hdf5':
                self.ROI_start = 0.0
                self.ROI_end = 2
            elif self.file_type == '.h5':
                self.ROI_start = 0.1
                self.ROI_end = 15

        else:
            if self.file_type == '.spx':
                self.spectra['0'], self.parameters = spx.spx2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.len_spectrum = spx.spx2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra['0']
                self.ROI_start = 0.5
                self.ROI_end = 15

            elif self.file_type == '.spe':
                self.spectra['0'], self.parameters = spe.spe2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.len_spectrum = spe.spe2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra['0']
                self.ROI_start = 0.5
                self.ROI_end = 25

            elif self.file_type == '.mca':
                self.spectra['0'], self.parameters = mca.mca2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.len_spectrum = mca.mca2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra['0']
                self.ROI_start = 0.5
                self.ROI_end = 15

            elif self.file_type == '.txt' and (not angle_file):
                self.spectra['0'], self.parameters = txt.txt2spec_para(
                    self.file_path)
                self.life_time = self.parameters[4]
                self.real_time = self.parameters[7]
                self.len_spectrum = txt.txt2channels(self.file_path)
                self.tensor_positions = np.array([1, 1, 1])
                self.position_dimension = [1, 1, 1]
                self.sum_spec = self.spectra['0']
                self.ROI_start = 0.5
                self.ROI_end = 15
                
            elif angle_file:
                if os.path.isdir(self.save_data_folder_path):
                    reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                            'Do you want to reload the measurement?',
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
                # self.parameters = np.array(self.parameters)
        self.parameters = np.array(self.parameters)
        # assert self.load_stored_spec_and_param()
        param_path = os.path.join(self.folderpath, self.default_param_file_name)
        self.max_pixel_spec = np.max(self.spectra, axis=0)
        self.check_param_file(param_path)
        self.update_channels()
        self.create_save_folder()
        if self.parameters.ndim == 1:
            self.parent.entry_a0.setText('%f' % self.parameters[0])
            self.parent.entry_a1.setText('%f' % self.parameters[1])
            self.parent.entry_fano.setText('%f' % self.parameters[2])
            self.parent.entry_FWHM.setText('%f' % self.parameters[3])
        elif self.parameters.ndim == 2:
            self.parent.entry_a0.setText('%f' % self.parameters[0][0])
            self.parent.entry_a1.setText('%f' % self.parameters[0][1])
            self.parent.entry_fano.setText('%f' % self.parameters[0][2])
            self.parent.entry_FWHM.setText('%f' % self.parameters[0][3])
        else: print('parameters has wrong dimension!')

    def open_data_folder(self):
        '''
        This function reads out the given folder, decides wether its .spx, or .txt 
        and loads the spectra, and parameters. 
        '''
        self.loadtype = 'folder'
        self.folderpath = self._get_folderpath()
        if self.folderpath != 'not_found':
            self.save_data_folder_path = os.path.join(self.folderpath, 'data')
            # if any spx in folder -> filetype spx
            self.file_type = self.determine_folder_file_type()
            # create a list of all file_type-files contained in the selected folder
            self.file_list = list(
                np.sort(glob('%s/*%s' % (self.folderpath, self.file_type))))
            if os.path.isdir(self.save_data_folder_path):
                reload = QtWidgets.QMessageBox.question(self.SpecFit_MainWindow, '?',
                                                        'Do you want to reload the measurement?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                        QtWidgets.QMessageBox.No)
                if reload == QtWidgets.QMessageBox.Yes:
                    if self.file_type == '.spx':
                        self.run_spx_worker()
                        self.ROI_start = 0.5
                    elif self.file_type == '.spe':
                        self.run_spe_worker()
                        self.ROI_start = 0.5
                        self.ROI_end = 25
                    elif self.file_type == '.bcf':
                        self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.many_bcf2spec_para(
                            self.folderpath)
                        self.ROI_start = 0.5
                    elif self.file_type == '.mca':
                        self.XANES = False
                        self.run_mca_worker(XANES=self.XANES)
                        self.ROI_start = 0.5
                    elif self.file_type == '.txt':
                        self.run_txt_worker()
                        self.ROI_start = 0.5
                    self.find_and_save_log_file()
                    if self.file_type == '.bcf':
                        self.positions = np.copy(self.tensor_positions)
                    else:  # for all different data types
                        self.tensor_positions, self.position_dimension, self.positions = self.tensor_positions_log_file(
                            self.file_list)
                    with h5py.File('%s/data.h5' % self.save_data_folder_path, 'a') as tofile:
                        if self.tensor_positions.dtype == 'O':
                            tofile.create_dataset('tensor positions', data=np.concatenate(self.tensor_positions, axis = 0))
                            tofile.create_dataset('positions', data=np.concatenate(self.positions, axis = 0))
                        else:
                            tofile.create_dataset('tensor positions', data=self.tensor_positions)
                            tofile.create_dataset('positions', data=self.positions)
                        tofile.create_dataset('position dimension', data=self.position_dimension)

                elif reload == QtWidgets.QMessageBox.No:
                    if self.load_stored_spec_and_param(folder=True):
                        pass
            else:
                if self.file_type == '.spx':
                    self.run_spx_worker()
                    self.ROI_start = 0.5
                elif self.file_type == '.bcf':
                    self.spectra, self.parameters, self.position_dimension, self.tensor_positions, self.sum_spec = bcf.many_bcf2spec_para(
                        self.folderpath)
                    self.ROI_start = 0.5
                elif self.file_type == '.spe':
                    self.run_spe_worker()
                    self.ROI_start = 0.5
                    self.ROI_end = 25
                elif self.file_type == '.mca':
                    self.XANES = False
                    self.run_mca_worker(XANES=self.XANES)
                    self.ROI_start = 0.5
                elif self.file_type == '.txt':
                    self.run_txt_worker()
                    self.ROI_start = 0.5
                self.find_and_save_log_file()
                if self.file_type == '.bcf':
                    self.positions = np.copy(self.tensor_positions)
                else:  # for all different data types
                    self.tensor_positions, self.position_dimension, self.positions = self.tensor_positions_log_file(
                        self.file_list)
                with h5py.File('%s/data.h5'%self.save_data_folder_path, 'a') as tofile:
                    if self.tensor_positions.dtype == 'O':
                        tofile.create_dataset('tensor positions', data=np.concatenate(self.tensor_positions, axis = 0))
                        tofile.create_dataset('positions', data=np.concatenate(self.positions, axis = 0))
                    else:
                        tofile.create_dataset('tensor positions', data=self.tensor_positions)
                        tofile.create_dataset('positions', data=self.positions)
                    tofile.create_dataset('position dimension', data=self.position_dimension)
                
        assert self.load_stored_spec_and_param(folder=True)
        if self.file_type == '.bcf':
            self.life_time = self.parameters[4]
            self.real_time = self.parameters[7]
            self.create_save_folder()
            self.check_param_file(os.path.join(
                self.folderpath, self.default_param_file_name))
            self.update_channels()
            self.parent.entry_a0.setText('%f' % self.parameters[0])
            self.parent.entry_a1.setText('%f' % self.parameters[1])
            self.parent.entry_fano.setText('%f' % self.parameters[2])
            self.parent.entry_FWHM.setText('%f' % self.parameters[3])

        else:  # for all other data types
            self.life_time = self.parameters[0][4]
            self.real_time = self.parameters[0][7]
            self.create_save_folder()
            self.check_param_file(os.path.join(
                self.folderpath, self.default_param_file_name))
            self.update_channels()
            self.parent.entry_a0.setText('%f' % self.parameters[0][0])
            self.parent.entry_a1.setText('%f' % self.parameters[0][1])
            self.parent.entry_fano.setText('%f' % self.parameters[0][2])
            self.parent.entry_FWHM.setText('%f' % self.parameters[0][3])
        self.max_pixel_spec = np.max(self.spectra, axis=0)
        

    def load_stored_spec_and_param(self, folder=False, stored_data=False):
        '''
        try to load previously stored data, return a boolean if this was 
        succesful or not
        '''
        if os.path.exists('%s/data.h5' % self.save_data_folder_path):
            self.label_loading_progress.showMessage('loading stored data')
            stored_data = True
            with h5py.File('%s/data.h5' % self.save_data_folder_path, 'r') as infile:
                self.tensor_positions = infile['tensor positions'][()]
                self.positions = infile['positions'][()]
                self.position_dimension = infile['position dimension'][()]
                self.sum_spec = infile['sum spec'][()]
                self.parameters = infile['parameters'][()]
                self.max_pixel_spec = infile['max pixel spec'][()]
                self.len_spectrum = len(self.sum_spec)
                self.spectra = infile['spectra'][()]
            if self.file_type == '.mca': ###!TODO
                self.len_scans = np.prod(self.position_dimension, axis = 1)
                self.sum_len_scans = [np.arange(0, self.len_scans[0])]
                for i, index in enumerate(self.len_scans[1:]):
                    self.sum_len_scans.append(np.arange(self.len_scans[i], index))
                if len(self.len_scans) > 1:
                    self.tensor_positions = np.array([[self.tensor_positions[0:self.len_scans[0]]],
                                                      [self.tensor_positions[self.len_scans[i]:index] for i, index in enumerate(self.len_scans[1:])]])
                    self.positions = np.array([[self.positions[0:self.len_scans[0]]],
                                                [self.positions[self.len_scans[i]:index] for i, index in enumerate(self.len_scans[1:])]])
            return stored_data
        if os.path.exists('%s/parameters.npy' % self.save_data_folder_path) and os.path.exists('%s/tensor_positions.npy' % self.save_data_folder_path) and os.path.exists('%s/sum_spec.npy' % self.save_data_folder_path) and os.path.exists('%s/position_dimension.npy' % self.save_data_folder_path):
            self.label_loading_progress.showMessage('loading stored data')
            if os.path.exists('%s/spectra.pickle' % self.save_data_folder_path):
                stored_data = True
                if not folder:
                    self.spectra = sfunc.open_dict_pickle(
                        '%s/spectra.pickle' % self.save_data_folder_path)
                self.tensor_positions = np.load(
                    '%s/tensor_positions.npy' % self.save_data_folder_path, allow_pickle=True)
                self.position_dimension = np.load(
                    '%s/position_dimension.npy' % self.save_data_folder_path)
                self.max_pixel_spec = np.load(
                    '%s/max_pixel_spec.npy' % self.save_data_folder_path)
                self.parameters = np.load(
                    '%s/parameters.npy' % (self.save_data_folder_path))
                self.sum_spec = np.load('%s/sum_spec.npy' %
                                        (self.save_data_folder_path))
                self.len_spectrum = len(self.sum_spec)
            elif os.path.isdir('%s/single_spectra' % self.folderpath):
                stored_data = True
                # spectra and tensor positions are not loaded here..
                self.parameters = np.load(
                    '%s/parameters.npy' % (self.save_data_folder_path))
                self.sum_spec = np.load('%s/sum_spec.npy' %
                                        (self.save_data_folder_path))
                self.len_spectrum = len(self.sum_spec)
                self.position_dimension = np.load(
                    '%s/position_dimension.npy' % self.save_data_folder_path)
        return stored_data

    def build_tensor_position(self, positions):
        ' needs list/arraywit [xmax,ymax,z_max]returns an array of Form [[x1,y1,z1],[x2,y1,z1],[x3,y1,z1], ..., [xn,yn,zn]]'
        pos_arr_1 = np.arange(positions[0])
        pos_arr_2 = np.arange(positions[1])
        pos_arr_3 = np.arange(positions[2])
        return np.array(np.meshgrid(pos_arr_1, pos_arr_2, pos_arr_3)).T.reshape(-1, 3)

    def run_msa_worker(self):
        msa_worker = msa_thread(self.file_path)
        # msa_worker.signals.progress.connect(self.run_msa_progress)
        msa_worker.start()
        msa_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_spx_worker(self):
        spx_worker = spx_thread(self.folderpath)
        spx_worker.signals.progress.connect(self.run_spx_progress)
        spx_worker.start()
        spx_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_bcf_worker(self):
        bcf_worker = bcf_thread(self.folderpath)
        bcf_worker.signals.progress.connect(self.run_bcf_progress)
        bcf_worker.start()
        bcf_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_spe_worker(self):
        spe_worker = spe_thread(self.folderpath)
        spe_worker.signals.progress.connect(self.run_spe_progress)
        spe_worker.start()
        spe_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_mca_worker(self, XANES=False):
        mca_worker = mca_thread(self.folderpath, XANES=XANES)
        mca_worker.signals.progress.connect(self.run_mca_progress)
        mca_worker.start()
        mca_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_txt_worker(self):
        txt_worker = txt_thread(self.folderpath)
        txt_worker.signals.progress.connect(self.run_txt_progress)
        txt_worker.start()
        txt_worker.wait()
        self.label_loading_progress.showMessage('loading done')

    def run_spx_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'spx-progress - %d files' % file_nr)

    def run_bcf_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'bcf-progress - %d files' % file_nr)

    def run_spe_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'spe-progress - %d files' % file_nr)

    def run_mca_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'mca-progress - %d files' % file_nr)

    def run_msa_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'msa-progress - %d files' % file_nr)

    def run_txt_progress(self, file_nr):
        self.label_loading_progress.showMessage(
            'txt-progress - %d files' % file_nr)

    def check_param_file(self, param_path):
        '''check if a file param_save.txt exists in the folder and load its content into specfit '''
        if os.path.exists(param_path):
            self.selected_elements = []
            self.selected_lines = []
            self.user_defined_lines = []
            paramfile = open(param_path, 'r')
            lines = paramfile.readlines()
            for line in lines:
                key, value = line.split(':')
                if key == 'a0':
                    self.parameters_user[0] = float(value)
                elif key == 'a1':
                    self.parameters_user[1] = float(value)
                elif key == 'fano':
                    self.parameters_user[2] = float(value)
                elif key == 'FWHM':
                    self.parameters_user[3] = float(value)
                elif key == 'use_parameters':
                    self.use_parameters = value.strip() == 'True'
                elif key == 'background=0':
                    self.bg_zero = value.strip() == 'True'
                elif key == 'strip_cycles':
                    self.strip_cycles = int(float(value.strip()))
                elif key == 'strip_width':
                    self.strip_width = int(float(value.strip()))
                elif key == 'smooth_cycles':
                    self.smooth_cycles = int(float(value.strip()))
                elif key == 'smooth_width':
                    self.smooth_width = int(float(value.strip()))
                elif key == 'ROI_start':
                    self.ROI_start = float(value.strip())
                elif key == 'ROI_end':
                    self.ROI_end = float(value.strip())
                elif key == 'log':
                    if value.strip() == 'True':
                        self.plot_style = 'log'
                elif key == 'nl-Fit':
                    self.nl_fit = value.strip() == 'True'
                elif key == 'calc_minima':
                    self.calc_minima = value.strip() == 'True'
                elif key == 'calc_minima_order':
                    self.calc_minima_order = int(value.strip())
                elif key == 'threshold':
                    self.threshold = value.strip()
                elif key == 'use_lib':
                    self.use_lib = value.strip()
                elif key == 'element':
                    line = value.strip().strip('\n').split(',')
                    elementname = line[0]
                    linelist = line[1:]
                    self.selected_elements.append(elementname)
                    self.selected_lines.append(linelist)
                elif key == 'user_defined_line':
                    label = value.strip().split(',')[0]
                    start = value.split(',')[1].strip()
                    end = value.split(',')[2].strip()
                    self.user_defined_lines.append([label, start, end])

        else:
            if self.loadtype == 'folder' and type(self.parameters[0] == list):
                self.parameters_user = self.parameters[0]

            else:
                if np.all(np.equal(self.parameters_user, np.zeros(8))):
                    self.parameters_user = self.parameters

    def save_param_file(self, filename):
        '''
        this function opens a dalog to determine a save location and saves settings and parameters
        '''
        try:
            with open(filename, 'w') as f:
                f.write('a0:%s\n' % self.parameters_user[0])
                f.write('a1:%s\n' % self.parameters_user[1])
                f.write('fano:%s\n' % self.parameters_user[2])
                f.write('FWHM:%s\n' % self.parameters_user[3])
                f.write('use_parameters:%s\n' % self.use_parameters)
                f.write('background=0:%s\n' % (self.bg_zero))
                f.write('strip_cycles:%s\n' % self.strip_cycles)
                f.write('strip_width:%s\n' % self.strip_width)
                f.write('smooth_cycles:%s\n' % self.smooth_cycles)
                f.write('smooth_width:%s\n' % self.smooth_width)
                f.write('ROI_start:%s\n' % self.ROI_start)
                f.write('ROI_end:%s\n' % self.ROI_end)
                f.write('log:%s\n' % (self.plot_style == 'log'))
                f.write('nl-Fit:%s\n' % (self.nl_fit))
                f.write('calc_minima:%s\n' % (self.calc_minima))
                f.write('calc_minima_order:%d\n' % (self.calc_minima_order))
                f.write('threshold:%s\n' % (str(self.threshold)))
                f.write('use_lib:%s' % self.use_lib)
                for sel_element, sel_lines in zip(self.selected_elements, self.selected_lines):
                    linestring = ','.join(sel_lines)
                    f.write('\nelement: %s,%s' % (sel_element, linestring))
                for label, start, end in self.user_defined_lines:
                    f.write('\nuser_defined_line: %s,%s,%s' %
                            (label, start, end))
                outstring = 'saved parameters in %s' % filename
        except:
            outstring = 'could not save parameters'

        return outstring

    def save_npy_probs(self):
        '''save data in folder_path/data.   '''
        if not os.path.exists(self.save_data_folder_path):
            os.mkdir(self.save_data_folder_path)
        with h5py.File('%s/data.h5' % self.save_data_folder_path, 'a') as tofile:
            if 'counts' in list(tofile.keys()):
                tofile['counts'][()] = self.counts
                tofile['parameters'][()] = self.parameters
                tofile['sum spec'][()] = self.sum_spec
                tofile['max pixel spec'][()] = self.max_pixel_spec
                tofile['spectra'][()] = self.spectra
            else:
                tofile.create_dataset('counts', data=self.counts)
                tofile.create_dataset('parameters', data=self.parameters)
                tofile.create_dataset('sum spec', data=self.sum_spec)
                tofile.create_dataset('max pixel spec', data=self.max_pixel_spec)
                tofile.create_dataset('spectra', data=self.spectra)

    def remove_element(self, label):
        self.selected_elements = np.array(self.selected_elements)
        try:
            indicee = np.where(self.selected_elements == label)[0][0]
        except IndexError:
            self.label_loading_progress.showMessage(
                'element is not in selected_element_list')
        else:
            del self.selected_elements[indicee]
            del self.selected_lines[indicee]

    def get_label_working_folder(self):
        if (self.loadtype == 'angle_file') or (self.loadtype == 'file'):  # or msa_file or folder??
            working_folder = ''
        return working_folder

    def get_file_type(self):
        return os.path.splitext(self.file_path)[1]

    def determine_folder_file_type(self):
        file_type = ''
        if len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.spx'))))) != 0:
            file_type = '.spx'
        elif len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.spe'))))) != 0:
            file_type = '.spe'
        elif len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.mca'))))) != 0:
            file_type = '.mca'
        elif len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.txt'))))) != 0:
            file_type = '.txt'
        elif len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.bcf'))))) != 0:
            file_type = '.bcf'
        elif len(list(np.sort(glob('%s/*%s' % (self.folderpath, '.csv'))))) != 0:
            file_type = '.csv'
        return file_type

    def get_lifetime(self):
        return self.life_time

    def get_folder_of_path(self):
        return os.path.dirname(self.file_path)

    def find_and_save_log_file(self):
        try:
            log_file = glob('%s/*.log' %
                            self.folderpath)[0]  # get the .log-file
        except:
            self.log_file_type = None
            return  # if none .log is provided, the results array is 1D and tried to be sorted by name
        else:
            try:  # check wether the .log file is the standart or the Louvre type
                self.log_content = spx.spx_log_content(log_file)
                self.log_file_type = 'spx or MSA'
            except:
                self.log_content = spx.Louvre_log_file_content(log_file)
                self.log_file_type = 'Louvre'

    def create_save_folder(self):
        if self.loadtype in ['file', 'msa_file', 'angle_file', 'hdf5_file', 'bcf_file',
                             'csv_file']:
            self.save_folder_path = '%s/results' % (self.folderpath)
        else:
            self.save_folder_path = '%s/results' % (self.folderpath)
        try:
            os.mkdir(self.save_folder_path)
        except:
            pass

    def build_specfit_addlines(self):
        # builds dict{'element':(True/False, ['linename',...], Z), '':...,}
        ### example: {'Cu': (True, ['K-line', 'L1'], 29)}
        self.specfit_addlines = {}
        for element, lines in zip(self.selected_elements, self.selected_lines):
            self.specfit_addlines['%s' % element] = (
                True, lines, z_elements[element])

    def tensor_positions_log_file(self, file_list):  # TODO
        ''' return tensor_positions and position_dim  depending on log file'''
        if self.log_file_type == None:
            if self.file_type == '.spx':
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
            elif self.file_type == '.spe':
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
            elif self.file_type == '.mca':
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
            elif self.file_type == 'txt':
                positions = [1, 1, len(file_list)]
                tensor_position = self.build_tensor_position(positions)
                position_dim = [1, 1, len(file_list)]
            return np.asarray(tensor_position), position_dim, np.nan_to_num(positions)
        else:
            log_file = glob('%s/*.log' % self.folderpath)[0]
            self.step_width = self.log_content[0]
            start_position = self.log_content[1]

            if self.log_file_type == 'Louvre':
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
            elif self.log_file_type == 'spx or MSA':  # for .spx files recorded with M4
                tensor_position = np.empty((len(file_list), 3))
                for i in range(len(file_list)):
                    start_file_time = time.time()
                    tensor_position[i] = spx.spx_tensor_position(file_list[i])
                    end_file_time = time.time()
                    self.label_loading_progress.showMessage('loading : %d/%d - %.1f %% - time to go: %s' % (i+1, len(file_list), i/float(
                        len(file_list))*100, time.strftime('%H:%M:%S', time.gmtime((end_file_time-start_file_time)*(len(file_list)-i)))))
                    QtWidgets.QApplication.processEvents()

                # now the tensor_position will be normed to matrix-positions
                tensor_position = (
                    tensor_position-start_position)/self.step_width
                tensor_position[tensor_position == np.inf] = 0
                tensor_position = np.around(tensor_position, 0).astype(int)
                position_dim = self.log_content[3]
                return tensor_position, position_dim

    def npy2bin(self):
        ''' This function converts numpy .npy-files to .bin-files.
        '''
        npy_filename = QtWidgets.QFileDialog.getOpenFileName()[0]
        npy_file = np.load(npy_filename)
        # now transform the npy file to bin file
        npy_file.tofile(npy_filename.replace('.npy', '.bin'))
