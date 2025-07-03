import sys
import os
import gc
import traceback
import h5py
import time
from glob import glob
import numpy as np
import dask.array as da
import subprocess as sp 
import natsort as ns
from PyQt6 import QtGui, QtWidgets, QtCore
from functools import partial
import matplotlib.backends.backend_qt5agg as pltqt
from matplotlib import figure
import matplotlib.image as mpimg
import xraylib as xrl
script_dir = os.path.dirname(os.path.abspath(__file__))
functions_path = os.path.join(script_dir, "functions")
data_path = os.path.join(script_dir, "Data")
sys.path.append(functions_path)
sys.path.append(data_path)
from fit_range_popup import FitThresholdPopup
from show_ROI import ShowROI
import specfit_deconvolution as sf # Module to fit lines to spectrum
import specfit_GUI_functions as sfunc # Module used for the specfit_GUI
from data_handler import DataHandler
import show_measurement_properties # Module to print the measurement properties in a text widget
from plot3d import Plot3D
from display_meas_points import DisplayMeasPoints as dmp
from periodic_table import PeriodicTable as PSE
import specfit_fit_settings as sfs
from split_h5 import split_h5
from export_functions import h5_to_tiff
from ipython_console import IPythonConsole # ipython console implementation for SpecFit

class RoDatabase(object):
    """
    This class performs all database readout processes
    """
    def __init__(self):
        """
        Initialise and execute complete readout
        """
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.read_elements()
        self.read_lines()
        self.read_PSE()
        self.init_check_elements()
        self.read_char_lines()
        self.read_lineE()
    
    def read_elements(self):
        """
        read out list of elements from elements.dat
        """
        self.elements = []
        with open(self.working_dir+"/Data/elements.dat", "r", 
                  encoding="ascii") as f:
            for line in f:
                line = line.replace("\n", "").replace(" ", "").split("\t")
                self.elements.append([int(line[0]), line[1], int(line[0])])
                if line[0] == "98":
                    break
    
    def read_lines(self):
        """
        read out list of lines from lines.dat
        """
        self.lines = []
        with open(self.working_dir+"/Data/lines.dat", "r", 
                  encoding="ascii") as f:
            for line in f:
                line = line.replace("\n", "").split("\t")
                self.lines.append([line[0], line[1]])
    
    def read_PSE(self):
        """
        read out structure of PSE from PSE.dat
        """
        self.pse = []
        with open(self.working_dir+"/Data/PSE.dat", "r", 
                  encoding="ascii") as f:
            for line in f:
                line = line.replace("\n", "").split("\t")
                self.pse.append([int(line[0]), line[1], int(line[2]), int(line[3]), int(line[4]), line[5]])
                if line[0] == "98":
                    break
    
    def init_check_elements(self):
        """
        here a list to determine wether a element is choosen or not is created
        """
        self.check_elements = []
        for i in range(len(self.elements)):
            self.check_elements.append(f"check_element_{i}")
    
    def read_char_lines(self):
        """
        read out characteristic lines
        """
        self.characteristic_lines = {}
        with open(self.working_dir+"/Data/characteristic_lines.dat", "r", 
                  encoding="ascii") as f:
            for line in f:
                line = line.split()
                self.characteristic_lines[line.pop(0)] = line
    def read_lineE(self):
        """
        read out fluorescence energy data from lineE.dat
        """
        self.lineE = {}
        with open(self.working_dir+"/Data/lineE.dat", "r", 
                  encoding="ascii") as lineE_file:
            for line in lineE_file:
                line = line.split()
                self.lineE[float(line[0])] = [line[1], int(line[2]), 
                                              line[3], int(line[4]), float(line[5])]

database = RoDatabase()

class SpecFitGUIMain(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        # initalize main window
        super(SpecFitGUIMain, self).__init__(parent)
        self.__version__ = u"SpecFit - 1.0"
        self.working_directory = os.getcwd()
        self.bg_color = "black"
        sp.call("", shell=True) # clear the screen to get rid of all the print
        # implementing database
        self.data = DataHandler(parent=self)
        self.data.SpecFit_MainWindow = self
        self.s = sf.SpecFit()  # load the SpecFit class of specfit_deconvolution.py
        self.roi_widget = ShowROI(parent=self)
        self.plot3d = Plot3D()
        self.pse_widget = PSE(parent=self, 
                              PSE=database.pse, 
                              elements=database.elements, 
                              lines=database.lines, 
                              check_elements=database.check_elements, 
                              characteristic_lines=database.characteristic_lines, 
                              lineE=database.lineE)
        self.popup_properties = show_measurement_properties.ShowProps(os.getcwd() + "/example_measurements/spx/1d/", self)
        self.ascii_specfit_logo = mpimg.imread(os.getcwd()+"/Data/images/SpecFit_ASCII.png")
        self.elements = database.elements  #: list of elements
        self.lines = database.lines  #: list of lines for ex. K, Ka, ..
        self.pse = database.pse  #: list with definitions of the PSE
        self.characteristic_lines = database.characteristic_lines  #: list of characteristic X-ray lines
        self.check_lines_list = []  #: list of checkable lines
        self.check_line_labels = []  #: list of check_line_labels
        self.check_elements = database.check_elements  #: list to determine the selected elements
        # set default stylesheets
        self.setStyleSheet("QWidget { color: black; background-color:white;}"\
                           +"color:black;"\
                           +"background-color:white;"\
                           +"border-width: 10px;"\
                           +"QLabel {font-size: 11px;} "\
                           +"QLineEdit {font-size: 11px; max-height: 18px;} "\
                           +"QCheckBox {font-size: 11px; max-height: 18px;} "\
                           +"QPushButton {font-size: 11px; max-height: 18px;}"\
                           +"QToolButton {color:black; background-color:grey; border: 1px solid #ccc;}"\
                           +"QRadioButton {font-size: 11px; max-height: 18px;} "\
                           +"QComboBox {font-size: 11px; max-height: 18px;} "\
                           +"QTextEdit {font-size: 11px}")
        self.row_height = 18
        self.button_width = 80
        self.label_width = 40
        # define geometry and properties
        
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.window_heigth = 600
        self.window_width = 1200
        self.setWindowTitle(self.__version__)
        self.setGeometry(int((self.screen_width-self.window_width)//2), 
                         int((self.screen_height-self.window_heigth)//2), 
                         self.window_width, self.window_heigth)
        self.menubar = self.menuBar()
        self.win_toolbar = self.addToolBar("SpecFit")
        self.win_toolbar.setIconSize(QtCore.QSize(30, 30))
        self.win_toolbar.setFloatable(False)
        self.win_toolbar.setMovable(False)
        # initialise all sub-GUIs and parameter
        self.__init__UI()
        self.__init__plot()
        self.__init__layout()
        self.sfs = sfs.SpecFitFitSettings(_parent=self)
        self.__init_connections()
        self.reset_2_default()
        self.activateWindow()

    def __init__UI(self):
        """
        Initialisation of the UI-layout
        """
        # set actions connected to the menu or toolbar
        # load folder action
        self.action_load_folder = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/folder-blue-open-icon.png"), "load folder", self)
        self.action_load_folder.setShortcut("Ctrl+O")
        self.action_load_folder.setStatusTip("load measurement folder (.txt or .spx)")
        self.action_load_folder.triggered.connect(self.load_folder)
        # load file action
        self.action_load_file = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/file-open.png"), "load file", self)
        self.action_load_file.setShortcut("Ctrl+Shift+O")
        self.action_load_file.setStatusTip("load measurement file")
        self.action_load_file.triggered.connect(lambda: self.load_file(angle_file=False) )
        # load angle file action
        self.action_load_angle = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/angle-open.png"), "load angle-file", self)
        self.action_load_angle.setShortcut("Ctrl+Shift+A")
        self.action_load_angle.setStatusTip("load angle file (for GI or GE measurements)")
        self.action_load_angle.triggered.connect(lambda: self.load_file(angle_file=True) )
        # load settings action
        self.action_load_settings = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/settings-open.png"), "load setting-file", self)
        self.action_load_settings.setShortcut("Ctrl+Shift+S")
        self.action_load_settings.setStatusTip("load settings file")
        self.action_load_settings.triggered.connect(self.open_param_file)
        # save settings action
        self.action_save_settings = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/settings-save.png"), "save settings", self)
        self.action_save_settings.setShortcut("Ctrl+Shift+P")
        self.action_save_settings.setStatusTip("save settings in file")
        self.action_save_settings.triggered.connect(self.save_param_file)
        # open fit settings dialog action
        self.action_fit_settings = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/settings-save.png"), "set fit settings", self)
        self.action_fit_settings.setShortcut("Ctrl+S")
        self.action_fit_settings.setStatusTip("fit settings")
        self.action_fit_settings.triggered.connect(self.show_specfit_fit_settings)
        # check fit action
        self.action_check_fit = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/check-fit.png"), "check fit", self)
        self.action_check_fit.setShortcut("F5")
        self.action_check_fit.setStatusTip("check fit")
        self.action_check_fit.triggered.connect(self.check_fit)
        # view ROI action
        self.action_roi_image = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/show-ROI.png"), "view ROI", self)
        self.action_roi_image.setShortcut("F6")
        self.action_roi_image.setStatusTip("view ROI (F6)")
        self.action_roi_image.triggered.connect(self.fill_and_display_show_roi)
        # plot maximum pixel spectrum action
        self.action_max_pixel_spec = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/video-display.png"), "maximum pixel spectrum", self)
        self.action_max_pixel_spec.setStatusTip("plot maximum pixel spec")
        self.action_max_pixel_spec.triggered.connect(lambda: self.plot_canvas(self.data.max_pixel_spec, self.data.energies, None, None ))
        self.action_max_pixel_spec.triggered.connect(lambda: self.set_spectra_nr(-2))
        # show 3d plot action
        self.action_show_plot3d = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/plot-3d.png"), "plot 3D", self)
        self.action_show_plot3d.setShortcut("F8")
        self.action_show_plot3d.setStatusTip("plot 3D")
        self.action_show_plot3d.triggered.connect(self.show_plot3d)
        # plot meas points action
        self.action_show_display_meas_points = QtGui.QAction("display meas points 3D", self)
        self.action_show_display_meas_points.setStatusTip("display measurement points 3D")
        self.action_show_display_meas_points.triggered.connect(self.show_display_meas_points)
        # clear plot action
        self.action_clear_plot = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/clear-plot.png"), "clear plot", self)
        self.action_clear_plot.setShortcut("F4")
        self.action_clear_plot.setStatusTip("clear plot")
        self.action_clear_plot.triggered.connect(self.clear_plot)
        # clear element selection action
        self.action_clear_element_lines = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/clear-elements.png"), "clear elements and lines", self)
        self.action_clear_element_lines.setShortcut("F2")
        self.action_clear_element_lines.setStatusTip("clear elements and lines")
        self.action_clear_element_lines.triggered.connect(self.pse_widget.clear_element_line_list)
        self.action_clear_element_lines.triggered.connect(self.clear_ax_canvas_spectrum)
        # fit action
        self.action_fit = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/fit-and-save.png"), "fit and save", self)
        self.action_fit.setShortcut("F1")
        self.action_fit.setStatusTip("fit and save")
        self.action_fit.triggered.connect(self.fit_folder)
        # npy to bin conversion action
        self.action_npy_2_bin = QtGui.QAction("convert .npy to .bin", self)
        self.action_npy_2_bin.setStatusTip("display measurement points 3D")
        self.action_npy_2_bin.triggered.connect(self.npy_2_bin)
        # save to npz action
        self.action_export_2_npz = QtGui.QAction("export .npy", self)
        self.action_export_2_npz.setStatusTip("export measurement to .npy")
        self.action_export_2_npz.triggered.connect(self.export_2_npz)
        # split data.h5 for batch fitting action
        self.action_split_datah5 = QtGui.QAction("split data.h5", self)
        self.action_split_datah5.setStatusTip("split the data.h5 to fit in batches")
        self.action_split_datah5.triggered.connect(self.split_h5)
        # results.h5 to tiff action
        self.action_h5_2_tiff = QtGui.QAction("results.h5 to .tiff", self)
        self.action_h5_2_tiff.setStatusTip("transform results.h5 to tiff images")
        self.action_h5_2_tiff.triggered.connect(self.h5_2_tiff)
        # show counts action
        self.action_show_counts = QtGui.QAction("show counts", self)
        self.action_show_counts.setStatusTip("show counts")
        self.action_show_counts.triggered.connect(self.fill_and_show_fit_threshold_widget)
        # start debugging console action
        self.action_ipython_console = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/bug.png"), "start ipython console", self)
        self.action_ipython_console.setStatusTip("start ipython console")
        self.action_ipython_console.triggered.connect(self.ipython_console)
        # exit action
        self.action_exit = QtGui.QAction(QtGui.QIcon(self.working_directory+"/Data/icons/exit.png"), "Exit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.setStatusTip("Exit application with Ctrl+Q")
        self.action_exit.triggered.connect(QtWidgets.QApplication.quit)
        # set StatusBar
        self.statusBar()
        # set Menubar and add actions
        # File menu
        self.menu_file = self.menubar.addMenu("&File")
        self.menu_file.addAction(self.action_load_folder)
        self.menu_file.addAction(self.action_load_file)
        self.menu_file.addAction(self.action_load_angle)
        self.menu_file.addAction(self.action_exit)
        # Settings menu
        self.menu_settings = self.menubar.addMenu("&Settings")
        self.menu_settings.addAction(self.action_load_settings)
        self.menu_settings.addAction(self.action_save_settings)
        self.menu_settings.addAction(self.action_fit_settings)
        # Fit menu
        self.menu_fit = self.menubar.addMenu("&Fit")
        self.menu_fit.addAction(self.action_fit)
        self.menu_fit.addAction(self.action_check_fit)
        self.menu_fit.addAction(self.action_clear_plot)
        self.menu_fit.addAction(self.action_clear_element_lines)
        # data menu
        self.menu_data = self.menubar.addMenu("&Data")
        self.menu_data.addAction(self.action_roi_image)
        self.menu_data.addAction(self.action_max_pixel_spec)
        self.menu_data.addAction(self.action_show_plot3d)
        self.menu_data.addAction(self.action_show_display_meas_points)
        self.menu_data.addAction(self.action_show_counts)
        # Export menu
        self.menu_export = self.menubar.addMenu("&Export")
        self.menu_export.addAction(self.action_export_2_npz)
        self.menu_export.addAction(self.action_npy_2_bin)
        self.menu_export.addAction(self.action_split_datah5)
        self.menu_export.addAction(self.action_h5_2_tiff)
        # edit menu
        self.menu_edit = self.menubar.addMenu("&Edit")
        self.menu_edit.addAction(self.action_ipython_console)
        # set Toolbar and add actions
        self.win_toolbar.addAction(self.action_load_folder)
        self.win_toolbar.addAction(self.action_load_file)
        self.win_toolbar.addAction(self.action_load_angle)
        self.win_toolbar.addAction(self.action_load_settings)
        self.win_toolbar.addAction(self.action_save_settings)
        self.win_toolbar.addAction(self.action_check_fit)
        self.win_toolbar.addAction(self.action_clear_plot)
        self.win_toolbar.addAction(self.action_clear_element_lines)
        self.win_toolbar.addAction(self.action_roi_image)
        self.win_toolbar.addAction(self.action_show_plot3d)
        self.win_toolbar.addAction(self.action_fit)
        self.win_toolbar.addAction(self.action_ipython_console)
        self.win_toolbar.addAction(self.action_exit)
        # create energy and fit parameter widget
        self.energy_and_para_widget = QtWidgets.QWidget()
        self.energy_and_para_widget.setFixedWidth(390)
        self.energy_and_para_widget_layout = QtWidgets.QGridLayout()
        self.energy_and_para_widget_layout.setVerticalSpacing(0)
        self.energy_and_para_widget.setLayout(self.energy_and_para_widget_layout)
        # create fit button and parameter widget
        self.fit_widget = QtWidgets.QWidget()
        self.fit_widget_layout = QtWidgets.QGridLayout()
        self.fit_widget_layout.setSpacing(0)
        self.fit_widget.setLayout(self.fit_widget_layout)
        # define labels
        self.label_a0 = QtWidgets.QLabel("a0", self)
        self.label_a1 = QtWidgets.QLabel("a1", self)
        self.label_fano = QtWidgets.QLabel("fano", self)
        self.label_fwhm = QtWidgets.QLabel("el.noise", self)
        self.label_strip_cycles = QtWidgets.QLabel("str cycles", self)
        self.label_strip_width = QtWidgets.QLabel("str width", self)
        self.label_smooth_cycles = QtWidgets.QLabel("sm cycles", self)
        self.label_smooth_width = QtWidgets.QLabel("sm width", self)
        self.label_roi_start = QtWidgets.QLabel("ROI start", self)
        self.label_roi_end = QtWidgets.QLabel("ROI end", self)
        self.label_calc_minima_order = QtWidgets.QLabel("min order", self)
        self.label_nl_fit_threshold = QtWidgets.QLabel("threshold", self)
        # set position of labels
        self.energy_and_para_widget_layout.addWidget(self.label_a0, 0, 0, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_a1, 1, 0, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_fano, 2, 0, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_fwhm, 3, 0, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_strip_cycles, 0, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_strip_width, 1, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_smooth_cycles, 2, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_smooth_width, 3, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_roi_start, 4, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_roi_end, 5, 2, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.label_calc_minima_order, 6, 2, 1, 1)
        self.fit_widget_layout.addWidget(self.label_nl_fit_threshold, 0, 1)
        # define checkboxes
        self.check_use_parameters = QtWidgets.QCheckBox("GUI-para", self)
        self.check_set_bg_zero = QtWidgets.QCheckBox("bg = 0", self)
        self.check_calc_minima = QtWidgets.QCheckBox("calc minima", self)
        self.check_calc_minima.setCheckState(QtCore.Qt.CheckState.Checked)
        self.check_calc_minima.clicked.connect(self.hide_calc_min)
        self.check_nl_fit = QtWidgets.QCheckBox("nl-fit", self)
        # set position of checkboxes
        self.energy_and_para_widget_layout.addWidget(self.check_use_parameters, 4, 0, 1, 2)
        self.energy_and_para_widget_layout.addWidget(self.check_set_bg_zero, 5, 0, 1, 2)
        self.energy_and_para_widget_layout.addWidget(self.check_calc_minima, 6, 0, 1, 2)
        self.fit_widget_layout.addWidget(self.check_nl_fit, 0, 0)
        # define entry fields
        self.entry_a0 = QtWidgets.QLineEdit("0", self)
        self.roi_widget.para_a0 = self.entry_a0 # inherit this entry to the show_ROI widget to be able to read the energy calibration into show_ROI
        self.entry_a1 = QtWidgets.QLineEdit("0.1", self)
        self.roi_widget.para_a1 = self.entry_a1 # inherit this entry to the show_ROI widget to be able to read the energy calibration into show_ROI
        self.entry_fano = QtWidgets.QLineEdit("0", self)
        self.entry_FWHM = QtWidgets.QLineEdit("0", self)
        self.entry_strip_cycles = QtWidgets.QLineEdit("5", self)
        self.entry_strip_width = QtWidgets.QLineEdit("60", self)
        self.entry_smooth_cycles = QtWidgets.QLineEdit("0", self)
        self.entry_smooth_width = QtWidgets.QLineEdit("1", self)
        self.entry_roi_start = QtWidgets.QLineEdit("1", self)
        self.roi_widget.roi_low = self.entry_roi_start # inherit this entry to the show_ROI widget to be able to read the energy range into show_ROI
        self.entry_roi_end = QtWidgets.QLineEdit("15", self)
        self.roi_widget.roi_high = self.entry_roi_end # inherit this entry to the show_ROI widget to be able to read the energy range into show_ROI
        self.entry_nl_fit_threshold = QtWidgets.QLineEdit(f"{self.s.minchange}", self)
        self.entry_calc_minima_order = QtWidgets.QLineEdit("15", self)
        self.entry_strip_cycles.setMaximumWidth(40)
        self.entry_strip_width.setMaximumWidth(40)
        self.entry_smooth_cycles.setMaximumWidth(40)
        self.entry_smooth_width.setMaximumWidth(40)
        self.entry_roi_start.setMaximumWidth(40)
        self.entry_roi_end.setMaximumWidth(40)
        self.entry_calc_minima_order.setMaximumWidth(40)
        # set position of entry fields
        self.energy_and_para_widget_layout.addWidget(self.entry_a0, 0, 1)
        self.energy_and_para_widget_layout.addWidget(self.entry_a1, 1, 1)
        self.energy_and_para_widget_layout.addWidget(self.entry_fano, 2, 1)
        self.energy_and_para_widget_layout.addWidget(self.entry_FWHM, 3, 1)
        self.energy_and_para_widget_layout.addWidget(self.entry_strip_cycles, 0, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_strip_width, 1, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_smooth_cycles, 2, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_smooth_width, 3, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_roi_start, 4, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_roi_end, 5, 3)
        self.energy_and_para_widget_layout.addWidget(self.entry_calc_minima_order, 6, 3)
        self.fit_widget_layout.addWidget(self.entry_nl_fit_threshold, 0, 2)
        # define buttons
        self.button_check_fit = QtWidgets.QPushButton("Check-Fit", self)
        self.button_clear_check_fit = QtWidgets.QPushButton("clear Check-Fit", self)
        self.button_fit_and_save = QtWidgets.QPushButton("fit and save", self)
        # set functions to buttons
        self.button_check_fit.clicked.connect(self.check_fit)
        self.button_clear_check_fit.clicked.connect(self.clear_plot)
        self.button_fit_and_save.clicked.connect(self.fit_folder)
        # set position of buttons
        self.fit_widget_layout.addWidget(self.button_check_fit, 1, 0)
        self.fit_widget_layout.addWidget(self.button_clear_check_fit, 1, 1)
        self.fit_widget_layout.addWidget(self.button_fit_and_save, 2, 0)
        self.button_fit_and_save.setStyleSheet("QWidget {background-color:lightblue}")
        # define dropdown
        self.combo_lib = QtWidgets.QComboBox(self)
        self.fit_widget_layout.addWidget(self.combo_lib, 1, 2)
        self.combo_lib.addItems(["xraylib"])
        # set ToolTips to Widgets
        self.button_check_fit.setToolTip("fit the sum spectrum with the choosen parameters")
        self.button_clear_check_fit.setToolTip("redraw sum spectrum and background")
        self.button_fit_and_save.setToolTip("Fit the loaded spectra and save the results")
        # create PSE
        self.pse_widget.tabs.currentChanged.connect(self.check_fit)
        self.data.label_loading_progress = self.statusBar()

    def __init__plot(self):
        """
        Initialise the layout of the plot frame
        """
        self.plot_widget = QtWidgets.QWidget()
        self.plot_widget_layout = QtWidgets.QGridLayout()
        self.plot_widget_layout.setSpacing(0)
        self.plot_widget.setLayout(self.plot_widget_layout)
        self.figure_sum_spec = figure.Figure()
        self.canvas_spectrum= pltqt.FigureCanvasQTAgg(self.figure_sum_spec)
        # define radiobuttons
        # define checkbuttons ###
        self.check_auto_scale = QtWidgets.QCheckBox('auto scale', self)
        self.check_auto_scale.setChecked(True)
        self.plot_widget_layout.addWidget(self.check_auto_scale, 1, 3)
        self.radio_semilogy = QtWidgets.QRadioButton("log", self)
        self.radio_linear = QtWidgets.QRadioButton("linear", self)
        self.radio_linear.setChecked(True)
        self.radio_none = QtWidgets.QRadioButton("None", self)
        # set position of radiobutton
        self.plot_widget_layout.addWidget(self.radio_semilogy, 1, 0)
        self.plot_widget_layout.addWidget(self.radio_linear, 1, 1)
        self.plot_widget_layout.addWidget(self.radio_none, 1, 2)
        # set functions of radiobuttons
        self.radio_semilogy.clicked.connect(partial(self.change_plot_style, "log"))
        self.radio_linear.clicked.connect(partial(self.change_plot_style, "lin"))
        self.radio_none.clicked.connect(partial(self.change_plot_style, "None"))
        # this is the Navigation widget for matplotlib plots
        self.toolbar_sum_spec = pltqt.NavigationToolbar2QT(self.canvas_spectrum, self)
        self.toolbar_sum_spec.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        # establish connections with User action
        self.canvas_spectrum.mpl_connect("button_press_event", self._on_button_press)
        # create an axis
        self.ax_canvas_spectrum = self.figure_sum_spec.add_subplot(111)
        self.ax_canvas_spectrum.set_title("Sum Spectrum")
        self.ax_canvas_spectrum.set_xlabel("Energy / keV")
        self.ax_canvas_spectrum.set_ylabel("Intensity / cps")
        self.ax_canvas_spectrum.set_xlim(0, 40)
        self.figure_sum_spec.tight_layout()
        self.plot_style = self.ax_canvas_spectrum.plot
        self.plot_style_changed = False
        self.roi_widget.main_GUI_plot_ax = self.ax_canvas_spectrum
        self.roi_widget.main_GUI_plot_canvas = self.canvas_spectrum
        self.roi_widget.plot_style = self.plot_style
        self.selected_energy = None
        self.selected_spectrum = None  #: container to store the current displayed spectrum, None as initialized, otherwise numpy array
        self.plot_widget_layout.setColumnStretch(3, 1)
        self.plot_widget_layout.addWidget(self.canvas_spectrum, 0, 0, 1, 4)
        self.plot_widget_layout.addWidget(self.toolbar_sum_spec, 2, 0, 1, 4)
        # access from PSE
        self.pse_widget.canvas_spectrum = self.canvas_spectrum
        self.pse_widget.ax_canvas_spectrum = self.ax_canvas_spectrum

    def __init__layout(self):
        # left Splitter
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_splitter.setFrameStyle(0)
        self.left_splitter.addWidget(self.energy_and_para_widget)
        self.left_splitter.addWidget(self.pse_widget)
        self.left_splitter.addWidget(self.fit_widget)
        # right Splitter
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.right_splitter.setMaximumWidth(200)
        self.right_splitter.setFrameStyle(0)
        self.right_splitter.setStretchFactor(0, 0)
        self.right_splitter.addWidget(self.popup_properties)
        # SpecFit Splitter
        self.specfit_gui_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.specfit_gui_splitter.setMaximumSize(self.screen_width, self.screen_height)
        self.specfit_gui_splitter.addWidget(self.left_splitter)
        self.specfit_gui_splitter.addWidget(self.plot_widget)
        self.specfit_gui_splitter.addWidget(self.right_splitter)
        # set SpecFit Splitter as Main Widget
        self.setCentralWidget(self.specfit_gui_splitter)

    def __init_connections(self, ):
        self.entry_a0.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_a0))
        self.entry_a1.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_a1))
        self.entry_fano.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_fano))
        self.entry_FWHM.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_FWHM))
        self.entry_strip_cycles.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_strip_cycles))
        self.entry_strip_width.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_strip_width))
        self.entry_smooth_cycles.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_smooth_cycles))
        self.entry_smooth_width.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_smooth_width))
        self.entry_calc_minima_order.textChanged.connect(partial(self.sfs.update_entries, self.sfs.entry_calc_minima_order))
        self.check_calc_minima.stateChanged.connect(lambda state: self.sfs.check_calc_minima.setChecked(state == QtCore.Qt.CheckState.CheckState.Checked))

    def set_spectra_nr(self, spectra_nr):
        """
        change the self.spectra_nr to the designated value
        introduced to distinguish between max_pixel spectrum and other"""
        self.spectra_nr = spectra_nr
        if spectra_nr == -1:
            self.selected_spectrum = self.data.max_pixel_spec
        elif spectra_nr < -1:
            self.selected_spectrum = self.data.sum_spec
        else:
            self.selected_spectrum = self.data.spectra[f"{spectra_nr}"]

    def hide_calc_min(self, ):
        """
        hide or show min_order entry
        """
        if self.check_calc_minima.isChecked():
            self.entry_calc_minima_order.show()
            self.label_calc_minima_order.show()
        else:
            self.entry_calc_minima_order.hide()
            self.label_calc_minima_order.hide()

    def load_folder(self):
        try:
            self.reset_2_default()
            self.data.elements = self.elements
            self.data.label_loading_progress = self.statusBar()
            self.data.open_data_folder()
            self.threshold_handler = FitThresholdPopup(self.data.folder_path)
            self.popup_properties.data_path = self.data.save_folder_path
            self.right_splitter.addWidget(self.popup_properties)
            self.popup_properties.fill_text(f"folder path:\n {self.data.folder_path}\n")
            self.popup_properties.fill_text(f"files: {len(self.data.file_list)}\n")
            self.load_parameter_in_specfit_deconvolution()
            if self.data.file_type == ".bcf":
                self.s.life_time = self.data.parameters[4]
            else:
                self.s.life_time = self.data.parameters[0][4]
            try: 
                self.s.real_time = self.data.parameters[0][7]  #TODO real time has to be implemented for every file format
            except:
                pass
            self.display_values_in_gui()
            self.show_sum_spec()
            self.roi_widget.file_type = self.data.file_type
            self.popup_properties.backup_text = str(self.popup_properties.measurement_properties.toPlainText())
            self.statusBar().showMessage("Loading done.")
        except Exception as e:
            print("data folder could not be loaded")
            print(f"error code {e}")
            traceback.print_exc()
            print( "__________________________________________________________")
            self.reset_2_default()
        gc.collect()

    def load_file(self, angle_file=False):
        try:
            self.reset_2_default()
            self.data.elements = self.elements
            self.data.label_loading_progress = self.statusBar()
            self.data.open_data_file(angle_file)
            self.threshold_handler = FitThresholdPopup(self.data.folder_path)
            self.load_parameter_in_specfit_deconvolution()
            self.statusBar().showMessage("loading done")
            self.popup_properties.data_path = self.data.save_folder_path
            self.popup_properties.fill_text(f"file path:\n {self.data.file_path}\n")
            self.display_values_in_gui()
            if angle_file:
                self.check_use_parameters.setChecked(False)
            self.show_sum_spec()
            self.roi_widget.file_type = self.data.file_type
            self.statusBar().showMessage("Loading done.")
            self.popup_properties.backup_text = str(self.popup_properties.measurement_properties.toPlainText())
        except Exception as e:
            print("data file could not be loaded")
            print(f"error code {e}")
            traceback.print_exc()
            print("__________________________________________________________")
            self.reset_2_default()
        gc.collect()

    def npy_2_bin(self):
        self.data.npy_2_bin()

    def split_h5(self, ):
        """
        Function to split the selected data.h5 file in order to fit the loaded
        data in batches
        """
        split_h5()
        self.statusBar().showMessage("### splitting done ###")

    def h5_2_tiff(self, ):
        """
        Function create tiff images from results.h5 file
        """
        h5_to_tiff()
        self.statusBar().showMessage("### tiff image creation done ###")

    def show_plot3d(self):
        self.plot3d.show_plot3d()

    def show_display_meas_points(self):
        self.dmp = dmp()
        self.dmp.show_display_meas_points()

    def show_specfit_fit_settings(self):
        self.sfs.show_sfs(_parent=self)

    def ipython_console(self, ):
        # Create an IPython console widget
        self.ipython_widget = IPythonConsole(self)
        # Pass the MyGUI instance to the IPython console's user namespace
        self.ipython_widget.ipython_widget.kernel_manager.kernel.shell.user_ns["specfit"] = self
        self.ipython_widget.show()

    def reset_2_default(self):
        """
        This function sets all the used variables to the default value.
        """
        self.clear_ax_canvas_spectrum()
        # remove all text instances in plot
        for text in self.ax_canvas_spectrum.texts:
            text.remove()
        self.pse_widget.clear_line_list()
        try:
            self.popup_properties.clear_popup()
        except:
            pass
        self.data.reset_2_default()
        self.data.label_loading_progress = self.statusBar()
        self.data.file_dialog = QtWidgets.QFileDialog(self)
        self.pse_widget.tab_udl.reset_2_default()
        self.s.fit_in_progress = False
        self.ax_canvas_spectrum.set_xlabel("Energy / keV")
        self.ax_canvas_spectrum.set_ylabel("Intensity / cps")
        self.batch_fitting = False
        self.sfs.check_batch_fitting.setChecked(False)
        self.data_batch_fitting = False
        self.spectra_nr = None
        self.canvas_spectrum.draw()

    def open_param_file(self):
        """
        Function to open a display to select a file and set it's parameters
        """
        self.pse_widget.tab_udl.reset_2_default()
        param_path = QtWidgets.QFileDialog(self).getOpenFileName(filter="(*.txt *.dat)")[0]
        self.data.check_param_file(param_path)
        self.display_values_in_gui()
        self.check_fit()

    def save_param_file(self):
        """
        Function to open a dalog to determine a save location and saves settings and parameters
        """
        filename = QtWidgets.QFileDialog(self).getSaveFileName()[0]
        self.store_gui_values()
        outstring =  self.data.save_param_file(filename)
        self.statusBar().showMessage(outstring)

    def store_gui_values(self):
        """
        Function to update DataHandler with GUI_values
        """
        self.data.parameters_user[:4] = np.asarray([self.entry_a0.text(), self.entry_a1.text(), self.entry_fano.text(), self.entry_FWHM.text()], dtype=float) #stores parameters defined by User in GUI, 
        self.data.update_channels(use_user_parameter=self.check_use_parameters.checkState().value)
        self.data.use_parameters = (self.check_use_parameters.checkState() == QtCore.Qt.CheckState.Checked)
        self.data.bg_zero = (self.check_set_bg_zero.checkState() == QtCore.Qt.CheckState.Checked)
        self.data.strip_cycles = int(self.entry_strip_cycles.text())
        if np.asarray(self.entry_strip_width.text(), dtype=int)<2:
            self.entry_strip_width.setText("2")
            self.data.strip_width = 2
        else:
            self.data.strip_width = int(self.entry_strip_width.text())
        self.data.smooth_cycles = int(self.entry_smooth_cycles.text())
        if int(self.entry_smooth_width.text()) < 2:
            self.entry_smooth_width.setText("2")
            self.data.smooth_width = 2
        else:
            self.data.smooth_width = int(self.entry_smooth_width.text())
        self.data.roi_start = float(self.entry_roi_start.text())
        self.data.roi_end = float(self.entry_roi_end.text())
        if self.radio_semilogy.isChecked():
            self.data.plot_style = "log"
        elif self.radio_linear.isChecked():
            self.data.plot_style = "lin"
        elif self.radio_none.isChecked():
            self.data.plot_style = "None"
        self.data.use_lib = self.combo_lib.currentText()
        self.data.nl_fit = (self.check_nl_fit.checkState().value==2)
        self.data.calc_minima = (self.check_calc_minima.checkState().value==2)
        self.data.calc_minima_order = int(self.entry_calc_minima_order.text())
        self.data.threshold = np.asarray(self.entry_nl_fit_threshold.text(), dtype=float)
        self.data.mincount = np.asarray(self.threshold_handler.entry_mincounts.text(), dtype=float)
        self.pse_widget.get_selected_lines()
        self.pse_widget.user_defined_lines = self.pse_widget.tab_udl.get_user_defined_lines_data()
        self.pse_widget.build_specfit_addlines()
        self.data.selected_elements = self.pse_widget.selected_elements
        self.data.selected_lines = self.pse_widget.selected_lines

    def display_values_in_gui(self):
        """
        Function to display the values from the data class in GUI
        """
        self.entry_a0.setText(str(self.data.parameters_user[0]))
        self.entry_a1.setText(str(self.data.parameters_user[1]))
        self.entry_fano.setText(str(self.data.parameters_user[2]))
        self.entry_FWHM.setText(str(self.data.parameters_user[3]))
        self.check_use_parameters.setChecked(self.data.use_parameters)
        self.check_set_bg_zero.setChecked(self.data.bg_zero)
        self.entry_strip_cycles.setText(f"{self.data.strip_cycles}")
        self.entry_strip_width.setText(f"{self.data.strip_width}")
        self.entry_smooth_cycles.setText(f"{self.data.smooth_cycles}")
        self.entry_smooth_width.setText(f"{self.data.smooth_width}")
        self.entry_roi_start.setText(f"{self.data.roi_start}")
        self.entry_roi_end.setText(f"{self.data.roi_end}")
        if self.data.use_lib == "xraylib": self.combo_lib.setCurrentIndex(0)
        else:self.combo_lib.setCurrentIndex(1)
        self.check_nl_fit.setChecked(self.data.nl_fit)
        self.entry_nl_fit_threshold.setText(f"{self.data.threshold}")
        self.check_calc_minima.setChecked(self.data.calc_minima)
        self.entry_calc_minima_order.setText(f"{self.data.calc_minima_order}")
        self.threshold_handler.entry_mincounts.setText(f"{self.data.mincount}")
        self.pse_widget.selected_elements = self.data.selected_elements
        self.pse_widget.selected_lines = self.data.selected_lines
        self.pse_widget.clear_line_list()
        self.pse_widget.tab_udl.reset_2_default()
        self.pse_widget.display_selected_lines()
        if self.data.plot_style == "log":
            self.radio_semilogy.setChecked(True)
            self.plot_style = self.ax_canvas_spectrum.semilogy
        else:
            self.radio_linear.setChecked(True)
            self.plot_style = self.ax_canvas_spectrum.plot

    def show_sum_spec(self): # TODO this is in principle the same as plot_canvas
        """
        Function to display a sum spectrum of the loaded spectra.
        """
        # newly initialize the roi_widget
        self.roi_widget.spec_nr = -1
        # clear the spectrum display in the main GUI
        for text in self.ax_canvas_spectrum.texts:
            text.remove()
        # determine the indices of the energies in the given ROI range
        low_index, high_index = self.data.get_roi_indicees()
        # store the sum_spec in the self.selected_spectrum container
        self.selected_spectrum = self.data.sum_spec
        # plot the sum spec of the loaded measurement
        self.plot_style(self.data.energies[low_index:high_index], self.data.sum_spec[low_index:high_index], 
                        color="#48a0dcff", label="measurement")
        # add labels and legend to plot
        self.ax_canvas_spectrum.set_xlim(self.data.roi_start, self.data.roi_end)
        self.ax_canvas_spectrum.set_xlabel("Energy / keV")
        self.ax_canvas_spectrum.set_ylabel("Intensity / cps")
        self.ax_canvas_spectrum.legend(loc="best")
        # actually show plot
        self.canvas_spectrum.draw()
        # fill properties display with measurement information
        self.popup_properties.fill_text(f"channels: {self.data.len_spectrum}\n")
        self.popup_properties.fill_text(f"energy : [{self.data.energies[0]:.2f}, {self.data.energies[-1]:.2f}] keV\n")
        if np.ndim(self.data.parameters) == 1:
            self.popup_properties.fill_text(f"life time : {np.mean(self.data.parameters[4])} s\n")
            try: self.popup_properties.fill_text(f"real time : {np.mean(self.data.parameters[7])} s\n")
            except: pass
        else:
            self.popup_properties.fill_text(f"life time : {np.mean(self.data.parameters[:, 4])} s\n")
            try: self.popup_properties.fill_text(f"real time : {np.mean(self.data.parameters[:, 7])} s \n")
            except: pass
        self.popup_properties.fill_text(f"position steps: {self.data.position_dimension[0]}-{self.data.position_dimension[1]}-{self.data.position_dimension[2]}\n")
        try:
            self.popup_properties.fill_text('position steps: {:d}-{:d}-{:d} \n'.format(*self.data.position_dimension))
        except:
            pass  # for MCA no position_dimension is generated at the moment?        
        # set main GUI on top
        self.activateWindow()

    def fill_and_show_lines_widget(self):
        """
        Function to load current spec and params in lines widget and display it.
        """
        self.check_fit() #wenn man in der GUI Daten geändert hat soll er sie auch direkt anzeigen und speichern
        self.pse_widget.tab_udl.load_spec((np.subtract(self.s.meas_load, self.s.Strip)), float(self.entry_a0.text()), float(self.entry_a1.text()), "test")
        self.pse_widget.tab_udl.display_lines_widget()

    def fill_and_show_fit_threshold_widget(self):
        self.store_gui_values()
        self.threshold_handler.show_frp()
        self.threshold_handler.roi_indicees = self.data.get_roi_indicees()
        if not self.data.loadtype == "folder":
            self.threshold_handler.spec = self.data.spectra
            self.threshold_handler.use_spectra = True
        self.threshold_handler.loadtype = self.data.loadtype
        self.threshold_handler.plot_counts()

    def fill_and_display_show_roi(self):
         """
         Function to load current params in ROI widget and display it.
         """
         self.roi_widget.load_type = self.data.loadtype
         self.roi_widget.parameters = self.data.parameters
         self.roi_widget.plot_style_str = self.data.plot_style
         one_d_array = (self.data.position_dimension[0]== 1 and self.data.position_dimension[1]==1)
         self.roi_widget.load_spectra(folder_path=self.data.folder_path, 
                                      save_folder_path=self.data.save_folder_path, 
                                      save_data_path=self.data.save_data_folder_path, 
                                      load_type=self.data.loadtype, 
                                      one_dim=one_d_array)
         self.roi_widget.display_show_ROI()

    def change_plot_style(self, style):
        self.data.plot_style = style
        if style != self.data.plot_style:
            self.clear_ax_canvas_spectrum()
            self.plot_style_changed = True
            for image in self.ax_canvas_spectrum.images:
                image.remove()
            if style == "lin":
                self.plot_style = self.ax_canvas_spectrum.plot
                self.ax_canvas_spectrum.set_yscale("linear")
                if not self.s.fit_in_progress:
                    self.check_fit()
                elif self.s.fit_in_progress:
                    self.plot_canvas(self.s.meas_load, self.data.energies, self.s.Strip, self.s.calc_spec(), 
                                     initialize=False)
            elif style == "log":
                self.plot_style = self.ax_canvas_spectrum.semilogy
                self.ax_canvas_spectrum.set_yscale("log")
                if not self.s.fit_in_progress:
                    self.check_fit()
                elif self.s.fit_in_progress:
                    self.plot_canvas(self.s.meas_load, self.data.energies, self.s.Strip, self.s.calc_spec(), 
                                     initialize=False)
            elif style == "None":
                self.plot_style = None
                self.ax_canvas_spectrum.set_yscale("linear")
                self.ascii_imshow = self.ax_canvas_spectrum.imshow(self.ascii_specfit_logo, aspect="auto")
                self.ax_canvas_spectrum.set_xlim(0, self.ascii_imshow.properties()["size"][1])
                self.ax_canvas_spectrum.set_ylim(self.ascii_imshow.properties()["size"][0], 1e-5)
        self.canvas_spectrum.draw()

    def load_parameter_in_specfit_deconvolution(self, spectra_nr=0):
        """
        This function loads the parameters a0, a1, Fano, el.noise, E_max life_time, 
        gating_time, strip_cycles, strip_width, smooth_cycles, smooth_width, 
        minchange (nl_fit threshold) and database (xraylib) from the
        DataHandler into specfit_deconvolution
        """
        assert self.data.loadtype in ["file", "msa_file", "angle_file", "folder", 
                                      "hdf5_file", "bcf_file", "csv_file"]
        parameters = self.get_selected_parameters()
        try:
            self.s.Det = parameters[:5]
            self.s.life_time = parameters[4]
            try: self.s.real_time = parameters[7]  # TODO real time has to be implemented for every file format
            except: pass
            self.s.gating_time = parameters[6]
        except:
            self.s.Det = parameters[spectra_nr, :5]
            self.s.life_time = parameters[spectra_nr, 4]
            try: self.s.real_time = parameters[spectra_nr, 7]  # TODO real time has to be implemented for every file format
            except: pass
            self.s.gating_time = parameters[spectra_nr, 6]
        self.s.strip_cycles = self.data.strip_cycles
        self.s.strip_width = self.data.strip_width
        self.s.smooth_cycles = self.data.smooth_cycles
        self.s.smooth_width = self.data.smooth_width
        self.s.minchange = self.data.threshold
        self.s.xraylib = self.data.use_lib == "xraylib"

    def get_selected_parameters(self):
        """
        Helper function to get the parameters depending on the loaded measurement
        type and if the GUI parameters are used (e.g. after non-linear calibration)
        """

        if self.data.use_parameters is False and self.data.loadtype in ["file", "msa_file", "angle_file", 
                                                                        "hdf5_file", "bcf_file", "csv_file"]:
            parameters = self.data.parameters
        elif self.data.use_parameters is False and self.data.loadtype in ["folder"]:
            if self.data.file_type == ".bcf":
                parameters = self.data.parameters
            else:
                parameters = self.data.parameters[self.roi_widget.spec_nr]
        elif self.data.use_parameters is True:
            parameters = self.data.parameters_user
        return parameters

    def load_spec_in_specfit_deconvolution(self, spectra_nr=None):
        """
        loads a spectrum in specfit_deconvolution depending on DataHandler.load_type
        and determines the ROI positions, the corresponding energies in
        specfit_deconvolution.Bins and corresponding intensities in
        specfit_deconvolution.Meas.
        negative spectra_nr means DataHandler.sum_spec
        IMPORTANT first load data, than the spectrum spec
        """
        if spectra_nr is None:
            spectra_nr = self.roi_widget.spec_nr
        if spectra_nr >= 0:
            self.s.load_spec(self.data.spectra[spectra_nr])
        elif spectra_nr == -1:
            self.s.load_spec(self.data.sum_spec)
        elif spectra_nr == -2:
            self.s.load_spec(self.data.max_pixel_spec)
        elif spectra_nr < -2:
            self.s.load_spec(self.data.sum_spec)
        else:
            assert self.data.loadtype in ["angle_file", "file", "folder", "msa_file", 
                                          "hdf5_file", "bcf_file", "csv_file"]
            if self.data.loadtype == "angle_file":
                angle = self.data.tensor_positions.T[2][spectra_nr]
                try: self.s.load_spec(self.data.spectra[angle])
                except:   self.statusBar().showMessage("Error - spec could not be loaded")
            elif self.data.loadtype == "file":
                if type(self.data.spectra) is dict:
                    self.s.load_spec(self.data.spectra["0"])
                elif type(self.data.spectra) is np.ndarray:
                    self.s.load_spec(self.data.spectra)
            elif self.data.loadtype in ["folder", "msa_file", "hdf5_file", "bcf_file", 
                                        "csv_file"]:
                if self.s.fit_in_progress:
                    self.s.load_spec(self.data.spectra[spectra_nr]) # dict not empty
                elif not self.s.fit_in_progress:
                    try:
                        self.s.load_spec(self.roi_widget.rect_sum_spec)
                    except:
                        self.s.load_spec(self.data.spectra[spectra_nr])
        self.s.set_ROI(self.data.get_roi_indicees())

    def export_2_npz(self, ):
        """
        This function exports the loaded spectra to a compressed numpy
        array with the same shape as the measurement. The compression used
        is bz2
        """
        try: self.data.loadtype
        except:
            self.statusBar().showMessage("Please load a measurment prior export.")
            return
        export_2_npz_path = QtWidgets.QFileDialog().getSaveFileName(self, 
                                                                  "select save path", 
                                                                  self.data.file_path.replace(self.data.file_type, ".npz"))[0]
        if ".npz" not in export_2_npz_path:
            export_2_npz_path += ".npz"
        # now read out data
        data = sfunc.open_dict_pickle(f"{self.data.save_data_folder_path}/spectra.pickle")
        # check the dimension of the loaded data
        dimension = list(self.data.position_dimension)
        dimension.append(self.data.len_spectrum)
        # reshape the data to the determined dimension
        spectra = np.asarray(list(data.values())).reshape(dimension)
        # compress data
        self.statusBar().showMessage("Data is being compressed.")
        np.savez_compressed(export_2_npz_path, data=spectra)
        self.statusBar().showMessage("export done")

    def clear_ax_canvas_spectrum(self, labels=list()):
        """
        This function clears all plot-lines from the canvas_spectrum except the
        lines with given labels
        """
        if labels is False:
            labels = ["measurement"]
        if labels == []:
            # clear everything from the plot exept the axes labels
            for text in list(self.ax_canvas_spectrum.texts):
                text.remove()
            for line in list(self.ax_canvas_spectrum.get_lines()):
                line.remove()
            if self.ax_canvas_spectrum.get_legend() is not None:
                self.ax_canvas_spectrum.get_legend().remove()
        else:
            for line in list(self.ax_canvas_spectrum.get_lines()):
                if line.properties()["label"] not in labels:
                    line.remove()
        for text in self.ax_canvas_spectrum.texts:
            text.remove()
        if self.ax_canvas_spectrum.get_legend() is not None:
            self.ax_canvas_spectrum.get_legend().remove()
        self.selected_energy = None
        self.canvas_spectrum.draw()

    def check_fit(self, params_user=True):
        """
        This function fits the selected spectrum with the specfit1.py
        1. stores all Values and Lines from GUI
        2. load parameters and data in specfit
        3. determine background
        4. calculate with specfit
        """
        # check if background or fitted spectrum are already plotted or
        # should be removed prior fitting in order to keep the plot canvas
        # tidy
        if self.pse_widget.selected_elements == []:
            self.plot_style_changed = True
        elif self.s.Strip is None:
            self.plot_style_changed = True
        line_labels = []
        for line in self.ax_canvas_spectrum.lines:
            line_labels.append(line.properties()["label"])
        if "fitted spectrum" not in line_labels:
            self.plot_style_changed = True
        # prior every check_fit remove the texts and vertical lines from
        # the plot canvas
        self.clear_ax_canvas_spectrum(["measurement", "background", "fitted spectrum"])
        # update the DataHandler with GUI values
        self.store_gui_values()
        # load all needed parameters into specfit_deconvolution
        self.load_parameter_in_specfit_deconvolution()
        # now load the spectrum in specfit_deconvolution and crop to ROI
        if self.spectra_nr is None:
            self.spectra_nr = self.roi_widget.spec_nr
        self.load_spec_in_specfit_deconvolution(spectra_nr=self.spectra_nr)
        # set order of Minimum detection
        self.s.calc_minima_order = int(self.entry_calc_minima_order.text())
        # set PU and Escape parameters
        self.sfs.set_PU_Escape_parameters()
        # if background is approximated to zero
        if self.data.bg_zero is True:
            self.s.Strip = np.zeros(len(self.s.meas_load))
        else:
            # otherwise approximate the background by smoothing and stripping
            if self.check_calc_minima.isChecked():
                self.data.calc_minima = True
                self.data.calc_minima_order = int(self.entry_calc_minima_order.text())
                self.s.strip(only_minima_on_sum_spec=False)
            else:
                self.data.calc_minima = False
                self.data.calc_minima_order = 1
                self.s.strip(calc_minima=False, only_minima_on_sum_spec=False)
        # load user defined lines from periodic_table
        self.pse_widget.tab_udl.load_spec((np.subtract(self.s.meas_load, self.s.Strip)), float(self.data.parameters_user[0]), float(self.data.parameters_user[1]), "test")
        label_list, self.s.user_defined_lines = self.pse_widget.tab_udl.get_user_defined_lines()
        # read out all lines and check_fit the spectrum
        addlines = self.pse_widget.specfit_addlines
        if addlines or self.s.user_defined_lines: #(if not empty)
            self.s.addLines(addlines)
            if self.check_nl_fit.checkState() == QtCore.Qt.CheckState.Unchecked:
                # calculate the fitted peaks using linear fit (entered parameter)
                self.s.linfit()
            elif self.check_nl_fit.checkState() == QtCore.Qt.CheckState.Checked:
                self.s.fit()
                self.data.parameters_user[0:4] = self.s.Det[0:4]
                self.display_values_in_gui()
                self.check_use_parameters.setChecked(True)
                self.check_nl_fit.setChecked(False)
            result_spec = self.s.calc_spec()
        else:
            self.s.Lines.clear()
            result_spec = []
        # display fit results on measurement properties
        self.popup_properties.clear_popup()
        self.popup_properties.fill_text(self.popup_properties.backup_text)
        for _s in range(len(self.s.Lines)):
            if _s == 0:
               self.popup_properties.fill_text("Results on Sum Spec\n+++++++++++++++++++\n")
            self.popup_properties.fill_text("{}-{} : {:.0f}cps\n".format(xrl.AtomicNumberToSymbol(self.s.Lines[_s]["Z"]), 
                                                                         self.s.Lines[_s]["edge"], 
                                                                         self.s.Lines[_s]["I"]))
        # plot the results
        self.plot_canvas(self.s.meas_load, self.data.energies, self.s.Strip, result_spec)

    def plot_canvas(self, spectrum, energy, background, calc_spec, 
                    initialize=True):
        """
        This function plots the given spectra in the canvas_spectrum
        
        Parameters
        ----------
        spectrum : numpy.ndarray
            intensity spectrum
        energy : numpy.ndarray
            energy scale
        background : numpy.ndarray
            approximated background
        calc_spec : numpy.ndarray
            background plus fluorescence peaks
        """
        if isinstance(spectrum, da.Array):
            spectrum = spectrum.compute()
        if self.check_auto_scale.checkState() == QtCore.Qt.CheckState.Checked:
            initialize = True
        if self.plot_style is not None:
            plot_start, plot_end = self.data.get_roi_indicees()
            if initialize:
                self.ax_canvas_spectrum.set_xlim(energy[plot_start], energy[plot_end])
                if self.plot_style == "lin":
                    self.ax_canvas_spectrum.set_ylim(0, np.max(spectrum[plot_start:plot_end]) * 1.1)
                else:
                    self.ax_canvas_spectrum.set_ylim(np.min(spectrum[plot_start:plot_end]) * 0.9, 
                                                     np.max(spectrum[plot_start:plot_end]) * 1.1)
            else:
                xmin, xmax = self.ax_canvas_spectrum.get_xlim()
                ymin, ymax = self.ax_canvas_spectrum.get_ylim()
                self.ax_canvas_spectrum.set_xlim(xmin, xmax)
                self.ax_canvas_spectrum.set_ylim(ymin, ymax)
            if background is not None:
                background = np.array(background)
                calc_spec = np.array(calc_spec)
            if self.plot_style_changed:
                self.clear_ax_canvas_spectrum()
                if self.selected_energy is not None:
                    self.selected_energy = self.ax_canvas_spectrum.axvline(self.selected_energy.get_xdata()[0], 
                                                                           ymax=self.ax_canvas_spectrum.get_ylim()[
                                                                                    -1] / 2, 
                                                                           color="g")
                self.plot_style(energy[plot_start:plot_end], 
                                spectrum[plot_start:plot_end], 
                                color="#48a0dcff", 
                                label="measurement")                       # plot the given-spec
                if background.size > 1: self.plot_style(energy[plot_start:plot_end], 
                                                        background[plot_start:plot_end], 
                                                        color="orange", 
                                                        label="background")                     # plot the background
                if calc_spec.size >1: self.plot_style(energy[plot_start:plot_end], 
                                                      calc_spec[plot_start:plot_end], 
                                                      color="red", 
                                                      label="fitted spectrum")
                self.plot_style_changed = False
            else:
                for line in self.ax_canvas_spectrum.lines:
                    if line.properties()["label"] == "measurement":
                        line.set_xdata(energy[plot_start:plot_end])
                        line.set_ydata(spectrum[plot_start:plot_end])
                    elif line.properties()["label"] == "background":
                        if background is not None:
                            line.set_xdata(energy[plot_start:plot_end])
                            line.set_ydata(background[plot_start:plot_end])
                    elif line.properties()["label"] == "fitted spectrum":
                        if calc_spec is not None:
                            line.set_xdata(energy[plot_start:plot_end])
                            line.set_ydata(calc_spec[plot_start:plot_end])
            self.ax_canvas_spectrum.legend()
            self.canvas_spectrum.draw()

    def clear_plot(self):
        """
        This function redraws the measurement and the calculated underground.
        """
        self.clear_ax_canvas_spectrum()
        self.s.Strip = None
        self.roi_widget.spec_nr = -1  # reset to sum spec
        self.set_spectra_nr(spectra_nr=-1)
        if self.plot_style is not None:
            plot_start, plot_end = self.data.get_roi_indicees()
            self.plot_style(self.data.energies[plot_start:plot_end], self.data.sum_spec[plot_start:plot_end], 
                            color="#48a0dcff", 
                            label="measurement")
            try: self.plot_style(self.data.energies[plot_start:plot_end], self.s.Strip[plot_start:plot_end], 
                                 color="orange", 
                                 label="background")
            except: pass
            self.ax_canvas_spectrum.set_ylim(np.min(self.data.sum_spec)*0.9, np.max(self.data.sum_spec)*1.1)
            self.ax_canvas_spectrum.legend(loc="best")
            self.canvas_spectrum.draw()

    def fit_folder(self):
        """
        This function fits the recorded spectrum of the given folder the specfit1.py
        program in he specific order:
        1st it adds the choosed lines
        3rd it fits the lines with specfit1.py and plots it in the same figure
        4th if "clear plot" is pressed it will clear the plot and redraw
        the measurement and the underground.
        """
        self.popup_properties.clear_popup()
        self.popup_properties.fill_text(self.popup_properties.backup_text)
        if self.data.loadtype in ["file"]:
            self.save_file_path = QtWidgets.QFileDialog().getSaveFileName(self, 
                                                                          "select save path", 
                                                                          self.data.file_path.replace(self.data.file_type, "_results.dat"))[0]
            self.save_folder_path = "/".join(self.save_file_path.split("/")[:-1])
        elif self.data.loadtype in ["folder", "msa_file", "hdf5_file", "bcf_file", "angle_file", 
                                    "csv_file"]:
            self.save_folder_path = QtWidgets.QFileDialog().getExistingDirectory(self, 
                                                                                 "select save folder", 
                                                                                 self.data.folder_path)

        def create_empty_results(load_type="folder", 
                                 selected_elements=["Fe"], 
                                 selected_lines=[["K"]], 
                                 batch_fitting=True, 
                                 dimension=[1, 1, 1], 
                                 verbose=False):
            """
            This function builds a dict with every chosen
            element-line-combination and returns an np.zeros with len of
            positions.

            Parameters
            ----------
            load_type : str, optional
                one of the following
                ['file', 'angle_file', 'folder', 'msa_file', 'hdf5_file', 'bcf_file'].
                The default is 'folder'.
            selected_elements : list, optional
                list of ZString of selected elements to deconvolve.
                The default is ['Fe'].
            selected_lines : TYPE, optional
                concatenated list of lines corresponding to the elements which
                have to be deconvolved. The default is [['K']].
            batch_fitting : Bool, optional
                if batch_fitting True the value of the first dimension is set to 1
                The default is True.
            dimension : list, optional
                list of the number of measurements per dimension.
                The default is [1, 1, 1].
            verbose : bool, optional
                wether to print or not, the default is False
            Returns
            -------
            results : dictionary
                In this dictionary every key [{element}_{line}_scan_{nr}] gets
                a np.zeros in the shape of the measurement dimension. While
                fitting this array is filled
            """
            assert load_type in ["file", "angle_file", "folder", "msa_file", "hdf5_file", "bcf_file", 
                                 "csv_file"]
            if batch_fitting:
                dimension[0] = 1
            results = {}
            if load_type in ["file", "angle_file"]:
                for element, linelist in zip(selected_elements, selected_lines):
                    for line in linelist:
                        results[f"{element}_{line}"] = np.zeros(dimension[2])
                for label in self.pse_widget.tab_udl.get_label_list():
                   results[f"{label}"] = np.zeros(dimension[2])
            elif load_type in ["folder", "msa_file", "hdf5_file", "bcf_file", "csv_file"]:
                for element, linelist in zip(selected_elements, selected_lines):
                    for line in linelist:
                        if not isinstance(dimension[0], np.ndarray):
                            results[f"{element}_{line}"] = np.zeros(dimension)
                        else:
                            for scan in range(len(dimension)):
                                results[f"{element}_{line}_scan_{scan}"] = np.zeros(dimension[scan])

                for label in self.pse_widget.tab_udl.get_label_list():
                    if not isinstance(dimension[0], np.ndarray):
                        results[f"{label}"] = np.zeros(dimension)
                    else:
                        for scan in range(len(dimension)):
                            results[f"{label}_scan_{scan}"] = np.zeros(dimension[scan])
            return results

        def save_results(results, 
                         save_path="results", 
                         batch_fitting=True, 
                         batch_iterator=0, 
                         file_type=".spx", 
                         dimension=[1, 1, 1], 
                         save_storage=".h5", 
                         verbose=False, 
                         ):
            if batch_fitting:
                dimension[0] = 1
            for key in results.keys():
                if batch_fitting:
                    results_key = f"{self.data.file_name}/{key}_{batch_iterator}"
                else:
                    results_key = f"{self.data.file_name}/{key}"
                if save_storage == ".h5":
                    with h5py.File(f"{save_path}/results.h5", "a") as tofile:
                        # read out stored results of h5 file as list
                        content = []
                        for meas in tofile.keys():
                            for res in tofile[meas].keys():
                                content.append(f"{meas}/{res}")
                        if file_type == ".MSA":
                            results[key] = results[key].reshape(np.flip(dimension))
                        if results_key in content:
                            tofile[results_key][()] = results[key]  # replace the results already in results.h5
                        else:
                            tofile.create_dataset(results_key, 
                                                  data=results[key])
                elif save_storage == ".npy":
                    if file_type == ".MSA":
                        results[key] = results[key].reshape(np.flip(dimension))
                    np.save(f"{save_path}/{results_key}.npy", results[key])

        def show_fit_prop_in_popup():
            """
            Function to write fit-lines to properties-popup
            """
            self.popup_properties.fill_text("fitted elements:\n")
            for element, lines in zip(self.pse_widget.selected_elements, self.pse_widget.selected_lines):
                self.popup_properties.fill_text(f"{element}: {lines}\n")
            self.popup_properties.fill_text(f"strip cycles: {self.data.strip_cycles}\n")
            self.popup_properties.fill_text(f"strip width: {self.data.strip_width}\n")
            self.popup_properties.fill_text(f"life time: {self.s.life_time:f}\n")
            self.popup_properties.fill_text(f"real time: {self.s.real_time:f}\n")
            self.popup_properties.fill_text(f"ROI: [{self.data.roi_start:.2f} - {self.data.roi_end:.2f}] keV\n")
            self.popup_properties.fill_text(f"used lib: {self.data.use_lib}\n")
            if self.data.use_parameters is True:
                self.popup_properties.fill_text(f"a0: {self.data.parameters_user[0]:f}\n")
                self.popup_properties.fill_text(f"a1: {self.data.parameters_user[1]:f}\n")
                self.popup_properties.fill_text(f"Fano: {self.data.parameters_user[2]:f}\n")
                self.popup_properties.fill_text(f"FWHM: {self.data.parameters_user[3]:f}\n")
            else:
                self.popup_properties.fill_text("Used parameters stored in Data files")
            self.popup_properties.show()

        self.roi_widget.spec_nr = -1
        roi_min, roi_max = self.data.get_roi_indicees()
        self.s.fit_in_progress = False
        self.store_gui_values()
        self.load_parameter_in_specfit_deconvolution()
        self.data.create_save_folder()
        # set PU and Escape parameters
        self.sfs.set_PU_Escape_parameters()
        self.statusBar().setStyleSheet("color: black")
        self.popup_properties.fill_text(f"Threshold - {self.data.mincount} cps\n")
        compute_time_start = time.time()
        time2go = time.gmtime(time.time()-time.time())
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximumSize(100, 18)
        self.fit_widget_layout.addWidget(self.progress_bar, 2, 1)
        self.progress_bar.show()
        # fit every spectrum
        if self.data.loadtype == "file":  # means only one spectrum
            self.check_nl_fit.setChecked(False)
            show_fit_prop_in_popup()   # write fit-lines to properties-popup
            results = create_empty_results(load_type=self.data.loadtype, 
                                           selected_elements=self.pse_widget.selected_elements, 
                                           selected_lines=self.pse_widget.selected_lines, 
                                           batch_fitting=False, 
                                           dimension=np.copy(self.data.position_dimension)
                                                )
            self.check_fit(params_user=self.data.use_parameters)
            self.s.udl_label_list, self.s.user_defined_lines = self.pse_widget.tab_udl.get_user_defined_lines()
            if self.check_calc_minima.isChecked():
                self.s.strip()
            else:
                self.s.strip(calc_minima=False)
            self.s.linfit()
            # get the results from specfit deconvolution
            getResults = self.s.get_result()
            # save the fitted intensities to a txt file
            with open(self.save_file_path, "a") as infile:
                for key in getResults.keys():
                    try:
                        infile.write("{}I: {}, Edge: {}, Z: {}{}\n".format("{", getResults[key]["I"], getResults[key]["Edge"], getResults[key]["Z"], "}"))
                    except:
                        infile.write("{}I: {}, custom line: {}{}\n".format("{", getResults[key]["I"], key, "}"))
            if self.sfs.check_save_background.isChecked():
                np.savetxt(f"{self.save_folder_path}/background.txt", 
                           np.column_stack([self.data.energies, self.s.Strip]), 
                           delimiter="\t")
            if self.sfs.check_save_fitted_spectrum.isChecked():
                np.savetxt(f"{self.save_folder_path}/fitted_spectrum.txt", 
                           np.column_stack([self.data.energies, self.s.calc_spec()]), 
                           delimiter="\t")
        elif self.data.loadtype == "angle_file":
            time_file_start = time.time()
            results = create_empty_results(load_type=self.data.loadtype, 
                                           selected_elements=self.pse_widget.selected_elements, 
                                           selected_lines=self.pse_widget.selected_lines, 
                                           batch_fitting=False, 
                                           dimension=np.copy(self.data.position_dimension))
            self.check_use_parameters.setChecked(True)
            show_fit_prop_in_popup()
            partitions = len(self.data.spectra)
            # if the background should be save, an empty container will be created here
            if self.sfs.check_save_background.isChecked():
                background = np.zeros((partitions, len(self.data.energies)))
            if self.sfs.check_save_fitted_spectrum.isChecked():
                fitted_spectra = np.zeros((partitions, len(self.data.energies)))
            for angle in range(partitions):
                if np.sum(self.data.spectra[angle][roi_min:roi_max])> self.data.mincount:
                    self.roi_widget.spec_nr = angle
                    self.load_spec_in_specfit_deconvolution()
                    if self.data.bg_zero == True:
                        self.s.Strip = np.zeros(len(self.s.meas_load))
                    else :
                        if self.check_calc_minima.isChecked():
                            self.s.strip()
                        else:
                            self.s.strip(calc_minima=False)
                    self.pse_widget.tab_udl.load_spec((np.subtract(self.s.meas_load, self.s.Strip)), 
                                                      self.data.parameters_user[0], 
                                                      self.data.parameters_user[1], 
                                                      str(angle))
                    self.s.udl_label_list, self.s.user_defined_lines = self.pse_widget.tab_udl.get_user_defined_lines()
                    self.s.linfit()
                    self.plot_canvas(spectrum=self.s.meas_load, 
                                     energy=self.data.energies, 
                                     background=self.s.Strip, 
                                     calc_spec=self.s.calc_spec(), 
                                     initialize=False)

                    sp_results = self.s.get_result()
                    for result_key in sp_results.keys():  # result_key for example: CR_K
                        results[result_key][angle] = sp_results[result_key]["I"]
                else:
                    result_keys = self.s.get_result_keys()
                    for rk in result_keys:  # result_key for example: CR_K
                        results[rk][angle] = 0
                # if the background should be saved, store the estimated
                # background s.Strip to the background array
                if self.sfs.check_save_background.isChecked():
                    background[angle] = self.s.Strip
                if self.sfs.check_save_fitted_spectrum.isChecked():
                    fitted_spectra[angle] = self.s.calc_spec()
                if angle % 1 == 0:
                    time_file_end = time.time()
                    time2go = time.gmtime((time_file_end-time_file_start)*(len(self.data.spectra)-(angle+1))/(angle+1))
                    self.progress_bar.setValue(int((angle+1)/float(len(self.data.spectra))*100.0))
                    self.statusBar().showMessage("progress : {:.0f}/{:.0f} - {:.2f} %% - time to go: {}".format(angle+1, 
                                                                                                len(self.data.spectra), 
                                                                                                (angle+1)/float(len(self.data.spectra))*100.0, 
                                                                                                time.strftime("%H:%M:%S", 
                                                                                                              time2go)))
                    QtWidgets.QApplication.processEvents()
                self.s.fit_in_progress = True
            for key in results.keys():
                print("here is the key as follows: ", key)
                with h5py.File(f"{self.save_folder_path}/results.h5", "a") as tofile:
                    if key in list(tofile.keys()):
                        tofile[key][()] = results[key]
                    else:
                        tofile.create_dataset(key, 
                                              data=results[key])
            if self.sfs.check_save_background.isChecked():
                with h5py.File(f"{self.save_folder_path}/results.h5", "a") as tofile:
                    if "background" in list(tofile.keys()):
                        tofile["background"][()] = background
                    else:
                        tofile.create_dataset(f"{self.data.file_name}/background", data=background)
            if self.sfs.check_save_fitted_spectrum.isChecked():
                with h5py.File(f"{self.save_folder_path}/results.h5", "a") as tofile:
                    if "fitted spectra" in list(tofile.keys()):
                        tofile["fitted spectra"][()] = fitted_spectra
                    else:
                        tofile.create_dataset(f"{self.data.file_name}/fitted spectra", data=fitted_spectra)
            self.popup_properties.save_props()
        elif self.data.loadtype in ["folder", "msa_file", "hdf5_file", "bcf_file", "csv_file"]:
            show_fit_prop_in_popup()
            # check if the data file was splitted
            datah5_files = ns.natsorted(glob(self.data.folder_path+"/data/data_*.h5"))
            if len(datah5_files) != 0:
                # if splitted files are detected, ask if batch fitting should
                # be performed or data.h5
                self.data_batch_fitting = QtWidgets.QMessageBox.question(self, "?", 
                                                                    "Splitted data.h5 detected. Do you want to batch fit or use data.h5?", 
                                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                                                                    QtWidgets.QMessageBox.No)
                if self.data_batch_fitting:
                    data_batches = len(datah5_files)
                    self.sfs.check_batch_fitting.setChecked(False)
            else:
                data_batches = 1
            # now iterate over all data_batches
            for data_batch in range(data_batches):
                # now load into specfit.data the data from
                if self.data_batch_fitting:
                    self.data.update_data_from_h5file(datah5_files[data_batch])
                # get the number of spectra
                partitions = len(self.data.spectra)
                # check if batch fitting is activated and create a batch_iterator
                self.batch_fitting = self.sfs.check_batch_fitting.checkState().value
                # check the type of save_storage
                if self.sfs.check_save_storage.checkState().value:
                    save_storage = ".npy"
                else:
                    save_storage = ".h5"
                if self.batch_fitting:
                    batch_iterator = -1
                else:
                    batch_iterator = 0
                # here the partitions of the loaded spectra are stored in a sorted order
                if not self.batch_fitting:
                    results = create_empty_results(load_type=self.data.loadtype, 
                                                   selected_elements=self.pse_widget.selected_elements, 
                                                   selected_lines=self.pse_widget.selected_lines, 
                                                   batch_fitting=False, 
                                                   dimension=np.copy(self.data.position_dimension))
                # if the background should be save, an empty container will be created here
                if self.sfs.check_save_background.isChecked():
                    background = np.zeros((partitions, len(self.data.energies)))
                if self.sfs.check_save_fitted_spectrum.isChecked():
                    fitted_spectra = np.zeros((partitions, len(self.data.energies)))
                # start the clock for calculation time
                time_file_start = time.time()
                # now sort the tensor_positions to be compatible with the
                # sorted spectra dict
                if not isinstance(self.data.position_dimension[0], np.ndarray):
                    self.data.tensor_positions = np.asarray(ns.natsorted(self.data.tensor_positions))
                # start the deconvolution for every measurement point
                for i in range(partitions):
                    # set the fit in progress to True if first deconvolution was
                    # succesfully performed
                    if i == 1:
                        self.s.fit_in_progress = True
                    # if batch fitting is activated create a new results for every
                    # row in the measured map
                    if self.batch_fitting:
                        if i % self.data.position_dimension[1] == 0:
                            # if already on row was deconvolved, save it in the results h5
                            if batch_iterator > -1:
                                save_results(results=results, 
                                             save_path=self.save_folder_path, 
                                             batch_fitting=self.batch_fitting, 
                                             batch_iterator=batch_iterator, 
                                             file_type=self.data.file_type, 
                                             save_storage=save_storage, 
                                             dimension=np.copy(self.data.position_dimension))
                            # create an empty list
                            results = create_empty_results(load_type=self.data.loadtype, 
                                                           selected_elements=self.pse_widget.selected_elements, 
                                                           selected_lines=self.pse_widget.selected_lines, 
                                                           batch_fitting=True, 
                                                           dimension=np.copy(self.data.position_dimension))
                            batch_iterator += 1
                    # try retrieving spectrum data, or load spectrum from
                    # single_spectra folder
                    try:
                        cur_spectra = self.data.spectra[i]
                    except IndexError:
                        cur_spectra = np.load(f"{self.folder_path}/single_spectra/spectrum_{i}.npy")
                    # if the set minimal count rate per spectrum is exceeded
                    # perform the deconvolution
                    if self.data.mincount == 0 or np.sum(cur_spectra[roi_min:roi_max])> self.data.mincount:# TODO geht nicht mit einzelspektren
                        self.roi_widget.spec_nr = i
                        # load the parameters of the spectrum into specfit deconvolution
                        self.load_parameter_in_specfit_deconvolution()
                        self.load_spec_in_specfit_deconvolution(spectra_nr=i)
                        # if the background is to be set to 0 (low scattering), 
                        # set the background s.Strip to zero
                        if self.data.bg_zero is True:
                            self.s.Strip = np.zeros(len(self.s.meas_load))
                        else:
                            if self.check_calc_minima.isChecked():
                                self.s.strip()
                            else:
                                self.s.strip(calc_minima=False)
                        # if the background should be saved, store the estimated
                        # background s.Strip to the background array
                        if self.sfs.check_save_background.isChecked():
                            background[i] = self.s.Strip
                        if self.pse_widget.tab_udl.number_lines > 0:
                            parameters = self.get_selected_parameters()
                            self.pse_widget.tab_udl.load_spec((np.subtract(self.s.meas_load, self.s.Strip)), parameters[0], parameters[1], str(i) )
                            self.s.udl_label_list, self.s.user_defined_lines = self.pse_widget.tab_udl.get_user_defined_lines()
                        self.s.linfit()
                        if self.sfs.check_save_fitted_spectrum.isChecked():
                            fitted_spectra[i] = self.s.calc_spec()
                        self.plot_canvas(self.s.meas_load, self.data.energies, self.s.Strip, self.s.calc_spec(), 
                                         initialize=False)
                        # now get all fit-results
                        getResults = self.s.get_result()
                        # and now fill them into the results array on the corresponding
                        # position
                        for k, key in enumerate(results.keys()):
                            if not isinstance(self.data.position_dimension[0], np.ndarray):
                                if self.batch_fitting:
                                    results[key][0][self.data.tensor_positions[i, 1]][self.data.tensor_positions[i, 2]] = getResults[key]["I"]
                                else:
                                    results[key][self.data.tensor_positions[i, 0]][self.data.tensor_positions[i, 1]][self.data.tensor_positions[i, 2]] = getResults[key]["I"]
                            else:
                                for scan, sum_len_scan in enumerate(self.data.sum_len_scans):
                                    if i in sum_len_scan:
                                        element_key = "_".join(key.split("_")[:-1])+f"{scan:d}"
                                        if len(self.data.len_scans) > 1:
                                            index_0 = self.data.tensor_positions[scan][0][
                                                i - np.sum(self.data.len_scans[:scan]), 0]
                                            index_1 = self.data.tensor_positions[scan][0][
                                                i - np.sum(self.data.len_scans[:scan]), 1]
                                            index_2 = self.data.tensor_positions[scan][0][
                                                i - np.sum(self.data.len_scans[:scan]), 2]
                                        else:
                                            index_0 = self.data.tensor_positions[scan][i][0]
                                            index_1 = self.data.tensor_positions[scan][i][1]
                                            index_2 = self.data.tensor_positions[scan][i][2]
                                        results[element_key][index_0][index_1][index_2] = \
                                            getResults["_".join(key.split("_")[:-2])]["I"]
                                        continue
                    else:
                        for key in self.s.get_result_keys():
                            results[key][self.data.tensor_positions[i][0]][self.data.tensor_positions[i][1]][
                                self.data.tensor_positions[i][2]] = 0
                        continue
                    # here the progress of the fitting will be calculated and displayed
                    if i % 1 == 0:
                        time_file_end = time.time()
                        time2go = time.gmtime((time_file_end-time_file_start)*(partitions-(i+1))/(i+1))
                    self.progress_bar.setValue(int(((i+1)//len(self.data.spectra))*100))
                    self.statusBar().showMessage(f"progress : batch {data_batch+1}/{len(datah5_files)} - {i+1}/{partitions} - {(i+1)/float(partitions)*100:.2f}% - time to go: {time.strftime('%H:%M:%S', time2go)} s")
                    QtWidgets.QApplication.processEvents()
                # save the fluorescence intensities to the results.h5 file
                # if the data file is splitted in batches, save each batch
                # as new results in results.h5
                if self.data_batch_fitting:
                    save_results(results=results, 
                                 save_path=self.save_folder_path, 
                                 batch_fitting=True, 
                                 batch_iterator=data_batch, 
                                 file_type=self.data.file_type, 
                                 save_storage=save_storage, 
                                 dimension=np.copy(self.data.position_dimension))
                else:
                    save_results(results=results, 
                                 save_path=self.save_folder_path, 
                                 batch_fitting=self.batch_fitting, 
                                 batch_iterator=batch_iterator, 
                                 file_type=self.data.file_type, 
                                 save_storage=save_storage, 
                                 dimension=np.copy(self.data.position_dimension))
                # if the background should be saved, store the background to the
                # results.h5 file
                if self.sfs.check_save_background.isChecked():
                    if self.data_batch_fitting:
                        background_key = f"{self.data.file_name}/background_{data_batch}"
                    else:
                        background_key = f"{self.data.file_name}/background"
                    with h5py.File(f"{self.save_folder_path}/results.h5", "a") as tofile:
                        if background_key in list(tofile.keys()):
                            tofile[background_key][()] = background
                        else:
                            tofile.create_dataset(background_key, data=background)
                # the fitted spectra should be saved, store them into results.h5
                if self.sfs.check_save_fitted_spectrum.isChecked():
                    if self.data_batch_fitting:
                        fitted_spectra_key = f"fitted spectra_{data_batch}"
                    else:
                        fitted_spectra_key = "fitted spectra"
                    with h5py.File(f"{self.save_folder_path}/results.h5", "a") as tofile:
                        if fitted_spectra_key in list(tofile.keys()):
                            tofile[fitted_spectra_key][()] = fitted_spectra
                        else:
                            tofile.create_dataset(fitted_spectra_key, data=fitted_spectra)
        # if batch fitting all the single batches are concatenated to one
        # result in the results.h5
        if self.batch_fitting or self.data_batch_fitting:
            if save_storage == ".h5":
                with h5py.File(f"{self.save_folder_path}/results.h5", "r+") as tofile:
                    unique_keys = np.unique(["_".join(key.split("_")[:2]) for key in tofile.keys()])
                    if self.batch_fitting:
                        rows = self.data.position_dimension[0]
                    elif self.data_batch_fitting:
                        rows = data_batch+1
                    for results_key in unique_keys:
                        print("the results key here is noted as ", results_key)
                        results_key = f"{self.data.file_name}/{results_key}"
                        if results_key+"_0" in tofile:
                            results = tofile[results_key+"_0"][()]
                            del tofile[results_key+"_0"]
                            for batch in range(rows-1):
                                results = np.concatenate((results, tofile[results_key+f"_{batch+1}"]), axis=0)
                                del tofile[results_key+f"_{batch+1}"]
                            if results_key in tofile:
                                tofile[results_key][()] = results
                            else:
                                tofile.create_dataset(results_key, 
                                                      data=results)
            else:
                files = glob(self.save_folder_path + "/*.npy")
                unique_keys = np.unique(["_".join(key.split("/")[-1].split("_")[:2]) for key in files])
                files = [key.split("/")[-1].replace(".npy", "") for key in files]
                rows = self.data.position_dimension[0]
                for results_key in unique_keys:
                    if results_key + "_0" in files:
                        results = np.load(self.save_folder_path+f"/{results_key}_0.npy")
                        os.remove(self.save_folder_path+f"/{results_key}_0.npy")
                    for batch in range(rows-1):
                        results = np.concatenate((results, np.load(self.save_folder_path+f"/{results_key}_{batch+1}.npy")), axis=0)
                        os.remove(self.save_folder_path+f"/{results_key}_{batch+1}.npy")
                    np.save(self.save_folder_path+f"/{results_key}.npy", results)
        self.popup_properties.save_props()
        compute_time_end = time.time()
        self.pse_widget.tab_udl.print_gaussian_fit_error_results()
        self.progress_bar.hide()
        self.statusBar().showMessage(f"saving done - time : {time.strftime('%H:%M:%S', time.gmtime(compute_time_end-compute_time_start))} s")

    # define functions used by plot interaction
    def _on_button_press(self, event):
        self.pse_widget.tab_line_finder.find_nearest_lines(event.xdata)
        if self.selected_energy is None:
            self.selected_energy = self.ax_canvas_spectrum.axvline(event.xdata, 
                                                                   ymax=self.ax_canvas_spectrum.get_ylim()[-1] / 2, 
                                                                   color="g")
        else:
            self.selected_energy.set_xdata([event.xdata])
        self.activateWindow()
        self.canvas_spectrum.draw()

def main():
    module_list = ["showing ROI", "deconvoultion", "major important validation", 
                   "data processing", "mathematical foundation", 
                   "3D plotting", "SpecFit is awesome", "spectra analyzation", 
                   "periodic table", "data base", "system optimization"]
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.getcwd()+"/Data/images/specfit_logo_256x256.png"))
    app.setStyle("GTK+")
    pixmap = QtGui.QPixmap(os.getcwd()+"/Data/images/logo_ls.png")
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    for module in module_list:
        splash.showMessage(f"{module} is set up", QtCore.Qt.AlignBottom, 
                           QtCore.Qt.red), app.processEvents(), time.sleep(np.random.uniform(0., 0.3))
    splash.showMessage("done", QtCore.Qt.AlignBottom, QtCore.Qt.red), app.processEvents()
    specFit_GUI = SpecFitGUIMain()
    splash.finish(specFit_GUI)
    specFit_GUI.show()
    app.exec_()
    # now close all hdf5 files, which are still open
    import tables
    tables.file._open_files.close_all()

if __name__ == "__main__":
    main()
