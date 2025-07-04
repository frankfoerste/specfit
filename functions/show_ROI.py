import os
import sys
import h5py
import time as t
import numpy as np
import dask.array as da
from pathlib import Path
from PyQt6 import QtGui, QtWidgets, QtCore
import matplotlib.backends.backend_qt5agg as pltqt
import matplotlib.figure as figure
from functools import partial
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import xraylib as xrl

file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = Path(__file__).parent
parent_dir = file_dir.parent
elements_path = parent_dir / "Data" / "elements.dat"
lines_path = parent_dir / "Data" / "lines.dat"

# here the list of elements is read out of the file elements.dat
elements = []
with open(elements_path, "r") as element_file:
    for line in element_file:
        line = line.replace("\n", "").replace(" ", "").split("\t")
        elements.append([int(line[0]), line[1], int(line[0])])
        if line[0] == "98":
            break
# here the list of lines is read out of the file lines.dat
lines = []
with open(lines_path, "r") as lines_file:
    for line in lines_file:
        line = line.replace("\n", "").split("\t")
        lines.append([line[0], line[1]])

class ShowROI(QtWidgets.QWidget):
    """
    This class calculates the intensities for given ROIs and given elements.
    To initialize the class 2 inputs are required.

    Parameters
    ----------
    spectra : list
        list containing the spectra
    energy : list
        list containing the energy axis
    elements: dict
        {element : [Lines]}
        {Fe : [K-Line]}
    delta_E : float
        FWHM /2 of the measurement
    """
    def __init__(self, parent=None):
        super(ShowROI, self).__init__()
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.popup_heigth = 500
        self.popup_width = 850
        self.setWindowTitle("show ROI")
        self.working_directory = os.getcwd()
        self.setGeometry((self.screen_width-self.popup_width)//2,
                         (self.screen_height-self.popup_heigth)//2,
                         self.popup_width, self.popup_heigth)
        self.time = t.time()
        self.elements = elements[5:84]
        self.para_a0 = None  # this is the a0 entry which show_ROI inherits from the SpecFit Main Window
        self.para_a1 = None  # this is the a1 entry which show_ROI inherits from the SpecFit Main Window
        self.roi_low = None  # this is the low ROI entry which show_ROI inherits from the SpecFit Main Window
        self.roi_high = None  # this is the high ROI entry which show_ROI inherits from the SpecFit Main Window
        self.plot_ROI = False  # this is to determine if a plot already exists
        self.delta_E = 0.133  # energy range for the evaluation of the ROI intensity
        self.rotation = "xy"  # initialize selected rotation
        self.file_type = None  # initialize the file type of the shown data set
        self.spec_nr = -1  # negativemeans sum_spec
        for i, _ in enumerate(self.elements):
            self.elements[i] = f"{self.elements[i][0]} - {self.elements[i][1]}"
        self.lines = lines[1:6]
        for i, _ in enumerate(self.lines):
            self.lines[i] = self.lines[i][0]
        self.layout_show_ROI = QtWidgets.QGridLayout()
        self.setStyleSheet("QWidget { color: black; background-color:white;}"\
                           +"background-color:white;"\
                           +"QLabel {font-size: 10px; max-width: 60px} "\
                           +"QLineEdit {font-size: 10px; max-height: 18px; max-width: 60px} "\
                           +"QCheckBox {font-size: 10px; max-height: 18px; max-width: 60px} "\
                           +"QPushButton {font-size: 10px; max-height: 18px; max-width: 80px}"\
                           +"QRadioButton {font-size: 10px; max-height: 18px; max-width: 80px} "\
                           +"QComboBox {font-size: 10px; max-height: 18px; max-width: 80px} "\
                           +"QTextEdit {font-size: 10px; max-width: 60px}")
        self.rect_sizes = ["1x1", "3x3", "5x5", "7x7", "9x9", "11x11"]
        self.first_time_loaded = True
        self.parent = parent
        self.init_UI()
        self.__init__plot()
        self.first_time_loaded = False

    def init_UI(self):
        """
        initialize widgets
        """
        self.label_layer = QtWidgets.QLabel("Layer", self)
        self.label_layer.setFixedWidth(40)
        self.label_delta_E = QtWidgets.QLabel(u"\u0394 E", self)
        self.button_show_ROI = QtWidgets.QPushButton("Show ROI", self)
        self.button_reset_ROI = QtWidgets.QPushButton("Reset ROI", self)
        self.button_reset_ROI.hide()
        self.slider_layer = QtWidgets.QSlider(QtCore.Qt.Vertical, self)
        self.slider_layer.setFixedWidth(40)
        self.radio_xy = QtWidgets.QRadioButton("xy", self)
        self.radio_xz = QtWidgets.QRadioButton("xz", self)
        self.radio_yz = QtWidgets.QRadioButton("yz", self)
        self.radio_xenergy = QtWidgets.QRadioButton("x_energy", self)
        self.radio_xenergy.setVisible(False)       #only visible if angle file is loaded
        self.slider_percentile = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider_percentile.setMinimum(0)
        self.slider_percentile.setMaximum(100)
        self.slider_percentile.setValue(100)
        self.slider_percentile.valueChanged.connect(self.plot_results)
        self.combo_elements = QtWidgets.QComboBox(self)
        self.combo_elements.currentTextChanged.connect(self.plot_line_energy)
        self.combo_lines = QtWidgets.QComboBox(self)
        self.combo_rect_size = QtWidgets.QComboBox(self)
        self.button_save_ROI = QtWidgets.QPushButton("save ROI", self)
        self.button_save_ROI.hide()
        self.button_save_selection = QtWidgets.QPushButton("save selection", self)
        self.button_save_selection.hide()
        # self.combo_rect_size
        self.entry_position = QtWidgets.QLineEdit("", self)
        self.entry_position.setFixedWidth(40)
        self.entry_delta_E = QtWidgets.QLineEdit(str(self.delta_E), self)
        self.entry_delta_E.setMinimumWidth(100)
        self.log_box = QtWidgets.QCheckBox("log", self)
        self.log_box.setChecked(True)
        self.log_box.setVisible(False)
        # set properties of widgets #
        self.layout_show_ROI.addWidget(self.label_layer, 0, 9, 1, 1)
        self.layout_show_ROI.addWidget(self.label_delta_E, 0, 0, 1, 1)
        self.slider_layer.setMinimum(0)
        self.slider_layer.setMaximum(0)
        self.slider_layer.setTickInterval(1)
        self.slider_layer.setTickPosition(QtWidgets.QSlider.TicksRight)
        self.layout_show_ROI.addWidget(self.slider_layer, 1, 9, 1, 1)
        self.layout_show_ROI.addWidget(self.radio_xy, 0, 4, 1, 1)
        self.radio_xy.setChecked(True)
        self.layout_show_ROI.addWidget(self.radio_xz, 0, 5, 1, 1)
        self.layout_show_ROI.addWidget(self.radio_yz, 0, 6, 1, 1)
        self.layout_show_ROI.addWidget(self.radio_xenergy, 0, 7, 1, 1)
        self.layout_show_ROI.addWidget(self.slider_percentile, 2, 4, 1, 4)
        self.layout_show_ROI.addWidget(self.combo_elements, 3, 4, 1, 1)
        self.layout_show_ROI.addWidget(self.combo_lines, 3, 5, 1, 1)
        self.layout_show_ROI.addWidget(self.button_show_ROI, 3, 6, 1, 1)
        self.layout_show_ROI.addWidget(self.button_reset_ROI, 3, 7, 1, 1)
        self.button_reset_ROI.setStyleSheet("QPushButton {color : white; background-color:red}")
        self.layout_show_ROI.addWidget(self.log_box, 3, 8, 1, 1)
        self.layout_show_ROI.addWidget(self.combo_rect_size, 4, 4, 1, 1)
        self.layout_show_ROI.addWidget(self.button_save_ROI, 4, 5, 1, 1)
        self.layout_show_ROI.addWidget(self.button_save_selection, 4, 6, 1, 1)
        self.button_show_ROI.clicked.connect(self.evaluate_roi)
        self.button_reset_ROI.clicked.connect(self.reset_ROI)
        self.button_save_ROI.clicked.connect(self.save_ROI)
        self.button_save_ROI.setStyleSheet("QPushButton {background-color:lightgreen}")
        self.button_save_selection.clicked.connect(self.save_selection)
        self.button_save_selection.setStyleSheet("QPushButton {background-color:lightblue}")
        self.slider_layer.valueChanged.connect(self.plot_results)
        self.radio_xy.clicked.connect(partial(self.rotate_results, "xy"))
        self.radio_xz.clicked.connect(partial(self.rotate_results, "xz"))
        self.radio_yz.clicked.connect(partial(self.rotate_results, "yz"))
        self.radio_xenergy.toggled.connect(self.show_colormap)
        self.log_box.stateChanged.connect(self.show_colormap)
        self.layout_show_ROI.addWidget(self.entry_position, 3, 9, 1, 1)
        self.layout_show_ROI.addWidget(self.entry_delta_E, 0, 1, 1, 2)
        self.entry_position.setText("%.7f"%self.spec_nr)
        self.setLayout(self.layout_show_ROI)
        try:
            self.combo_elements.addItems(self.elements)
        except:
            pass
        try:
            self.combo_lines.addItems(self.lines)
        except:
            pass
        self.combo_rect_size.addItems(self.rect_sizes)
        self.plot_style_str = "linear"

    def __init__plot(self):
        """
        define the layout of the plot frame
        """
        self.figure_ROI = figure.Figure(dpi=70)
        self.figure_sum_spec = figure.Figure(dpi=70)
        self.canvas_roi = pltqt.FigureCanvasQTAgg(self.figure_ROI)
        self.canvas_roi.setParent(self)
        self.canvas_spectrum = pltqt.FigureCanvasQTAgg(self.figure_sum_spec)
        self.canvas_spectrum.setParent(self)
        self.layout_show_ROI.addWidget(self.canvas_roi, 1, 4, 1, 5)
        self.layout_show_ROI.addWidget(self.canvas_spectrum, 1, 0, 1, 4)
        self.toolbar_ROI = pltqt.NavigationToolbar2QT(self.canvas_roi, self)
        self.toolbar_ROI.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        self.toolbar_sum_spec = pltqt.NavigationToolbar2QT(self.canvas_spectrum, self)
        self.toolbar_sum_spec.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        self.layout_show_ROI.addWidget(self.toolbar_ROI, 5, 4, 1, 5)
        self.layout_show_ROI.addWidget(self.toolbar_sum_spec, 5, 0, 1, 3)
        self.toolbar_ROI.actions()[0].setIcon(QtGui.QIcon(self.working_directory+"/Data/icons/bug.png"))
        # establish connections with User action
        self.canvas_roi.mpl_connect("button_press_event", self.plot_single_spectra_on_click)
        self.canvas_roi.mpl_connect("button_release_event", self.draw_rectangle)
        # create an axis
        self.ax_canvas_roi = self.figure_ROI.add_subplot(111)
        self.ax_canvas_spectrum = self.figure_sum_spec.add_subplot(111)
        self.ax_canvas_roi.set_xlabel("")
        self.ax_canvas_roi.set_ylabel("")
        self.ax_canvas_spectrum.set_xlabel("Energy / keV")
        self.ax_canvas_spectrum.set_ylabel("Intensity / cps")
        self.roi_plotted = False

    def display_show_ROI(self, ):
        try:
            self.parent.data.file_type
        except:
            self.parent.statusBar().showMessage("no measurement loaded")
            return
        self.show()
        self.activateWindow()

    def load_spectra(self, folder_path, save_folder_path, save_data_path,
                     load_type, one_dim=False):
        """
        load spectra into the show ROI widget
        """
        # make sure the loaded data is not a single spectrum, else return
        if self.parent.data.loadtype == "file":
            self.parent.statusBar().showMessage("no ROI for single spectrum")
            return
        # receive and self the folder path
        self.folder_path = folder_path
        # receive and self the save folder path
        self.save_folder_path = save_folder_path
        # receive and self the save data path
        self.save_data_path = save_data_path
        # receive and self the load_type (.spx, .txt, ...)
        self.load_type = load_type
        # now load all spectra from pickle
        with h5py.File(f"{self.save_data_path}/data.h5", "r") as infile:
            if "positions" in infile.keys():
                file_key = ""
            else:
                file_key = f"{self.parent.data.file_name}/"
            self.spectra = da.asarray(infile[f"{file_key}spectra"][()])
            self.angles = infile[f"{file_key}positions"][()]
            self.positions = infile[f"{file_key}positions"][()]
            self.tensor_positions = infile[f"{file_key}tensor positions"][()]
            self.parameters = infile[f"{file_key}parameters"][()]
            self.counts = infile[f"{file_key}counts"][()]
            self.len_x, self.len_y, self.len_z = infile[f"{file_key}position dimension"][()]
        # check dimension of loaded data
        if one_dim:
            self.radio_xenergy.setVisible(True)
            self.log_box.setVisible(False)
            self.one_dim = True
            # load the positions of the loaded spectra
        else:
            self.radio_xenergy.setVisible(False)
            self.log_box.setVisible(False)
            self.one_dim = False
            with h5py.File(f"{self.save_data_path}/data.h5", "r") as infile:
                if "positions" in infile.keys():
                    file_key = ""
                else:
                    file_key = f"{self.parent.data.file_name}/"
                self.positions = infile[f"{file_key}positions"][()]
                self.tensor_positions = infile[f"{file_key}tensor positions"][()]
                self.counts = infile[f"{file_key}counts"][()]
                self.parameters = infile[f"{file_key}parameters"][()]
                self.len_x, self.len_y, self.len_z = infile[f"{file_key}position dimension"][()]
        # get the points of every axis
        self.x = np.unique(self.positions[:, 0])
        self.y = np.unique(self.positions[:, 1])
        self.z = np.unique(self.positions[:, 2])
        # find the xyz-position of the origin
        origin_idx = int(np.where((self.tensor_positions == np.array([0, 0, 0])).all(1))[0])
        end_idx = int(np.where((self.tensor_positions == np.array([self.tensor_positions[:, 0].max(),
                                                                   self.tensor_positions[:, 1].max(),
                                                                   self.tensor_positions[:, 2].max()])).all(1))[0])
        # sort the unique points corresponding to the measurement
        if self.x[0] != self.positions[:, 0][origin_idx]:
            self.x = self.x[::-1]
        if self.y[0] != self.positions[:, 1][origin_idx]:
            self.y = self.y[::-1]
        if self.z[0] != self.positions[:, 2][origin_idx]:
            self.z = self.z[::-1]
        # determine step size for every axis
        try:
            self.step_x = np.nan_to_num(
                np.round((self.positions[:, 0][end_idx] - self.positions[:, 0][origin_idx]) / (len(self.x) - 1), 3),
                nan=1.)
        except:
            self.step_x = 1.
        try:
            self.step_y = np.nan_to_num(
                np.round((self.positions[:, 1][end_idx] - self.positions[:, 1][origin_idx]) / (len(self.y) - 1), 3),
                nan=1.)
        except:
            self.step_y = 1.
        try:
            self.step_z = np.nan_to_num(
                np.round((self.positions[:, 2][end_idx] - self.positions[:, 2][origin_idx]) / (len(self.z) - 1), 3),
                nan=1.)
        except:
            self.step_z = 1.
        # handle data type specific exeptions
        if not isinstance(self.parameters[0], np.float64) and load_type not in ["angle_file", "file"]:
            self.parameters = self.parameters[0]
        try:
            self.counts = self.counts.reshape(self.positions)
        except:
            if self.file_type == ".MSA":
                self.counts = self.counts.reshape((self.len_z, self.len_y, self.len_x))
                self.counts = np.flip(self.counts, 0)
                x = np.copy(self.x)
                self.x = np.copy(self.z)
                self.z = np.copy(x)
            elif self.file_type == ".bcf":
                self.counts = np.reshape(self.counts, (self.len_x, self.len_y, self.len_z))
            else:
                self.counts = self.counts.reshape((self.len_x, self.len_y, self.len_z))
        # back up the counts
        self.counts_backup = np.copy(self.counts)
        # display the sum_spec
        self.sum_spec = self.parent.data.sum_spec
        self.delta_E = self.parameters[3]
        self.entry_delta_E.setText("%f"%self.delta_E)
        self.slider_layer.setMaximum(self.counts.shape[-1]-1)
        self.evaluate_energy()
        self.roi_plotted = False
        self.ax_canvas_roi.set_title("Counts")
        self.plot_results()
        self.plot_sum_spec()

    def reset_ROI(self):
        """
        reset the ROI so that again the whole energie range is displayed
        """
        self.roi_plotted = False
        self.counts = np.copy(self.counts_backup)
        self.results = np.zeros(self.counts.shape)
        self.button_save_ROI.hide()
        self.button_save_selection.hide()
        self.button_reset_ROI.hide()
        self.plot_ROI = None
        self.plot_results()

    def get_line_energy(self):
        """
        read out the energy of the selected element fluorescence line
        """
        self.elements = self.combo_elements.currentText().split(" - ")
        self.element_str = self.elements[1]
        self.element_Z = xrl.SymbolToAtomicNumber(self.element_str)
        self.elements[0] = int(self.elements[0])
        self.line = self.combo_lines.currentText()
        if self.line == "Ka":
            self.elements[1] = [xrl.LineEnergy(self.element_Z, xrl.KA1_LINE)]
        elif self.line == "Kb":
            self.elements[1] = [xrl.LineEnergy(self.element_Z, xrl.KB1_LINE)]
        elif self.line =="L3":
            self.elements[1] = [xrl.LineEnergy(self.element_Z, xrl.L3M5_LINE)]
        elif self.line =="L2":
            self.elements[1] = [xrl.LineEnergy(self.element_Z, xrl.L2M4_LINE)]
        elif self.line =="L1":
            self.elements[1] = [xrl.LineEnergy(self.element_Z, xrl.L1M3_LINE)]

    def evaluate_roi(self):
        self.plot_line_energy()
        self.delta_E = float(self.entry_delta_E.text()) # keV
        self.get_line_energy()
        self.calc_roi_intensity()
        self.results = np.copy(self.results.reshape(self.counts.shape))
        if self.file_type in [".MSA"]:
            self.results = np.flip(self.results, 0)
        self.slider_layer.setMaximum(self.results.shape[-1]-1)
        self.button_save_ROI.show()
        self.button_save_selection.show()
        self.button_reset_ROI.show()
        self.roi_plotted = True
        self.plot_results()

    def calc_roi_intensity(self):
        """
        calculate the intensities of the given ROI
        """
        index_high = (np.abs(self.energy - self.elements[1][0] - self.delta_E)).argmin()
        index_low = (np.abs(self.energy - self.elements[1][0] + self.delta_E)).argmin()
        self.results = np.sum(self.spectra[..., index_low:index_high], axis=-1)

    def evaluate_energy(self):
        """
        read out the energie from the loaded data
        """
        self.energy = np.copy(self.parent.data.energies)

    def plot_sum_spec(self):
        if not self.first_time_loaded:
            self.ax_canvas_spectrum.clear()
            self.evaluate_energy()
            low_index = int((float(self.roi_low.text())-self.parameters[0])/self.parameters[1])
            high_index = int((float(self.roi_high.text())-self.parameters[0])/self.parameters[1])
            self.ax_canvas_spectrum.plot(self.energy, self.sum_spec)
            self.ax_canvas_spectrum.set_xlim(float(self.roi_low.text()), float(self.roi_high.text()))
            self.ax_canvas_spectrum.set_ylim(np.min(self.sum_spec[low_index: high_index])*0.9,
                                             np.max(self.sum_spec[low_index: high_index])*1.1)
            self.ax_canvas_spectrum.set_xlabel("Energy / keV")
            self.ax_canvas_spectrum.set_ylabel("Intensity / cps")
            self.ax_canvas_spectrum.set_title("Sum Spectrum")
            self.canvas_spectrum.draw()

    def plot_line_energy(self):
        self.get_line_energy()
        self.plot_sum_spec()
        self.delta_E = float(self.entry_delta_E.text())
        if not self.first_time_loaded:
            self.ax_canvas_spectrum.axvspan(self.elements[1][0]-self.delta_E,
                                            self.elements[1][0]+self.delta_E,
                                            color="r", alpha=0.3)
            self.canvas_spectrum.draw()

    def plot_results(self):
        self.entry_position.setText("")
        layer = self.slider_layer.value()
        aspect = "auto"
        if self.one_dim:
            y_dim, x_dim = self.counts[layer].shape
        line = self.line.replace("Ka", u"K\u03B1").replace("Kb", u"K\u03b2")
        if not self.plot_ROI:
            if self.roi_plotted is True:
                maximum = np.percentile(self.results, self.slider_percentile.value())
                if self.rotation == "xy":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.results[:, :, layer], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.x[-1] + self.step_x / 2,
                                                                      self.x[0] - self.step_x / 2,
                                                                      self.y[-1] + self.step_y / 2,
                                                                      self.y[0] - self.step_y / 2])
                if self.rotation == "xz":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.results[:, layer, :], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.x[-1] + self.step_x / 2,
                                                                      self.x[0] - self.step_x / 2,
                                                                      self.z[-1] + self.step_z / 2,
                                                                      self.z[0] - self.step_z / 2])
                if self.rotation == "yz":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.results[layer, :, :], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.y[-1] + self.step_y / 2,
                                                                      self.y[0] - self.step_y / 2,
                                                                      self.z[-1] + self.step_z / 2,
                                                                      self.z[0] - self.step_z / 2])
            else:
                maximum = np.percentile(self.counts, self.slider_percentile.value())
                if self.rotation == "xy":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.counts[:, :, layer], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.x[-1] + self.step_x / 2,
                                                                      self.x[0] - self.step_x / 2,
                                                                      self.y[-1] + self.step_y / 2,
                                                                      self.y[0] - self.step_y / 2])
                elif self.rotation == "xz":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.counts[:, layer, :], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.x[-1] + self.step_x / 2,
                                                                      self.x[0] - self.step_x / 2,
                                                                      self.z[-1] + self.step_z / 2,
                                                                      self.z[0] - self.step_z / 2])
                elif self.rotation == "yz":
                    self.plot_ROI = self.ax_canvas_roi.imshow(np.rot90(self.counts[layer, :, :], k=-1),
                                                              vmin=0,
                                                              vmax=maximum, aspect=aspect,
                                                              extent=[self.y[-1] + self.step_y / 2,
                                                                      self.y[0] - self.step_y / 2,
                                                                      self.z[-1] + self.step_z / 2,
                                                                      self.z[0] - self.step_z / 2])
        else:
            if self.roi_plotted is True:
                maximum = np.percentile(self.results, self.slider_percentile.value())
                if self.rotation == "xy":
                    self.plot_ROI.set_data(np.rot90(self.results[:, :, layer], k=-1))
                elif self.rotation == "xz":
                    self.plot_ROI.set_data(np.rot90(self.results[:, layer, :], k=-1))
                elif self.rotation == "yz":
                    self.plot_ROI.set_data(np.rot90(self.results[layer, :, :], k=-1))
            else:
                maximum = np.percentile(self.counts, self.slider_percentile.value())
                if self.rotation == "xy":
                    self.plot_ROI.set_data(np.rot90(self.counts[:, :, layer], k=-1))
                if self.rotation == "xz":
                    self.plot_ROI.set_data(np.rot90(self.counts[:, layer, :], k=-1))
                if self.rotation == "yz":
                    self.plot_ROI.set_data(np.rot90(self.counts[layer, :, :], k=-1))
        self.plot_ROI.set_clim(0, maximum)
        try: self.colorbar.remove()
        except: pass
        self.colorbar = self.figure_ROI.colorbar(self.plot_ROI,
                                                 ax=self.ax_canvas_roi)
        if self.rotation == "xy":
            layer = self.z[layer]
        elif self.rotation == "xz":
            layer = self.y[layer]
        else: layer = self.x[layer]
        if self.file_type == ".spx":
            if self.roi_plotted is True:
                self.ax_canvas_roi.set_title("ROI Counts %s %s | %s position : %.3f mm"%(self.element_str,
                                                                                     line,
                                                                                     "xyz".replace(self.rotation[0], "").replace(self.rotation[1], ""),
                                                                                     layer))
            else:
                self.ax_canvas_roi.set_title("Counts | %s position : %.3f mm"%("xyz".replace(self.rotation[0], "").replace(self.rotation[1], ""),
                                                                            layer))
            self.ax_canvas_roi.set_ylabel(u"%s / mm"%self.rotation[1])
            self.ax_canvas_roi.set_xlabel(u"%s / mm"%self.rotation[0])
        else:
            if self.roi_plotted is True:
                self.ax_canvas_roi.set_title("ROI Counts %s %s | %s position : %i"%(self.element_str,
                                                                                     line,
                                                                                     "xyz".replace(self.rotation[0], "").replace(self.rotation[1], ""),
                                                                                     layer))
            else:
                self.ax_canvas_roi.set_title("Counts | %s position : %i"%("xyz".replace(self.rotation[0], "").replace(self.rotation[1], ""),
                                                                            layer))
            self.ax_canvas_roi.set_ylabel(u"%s"%self.rotation[1])
            self.ax_canvas_roi.set_xlabel(u"%s"%self.rotation[0])
        self.canvas_roi.draw()

    def rotate_results(self, rotation):
        self.slider_layer.setVisible(True)
        self.log_box.setVisible(False)
        self.rotation = rotation
        self.slider_layer.setValue(0)
        if rotation == "xy":                                     # in order to get the natural orientation the data have to be rotated
            if self.roi_plotted is not False:
                self.slider_layer.setMaximum(self.results.shape[2]-1)
            else:
                self.slider_layer.setMaximum(self.counts.shape[2]-1)
        elif rotation == "xz":
            if self.roi_plotted is not False:
                self.slider_layer.setMaximum(self.results.shape[1]-1)
            else:
                self.slider_layer.setMaximum(self.counts.shape[1]-1)
        elif rotation == "yz":
            if self.roi_plotted is not False:
                self.slider_layer.setMaximum(self.results.shape[0]-1)
            else:
                self.slider_layer.setMaximum(self.counts.shape[0]-1)
        self.plot_ROI = False
        self.plot_results()

    def spectra_array(self):
        """
        build an array out of a given ordered! dict
        """
        spectra_array = np.copy(self.spectra)
        return spectra_array

    def show_colormap(self):
        """
        shows -only for angle-files- the spectra for every angle as a colormap
        """
        def x_axis(xticks, pos):
            return  np.around(np.add(np.multiply(xticks, self.parameters[1]), self.parameters[0]), decimals =1 )
        def y_axis(yticks, pos):
            try:
                new_yticks = np.around(self.angles[int(abs(yticks))], decimals =4)
            except:
                new_yticks = yticks
            return new_yticks
        if self.one_dim :
            self.log_box.setVisible(True)
            # to set axis labels that are after zooming still correct
            self.figure_ROI.delaxes(self.ax_canvas_roi)
            self.slider_layer.setVisible(False)
            formatter_x = FuncFormatter(x_axis)
            formatter_y = FuncFormatter(y_axis)
            self.entry_position.setText("")
            self.ax_canvas_roi = self.figure_ROI.add_subplot(111)
            self.ax_canvas_roi.set_xlabel("energy [keV]")
            p01_min, p01_max = -0.5, len(self.spectra_array()[0])+0.5
            try:
                p02_min, p02_max = -0.5, len(self.angles)+0.5
            except TypeError:
                p02_min, p02_max = -0.5, 1.5
            extent=[p01_min, p01_max, p02_min, p02_max]
            aspect = (p01_max-p01_min)/(p02_max-p02_min)
            if self.log_box.checkState().value==2:
                positive_spectra = np.copy(self.spectra_array())
                positive_spectra[positive_spectra<0.001]= 1
                self.plot_histo = self.ax_canvas_roi.imshow(positive_spectra,
                                                            norm=LogNorm(vmin=1, ),
                                                            origin="lower",
                                                            extent=extent,
                                                            aspect=aspect)
            else:
                self.plot_histo = self.ax_canvas_roi.imshow(self.spectra_array(), vmin=0, origin="lower", extent=extent,
                                                            aspect=aspect)
            self.ax_canvas_roi.xaxis.set_major_formatter(formatter_x)
            self.ax_canvas_roi.yaxis.set_major_formatter(formatter_y)
            try:
                self.colorbar.remove()
            except:
                pass
            self.colorbar = self.figure_ROI.colorbar(self.plot_histo,
                                                     ax=self.ax_canvas_roi)
        else:
            self.figure_ROI.delaxes(self.ax_canvas_roi)
        self.canvas_roi.draw()

    def retrieve_xy_from_click(self, event):
        """
        This function retrieves the x and y data of the ROI canvas

        Parameters
        ----------
        event : list
            event data from button_press_event or button_release_event

        Returns
        -------
        x, y, z: int
        position in array
        """
        if self.radio_xy.isChecked():
            if self.file_type != ".spx":
                x = int(np.round(event.xdata))
                y = int(np.round(event.ydata))
                z = int(self.slider_layer.value())
            elif self.file_type == ".spx":
                x = self.x[np.abs(self.x-np.round(event.xdata, 3)).argmin()]
                y = self.y[np.abs(self.y-np.round(event.ydata, 3)).argmin()]
                z = self.z[self.slider_layer.value()]
        elif self.radio_xz.isChecked():
            if self.file_type != ".spx":
                x = int(event.ydata)
                y = int(self.slider_layer.value())
                z = int(event.xdata)
            elif self.file_type == ".spx":
                x = self.x[np.abs(self.x-event.xdata).argmin()]
                y = self.y[self.slider_layer.value()]
                z = self.z[np.abs(self.z-event.ydata).argmin()]
        elif self.radio_yz.isChecked():
            if self.file_type != ".spx":
                x = int(self.slider_layer.value())
                y = int(event.ydata)
                z = int(event.xdata)
            elif self.file_type == ".spx":
                x = self.x[self.slider_layer.value()]
                y = self.y[np.abs(self.y-event.xdata).argmin()]
                z = self.z[np.abs(self.z-event.ydata).argmin()]
        return x, y, z

    def plot_single_spectra_on_click(self, event):
        time = t.time()
        size_hor, size_ver = [int(i) for i in self.combo_rect_size.currentText().split("x")]
        if event.dblclick:
            artists = self.ax_canvas_roi.get_children()
            rectangle_type = type(patches.Rectangle((1, 1), 1, 1))
            for artist in artists[:4]:
                if isinstance(artist, rectangle_type):
                    artist.remove()
            x, y, z = self.retrieve_xy_from_click(event)
            if self.radio_xy.isChecked():
                if size_hor != 1:
                    x_rect, y_rect = np.mgrid[np.round(x-(size_hor-1)/2*self.step_x, 3):np.round(x+(size_hor-1)/2*self.step_x+self.step_x, 3):self.step_x,
                                              np.round(y-(size_ver-1)/2*self.step_y, 3):np.round(y+(size_ver-1)/2*self.step_y+self.step_y, 3):self.step_y]
                    z_rect = np.full((size_hor, size_ver), z)
                    if x_rect.shape != z_rect.shape:
                        x_rect = np.resize(x_rect, z_rect.shape)
                        y_rect = np.resize(y_rect, z_rect.shape)
                else:
                    x_rect, y_rect, z_rect = x, y, z
                if self.file_type == ".spx":
                    size_hor *= self.step_y
                    size_ver *= self.step_x
                self.rect = patches.Rectangle((x-size_ver/2, y-size_hor/2),
                                              size_ver, size_hor,
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            elif self.radio_xz.isChecked():
                if size_hor != 1:
                    y_rect = np.full((size_hor, size_ver), y)
                    x_rect, z_rect = np.mgrid[np.round(int(x-(size_hor)/2)+1, 3):np.round(int(x+(size_hor)/2)+1, 3):self.step_x,
                                              np.round(int(z-size_ver/2)+1, 3):np.round(int(z+size_ver/2)+1, 3):self.step_z]
                    if x_rect.shape != y_rect.shape:
                        x_rect = np.resize(x_rect, y_rect.shape)
                        z_rect = np.resize(z_rect, y_rect.shape)
                else:
                    y_rect = y
                    x_rect, z_rect = x, z
                if self.file_type == ".spx":
                    size_hor *= self.step_x
                    size_ver *= self.step_z
                self.rect = patches.Rectangle((x-size_ver/2, z-size_hor/2),
                                              size_ver, size_hor,
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            elif self.radio_yz.isChecked():
                if size_hor != 1:
                    x_rect = np.full((size_hor, size_ver), x)
                    y_rect, z_rect = np.mgrid[np.round(int(y-(size_hor)/2)+1, 3):np.round(int(y+(size_hor)/2)+1, 3):self.step_y,
                                              np.round(int(z-size_ver/2)+1, 3):np.round(int(z+size_ver/2)+1, 3):self.step_z]
                    if y_rect.shape != x_rect.shape:
                        y_rect = np.resize(y_rect, x_rect.shape)
                        z_rect = np.resize(z_rect, x_rect.shape)
                else:
                    z_rect = z
                    x_rect, y_rect = x, y
                if self.file_type == ".spx":
                    size_hor *= self.step_y
                    size_ver *= self.step_z
                self.rect = patches.Rectangle((y-size_ver/2, z-size_hor/2),
                                              size_ver, size_hor,
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            len_X, len_Y, len_Z = self.counts.shape
            if self.file_type == ".spx":
                self.spec_nr = int((z-self.z[0])/self.step_z + (y-self.y[0])/self.step_y * len_Z + (x-self.x[0])/self.step_x * len_Y* len_Z)
                self.spec_rect = (z_rect-self.z[0])/self.step_z + (y_rect-self.y[0])/self.step_y * len_Z + (x_rect-self.x[0])/self.step_x * len_Y * len_Z
                self.spec_rect = np.round(self.spec_rect, 0).astype(np.int32)
            else:
                self.spec_nr = int(y + x * len_Y + z * len_Y * self.len_x)
                self.spec_rect = y_rect + x_rect * len_Y + z_rect * len_Y * self.len_x
            if not isinstance(self.spec_rect, np.ndarray):
                self.spec_rect = int(self.spec_rect)
            else:
                self.spec_rect = self.spec_rect.astype(np.int32)
            if self.load_type == "angle_file":
                pass
            else:
                if self.plot_style_str == "linear":
                    self.plot_style = self.parent.ax_canvas_spectrum.plot
                elif self.plot_style_str == "log":
                    self.plot_style = self.parent.ax_canvas_spectrum.semilogy
                low_index = int((float(self.roi_low.text())-self.parameters[0])/self.parameters[1])
                high_index = int((float(self.roi_high.text())-self.parameters[0])/self.parameters[1])
                if self.load_type == "angle_file":
                    xplot = self.energy#[low_index:high_index]
                    yplot = list(self.spectra.values())[self.spec_nr]#[low_index:high_index]
                    self.plot_style(xplot, yplot)
                    self.entry_position.setText(f"{np.around(self.angles[self.spec_nr], decimals=6)}")
                else:
                    self.rect_sum_spec = self.calc_rect_sum_spec(np.unique(self.spec_rect))
                    self.parent.clear_ax_canvas_spectrum(["measurement"])
                    for line in self.parent.ax_canvas_spectrum.lines:
                        if line.properties()["label"] == "measurement":
                            line.set_xdata(self.energy[low_index:high_index])
                            line.set_ydata(self.rect_sum_spec[low_index:high_index])
                    self.parent.ax_canvas_spectrum.set_xlim(low_index*self.parameters[1]+self.parameters[0], high_index*self.parameters[1]+self.parameters[0])
                    self.parent.ax_canvas_spectrum.set_ylim(1e-5, np.max(self.rect_sum_spec[low_index:high_index])*1.1)
                    self.ax_canvas_roi.add_patch(self.rect)
                    self.canvas_roi.draw()
                    self.entry_position.setText("%d"%np.round(self.spec_nr, 0))
                self.parent.ax_canvas_spectrum.legend()
                self.parent.canvas_spectrum.draw()
                self.button_save_selection.show()
            self.parent.set_spectra_nr(spectra_nr=np.round(self.spec_nr, 0))
            self.parent.roi_widget.spec_nr = np.round(self.spec_nr, 0)
        else:
            self.x0, self.y0, self.z0 = self.retrieve_xy_from_click(event)
        self.time = time

    def draw_rectangle(self, event):
        time = t.time()
        if not event.dblclick and ((time-self.time)>0.5):
            artists = self.ax_canvas_roi.get_children()
            rectangle_type = type(patches.Rectangle((1, 1), 1, 1))
            for artist in artists[:4]:
                if isinstance(artist, rectangle_type):
                    artist.remove()
            x, y, z = self.retrieve_xy_from_click(event)
            x = np.sort([x, self.x0])
            y = np.sort([y, self.y0])
            z = np.sort([z, self.z0])
            if self.radio_xy.isChecked():
                if np.diff(x) != 0:
                    x_rect, y_rect = np.mgrid[x.min():x.max():self.step_x,
                                              y.min():y.max():self.step_y]
                    size_hor, size_ver = x_rect.shape
                    z_rect = np.full((size_hor, size_ver), z[0])
                else:
                    x_rect = x
                    y_rect, z_rect = y, z
                self.rect = patches.Rectangle((x[0]-self.step_x/2, y[0]-self.step_y/2),
                                              np.diff(x), np.diff(y),
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            elif self.radio_xz.isChecked():
                if np.diff(x) != 0:
                    x_rect, z_rect = np.mgrid[x[0]:x[-1]:self.step_x, z[0]:z[-1]:self.step_z]
                    size_hor, size_ver = x_rect.shape
                    y_rect = np.full((size_hor, size_ver), y[0])
                else:
                    y_rect = y
                    x_rect, z_rect = x, z
                self.rect = patches.Rectangle((x[0]-self.step_x/2, z[0]-self.step_z/2),
                                              np.diff(x), np.diff(z),
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            elif self.radio_yz.isChecked():
                if np.diff(y) != 0:
                    y_rect, z_rect = np.mgrid[y[0]:y[-1]:self.step_y, z[0]:z[-1]:self.step_z]
                    size_hor, size_ver = y_rect.shape
                    x_rect = np.full((size_hor, size_ver), x[0])
                else:
                    z_rect = z
                    x_rect, y_rect = x, y
                self.rect = patches.Rectangle((y[0]-self.step_y/2, z[0]-self.step_z/2),
                                              np.diff(y), np.diff(z),
                                              linewidth=1, edgecolor="r",
                                              facecolor="None")
            try:
                self.len_x, len_Y, len_Z = self.counts.shape
                if self.file_type == ".spx":
                    self.spec_nr = int(round((z[0]-self.z[0])/self.step_z)) + \
                                   int(round((y[0]-self.y[0])/self.step_y)) * len_Z + \
                                   int(round((x[0]-self.x[0])/self.step_x)) * len_Y * len_Z
                    self.spec_rect = (z_rect-self.z[0])/self.step_z + (y_rect-self.y[0])/self.step_y * len_Z + (x_rect-self.x[0])/self.step_x * len_Y * len_Z
                    self.spec_rect = np.round(self.spec_rect, 0).astype(np.uint)
                else:
                    self.spec_nr = int(z[0] + y[0] * len_Z + x[0] * len_Y * len_Z)
                    self.spec_rect = z_rect + y_rect * len_Z + x_rect * len_Y * len_Z
            except:
                print("please press \"show ROI\" first")
            self.spec_rect = self.spec_rect.astype(np.int32)
            if size_hor == 1:
                self.spec_nr = int(np.where((self.positions==np.array([x[0], y[0], z[0]])).all(1))[0])
            if self.load_type == "angle_file":
                pass
            else:
                self.parent.ax_canvas_spectrum.clear()
                if self.plot_style_str == "linear":
                    self.plot_style = self.parent.ax_canvas_spectrum.plot
                if self.plot_style_str == "log":
                    self.plot_style = self.parent.ax_canvas_spectrum.semilogy
                self.rect_sum_spec = self.calc_rect_sum_spec(np.unique(self.spec_rect))
                low_index = int((float(self.roi_low.text())-self.parameters[0])/self.parameters[1])
                high_index = int((float(self.roi_high.text())-self.parameters[0])/self.parameters[1])
                self.plot_style(self.energy[low_index:high_index], self.rect_sum_spec[low_index:high_index])
                self.ax_canvas_roi.add_patch(self.rect)
                self.canvas_roi.draw()
            self.parent.canvas_spectrum.draw()
            self.button_save_selection.show()

    def calc_rect_sum_spec(self, keys):
        """
        This function calculates the sum spectrum from a given set of spec_nr
        keys.

        Parameters
        ----------
        keys : list of str
            list of strings which contain the spectrum number

        Returns
        -------
        spectrum : array
            the summation of the selected spectra
        """
        len_keys = len(keys)
        for i, spec_nr in enumerate(keys):
            if isinstance(self.spectra, da.Array):
                spectrum = self.spectra[spec_nr].compute()
            else:
                spectrum = self.spectra[str(spec_nr)]
            if i == 0:
                rect_sum_spec = spectrum.copy()
            else:
                rect_sum_spec += spectrum
        return rect_sum_spec / len_keys

    def save_selection(self):
        """
        This function save the selected spectra as a numpy file into a selected
        path.

        Parameters
        ----------
        keys : list of str
            list of strings which contain the spectrum number
        """
        self.save_selection_path = QtWidgets.QFileDialog().getSaveFileName(self,
                                                                           "select save path",
                                                                           self.parent.data.file_path.replace(
                                                                               self.parent.data.file_type,
                                                                               "_selection.npy"))[0]
        spectra = []
        if isinstance(self.spec_rect, int):
            self.spec_rect = np.asarray([self.spec_rect])
            self.spec_rect = np.expand_dims(self.spec_rect, 1)
        shape = self.spec_rect.shape
        for x, _ in enumerate(self.spec_rect):
            for spec_nr in self.spec_rect[x]:
                spectra.append(self.spectra[spec_nr])
        np.save(self.save_selection_path, np.asarray(spectra).reshape(shape + self.spectra[spec_nr].shape))

    def save_ROI(self):
        """
        This function saves the calculated ROI image into a selected file.
        """
        self.save_roi_path = QtWidgets.QFileDialog().getSaveFileName(self,
                                                                     "select save path",
                                                                     self.parent.data.file_path.replace(
                                                                         self.parent.data.file_type,
                                                                         "_%s_%s_ROI.npy" % (
                                                                             self.element_str, self.line)))[0]
        np.save(self.save_roi_path, self.results)
