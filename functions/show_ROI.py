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
        self.time = 0
        self.elements = elements[5:84]
        self.para_a0 = None  # this is the a0 entry which show_ROI inherits from the SpecFit Main Window
        self.para_a1 = None  # this is the a1 entry which show_ROI inherits from the SpecFit Main Window
        self.roi_low = None  # this is the low ROI entry which show_ROI inherits from the SpecFit Main Window
        self.roi_high = None  # this is the high ROI entry which show_ROI inherits from the SpecFit Main Window
        self.ROI_image = False  # this is to determine if a plot already exists
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
        self.setStyleSheet("QWidget { " \
                        #    +"color: black; background-color:white;" \
                           +"}"\
                        #    +"background-color:white;"\
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
        self.canvas_roi.mpl_connect("button_press_event", self.retrieve_selection)
        self.canvas_roi.mpl_connect("button_release_event", self.retrieve_selection)
        #  self.canvas_roi.mpl_connect("button_press_event", self.create_fixed_bbox)
        # self.canvas_roi.mpl_connect("button_release_event", self.create_drawn_rectangle)
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
        # create a link to the needed data from the parent instance
        self.spectra = self.parent.data.spectra
        self.angles = self.parent.data.positions
        self.positions = np.round(self.parent.data.positions, 3)
        self.tensor_positions = self.parent.data.tensor_positions
        self.parameters = self.parent.data.parameters[0]
        self.counts = self.parent.data.counts
        self.len_x, self.len_y, self.len_z = self.parent.data.position_dimension
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
        for dim in ["x", "y", "z"]:
            setattr(self, f"step_{dim}", np.abs(getattr(self, f"step_{dim}")))
        # display the sum_spec
        self.sum_spec = self.parent.data.sum_spec
        self.delta_E = self.parameters[3]
        self.entry_delta_E.setText(f"{self.delta_E}")
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
        self.results = np.zeros(self.counts.shape)
        self.button_save_ROI.hide()
        self.button_save_selection.hide()
        self.button_reset_ROI.hide()
        # self.ROI_image.remove()
        # self.ROI_image = False
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
        self.results = self.calc_roi_intensity()
        self.results = self.results.reshape(self.counts.shape)
        # if self.file_type in [".MSA", ".msa"]:
        #     self.results = np.flip(self.results, 0)
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
        return self.spectra[..., index_low:index_high].sum(axis=-1).compute()

    def evaluate_energy(self):
        """
        read out the energie from the loaded data
        """
        self.energy = self.parent.data.energies
    
    def slice_array(self, array, axis="xy", layer=0):
        """
        Function to get the layer of the 3D data based on the radio button
        selected and rotate it so that the first axis is at the bottom
        """
        if axis == 'xy':  # layer in z-axis
            return array[:, :, layer]
            # return np.rot90(array[:, :, layer], k=-1)
        elif axis == 'xz':  # layer in y-axis
            return array[:, layer, :]
            # return np.rot90(array[:, layer, :], k=-1)
        elif axis == 'yz':  # layer in x-axis
            return array[layer, :, :]
            # return np.rot90(array[layer, :, :], k=-1)
        
    def remove_rectangle(self, ):
        """
        Function to remove the rectangle shape in the ROI plot
        """
        self.rect.remove()
        del self.rect
        del self.rect_data

    def remove_image(self, ):
        """
        Function to remove the previously drawn image from the ROI plot
        """
        self.ROI_image.remove()

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
            self.canvas_spectrum.draw_idle()

    def plot_line_energy(self):
        self.get_line_energy()
        self.plot_sum_spec()
        self.delta_E = float(self.entry_delta_E.text())
        if not self.first_time_loaded:
            self.ax_canvas_spectrum.axvspan(self.elements[1][0]-self.delta_E,
                                            self.elements[1][0]+self.delta_E,
                                            color="r", alpha=0.3)
            self.canvas_spectrum.draw_idle()

    def plot_results(self):
        self.entry_position.setText("")
        # clear ax_canvas from drawn images
        layer = self.slider_layer.value()
        aspect = "auto"
        line = self.line.replace("Ka", u"K\u03B1").replace("Kb", u"K\u03b2")
        # get the axis to display
        dim1, dim2 = self.rotation
        dim3 = 'xyz'.replace(dim1, '').replace(dim2, '')
        ax1, step_ax1 = getattr(self, dim1), getattr(self, f"step_{dim1}")
        ax2, step_ax2 = getattr(self, dim2), getattr(self, f"step_{dim2}")
        ax3, step_ax3 = getattr(self, dim3), getattr(self, f"step_{dim3}")
        extent = [ax1[-1] + step_ax1 / 2, ax1[0] - step_ax1 / 2,
                  ax2[-1] + step_ax2 / 2, ax2[0] - step_ax2 / 2]
        # if counts are displayed
        if self.roi_plotted is True:
            data = self.results
        else:
            data = self.counts
        maximum = np.percentile(data, self.slider_percentile.value())
        data_layer = self.slice_array(array=data,
                                    axis=self.rotation,
                                    layer=layer)
        if not self.ROI_image:
            self.ROI_image = self.ax_canvas_roi.imshow(data_layer,
                                                    vmin=0,
                                                    vmax=maximum,
                                                    aspect=aspect,
                                                    extent=extent,
                                                    origin="lower")
        else:
            self.ROI_image.set_data(data_layer)

            self.ROI_image.set_extent(extent)
        self.ROI_image.set_clim(0, maximum)
        try: 
            self.colorbar.update_ticks()
        except AttributeError: 
            self.colorbar = self.figure_ROI.colorbar(self.ROI_image,
                                                    ax=self.ax_canvas_roi)
        layer = ax3[layer]
        if self.roi_plotted is True:
            self.ax_canvas_roi.set_title(f"ROI Counts {self.element_str} {line} | {dim3} position : {layer}")                                                                  
        else:
            self.ax_canvas_roi.set_title(f"Counts | {dim3} position : {layer}")
        self.ax_canvas_roi.set_xlabel(dim1)
        self.ax_canvas_roi.set_ylabel(dim2)
        
        self.canvas_roi.draw_idle()

    def rotate_results(self, rotation):
        self.slider_layer.setVisible(True)
        self.log_box.setVisible(False)
        self.rotation = rotation
        dim1, dim2 = self.rotation
        dim3 = 'xyz'.replace(dim1, '').replace(dim2, '')
        idx = ["x", "y", "z"].index(dim3)
        if hasattr(self, "rect_data"):
            self.rotate_rectangle(new_idx=[dim1, dim2, dim3])
        if self.roi_plotted is not False:
            data = self.results
        else:
            data = self.counts
        self.slider_layer.setMaximum(data.shape[idx]-1)
        if self.slider_layer.value() != 0:
            self.slider_layer.setValue(0)
        else:
            self.plot_results()
        
    def rotate_rectangle(self, new_idx):
        """
        Rotate the selection rectangle
        """
        old_idx = [self.rect_data[idx] for idx in [f"dim{i}" for i in range(1,4)]]
        idx_transform = [old_idx.index(idx)+1 for idx in new_idx]
        self.rect.set_width(self.rect_data[f"dim{idx_transform[0]}_size"]*getattr(self, f"step_{new_idx[0]}"))
        self.rect.set_height(self.rect_data[f"dim{idx_transform[1]}_size"]*getattr(self, f"step_{new_idx[1]}"))
        self.rect.set_xy((self.rect_data[f"dim{idx_transform[0]}_pos"], self.rect_data[f"dim{idx_transform[1]}_pos"]))

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
            # p01_min, p01_max = -0.5, len(self.spectra_array()[0])+0.5
            p01_min, p01_max = -0.5, len(self.spectra[0])+0.5
            try:
                p02_min, p02_max = -0.5, len(self.angles)+0.5
            except TypeError:
                p02_min, p02_max = -0.5, 1.5
            extent=[p01_min, p01_max, p02_min, p02_max]
            aspect = (p01_max-p01_min)/(p02_max-p02_min)
            if self.log_box.checkState().value==2:
                # positive_spectra = np.copy(self.spectra_array())
                positive_spectra = self.spectra
                positive_spectra[positive_spectra<0.001]= 1
                self.plot_histo = self.ax_canvas_roi.imshow(positive_spectra,
                                                            norm=LogNorm(vmin=1, ),
                                                            origin="lower",
                                                            extent=extent,
                                                            aspect=aspect)
            else:
                self.plot_histo = self.ax_canvas_roi.imshow(self.spectra, 
                                                            vmin=0, 
                                                            origin="lower", 
                                                            extent=extent,
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
        self.canvas_roi.draw_idle()

    def retrieve_pos_from_click(self, event):
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
        dim1, dim2 = self.rotation
        dim3 = 'xyz'.replace(dim1, '').replace(dim2, '')
        positions = {}
        positions[dim1] = getattr(self, dim1)[np.abs(getattr(self, dim1)- np.round(event.xdata, 3)).argmin()]
        positions[dim2] = getattr(self, dim2)[np.abs(getattr(self, dim2)- np.round(event.ydata, 3)).argmin()]
        positions[dim3] = getattr(self, dim3)[self.slider_layer.value()]
        idx = int(np.where((self.positions == np.round([positions[key] for key in sorted([dim1, dim2, dim3])], 3)).all(1))[0])
        tensor_x, tensor_y, tensor_z = self.tensor_positions[idx]
        return positions[dim1], positions[dim2], positions[dim3], tensor_x, tensor_y, tensor_z, idx

    def retrieve_selection(self, event):
        """
        Function which decides which mouse press event was executed. It calls
        the corresponding function to draw either a drawn rectangle or a
        fixed size rectangle.
        """
        time = t.time()
        if self.time == 0 or ((time-self.time)>10):
            self.time = t.time()
        if event.dblclick:
            self.create_fixed_bbox(event=event)
        elif not event.dblclick and ((time-self.time)>0.5) and event.name == "button_release_event":
            self.create_drawn_rectangle(event=event)
        else:
            self.pos1_0, self.pos2_0, self.pos3_0, self.tx0, self.ty0, self.tz0, self.idx0 = self.retrieve_pos_from_click(event)
        self.time = time

    def create_rectangel(self, event):
        """
        Function to create the rectangle and calculate the sum spectrum over
        the selected spectra
        """

    def draw_rectangle(self, ):
        """
        Function to draw the rectangle into the roi_canvas
        """
        if self.load_type == "angle_file":
            pass
        else:
            if self.plot_style_str == "linear":
                self.plot_style = self.parent.ax_canvas_spectrum.plot
            elif self.plot_style_str == "log":
                self.plot_style = self.parent.ax_canvas_spectrum.semilogy
            try:
                self.remove_rectangle()
            except NotImplementedError:
                pass
            low_index = int((float(self.roi_low.text())-self.parameters[0])/self.parameters[1])
            high_index = int((float(self.roi_high.text())-self.parameters[0])/self.parameters[1])
            
            self.rect_sum_spec = self.calc_rect_sum_spec(np.unique(self.spec_rect))
            self.parent.check_fit()
            self.parent.ax_canvas_spectrum.set_xlim(low_index*self.parameters[1]+self.parameters[0], high_index*self.parameters[1]+self.parameters[0])
            self.parent.ax_canvas_spectrum.set_ylim(1e-5, np.max(self.rect_sum_spec[low_index:high_index])*1.1)
            self.ax_canvas_roi.add_patch(self.rect)
            self.canvas_roi.draw_idle()
            self.entry_position.setText(f"{self.spec_nr:.0f}")
            self.parent.ax_canvas_spectrum.legend()
            self.parent.canvas_spectrum.draw_idle()
            self.button_save_selection.show()
        self.parent.set_spectrum_nr(spectrum_nr=np.round(self.spec_nr, 0))
        self.parent.roi_widget.spec_nr = np.round(self.spec_nr, 0)
        self.time = 0

    def create_fixed_bbox(self, event):
        size_hor, size_ver = [int(i) for i in self.combo_rect_size.currentText().split("x")]
        pos1, pos2, pos3, tx, ty, tz, idx = self.retrieve_pos_from_click(event)
        # create the rectangle
        dim1, dim2 = self.rotation
        dim3 = 'xyz'.replace(dim1, '').replace(dim2, '')
        sorted_dims_idx = [[dim1, dim2, dim3].index(key) for key in sorted([dim1, dim2, dim3])]
        step_dim1 = getattr(self, f"step_{dim1}")
        step_dim2 = getattr(self, f"step_{dim2}") 
        size_hor, size_ver = [int(i) for i in self.combo_rect_size.currentText().split("x")]
        size_hor *= step_dim1
        size_ver *= step_dim2
        if size_hor != step_dim1:
            dim1_rect, dim2_rect = np.mgrid[np.round(pos1-(size_hor-step_dim1)/2, 3):np.round(pos1+(size_hor-step_dim1)/2+step_dim1, 3):step_dim1,
                                            np.round(pos2-(size_ver-step_dim2)/2, 3):np.round(pos2+(size_ver-step_dim2)/2+step_dim2, 3):step_dim2]
            dim3_rect = np.full((int(size_hor//step_dim1), int(size_ver//step_dim1)), pos3)
            if dim1_rect.shape != dim3_rect.shape:
                dim1_rect = np.resize(dim1_rect, dim3_rect.shape)
                dim2_rect = np.resize(dim2_rect, dim3_rect.shape)
        else:
            dim1_rect, dim2_rect, dim3_rect = pos1, pos2, pos3
        self.rect = patches.Rectangle(xy=(pos1-size_ver/2, pos2-size_hor/2),
                                        width=size_ver, height=size_hor,
                                        linewidth=1, edgecolor="r",
                                        facecolor="None")
        self.rect_data = {"dim1": dim1, "dim2": dim2, "dim3": dim3, 
                          "dim1_pos": pos1, "dim2_pos": pos2, "dim3_pos": pos3,
                          "dim1_size": size_hor, "dim2_size": size_ver, "dim3_size": getattr(self, f"step_{dim3}") }
        positions_rect = np.asarray([dim1_rect.flatten(), dim2_rect.flatten(), dim3_rect.flatten()]).T
        # find the position indices for the selected rectangle 
        indices = np.zeros(shape=len(positions_rect))
        for i, row in enumerate(positions_rect):
            indices[i] = np.where((self.positions == np.round(row[sorted_dims_idx], 3)).all(1))[0][0]
            
        self.spec_nr = idx
        self.spec_rect = indices.reshape(dim1_rect.shape).astype(np.int32)
        self.draw_rectangle()
        
    def create_drawn_rectangle(self, event):
        if "pos1_0" not in dir(self):
            self.pos1_0, self.pos2_0, self.pos3_0, self.tx0, self.ty0, self.tz0, self.idx0 = self.retrieve_pos_from_click(event)
        # check if a rectangle is already displayed and remove it
        artists = self.ax_canvas_roi.get_children()
        rectangle_type = type(patches.Rectangle((1, 1), 1, 1))
        for artist in artists[:4]:
            if isinstance(artist, rectangle_type):
                artist.remove()
        # get the current position of the click event
        pos1, pos2, pos3, tx, ty, tz, idx = self.retrieve_pos_from_click(event)
        # create the rectangle
        dim1, dim2 = self.rotation
        dim3 = 'xyz'.replace(dim1, '').replace(dim2, '')
        step_dim1 = getattr(self, f"step_{dim1}")
        step_dim2 = getattr(self, f"step_{dim2}") 
        step_dim3 = getattr(self, f"step_{dim3}") 
        ax1 = np.sort([pos1, self.pos1_0])
        ax2 = np.sort([pos2, self.pos2_0])
        ax3 = np.sort([pos3, self.pos3_0])
        
        if np.diff(ax1) != 0:
            np.mgrid[ax1.min():ax1.max():step_dim1,
                        ax2.min():ax2.max():step_dim2]
            dim1_rect, dim2_rect = np.mgrid[ax1.min():ax1.max():step_dim1,
                                            ax2.min():ax2.max():step_dim2]
            
            size_hor, size_ver = dim1_rect.shape
            dim3_rect = np.full((size_hor, size_ver), ax3[0])
        else:
            dim1_rect = ax1
            dim2_rect, dim3_rect = ax2, ax3
            size_hor = dim1_rect.shape
        self.rect = patches.Rectangle(xy=(float(ax1[0]-step_dim1/2), float(ax2[0]-step_dim2/2)),
                                        width=float(np.diff(ax1)), height=float(np.diff(ax2)),
                                        linewidth=1, edgecolor="r",
                                        facecolor="None")
        self.rect_data = {"dim1": dim1, "dim2": dim2, "dim3": dim3, 
                          "dim1_pos": pos1, "dim2_pos": pos2, "dim3_pos": pos3,
                          "dim1_size": size_hor, "dim2_size": size_ver, "dim3_size": getattr(self, f"step_{dim3}") }
        
        positions_rect = np.asarray([dim1_rect.flatten(), dim2_rect.flatten(), dim3_rect.flatten()]).T
        # find the position indices for the selected rectangle 
        indices = np.zeros(shape=len(positions_rect))
        for i, row in enumerate(positions_rect):
            indices[i] = np.where((self.positions == np.round(row, 3)).all(1))[0][0]
            
        self.spec_nr = idx
        self.spec_rect = indices.reshape(dim1_rect.shape).astype(np.int32)

        if size_hor == 1:
            self.spec_nr = idx
        self.draw_rectangle()
        
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
                spectrum = self.spectra[spec_nr]
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
