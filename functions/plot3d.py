import os
import sys
import time
import numpy as np
from functools import partial
from PyQt6 import QtWidgets, QtGui, QtCore
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.backends.backend_qt5agg as pltqt
import matplotlib.figure as figure
from skimage import measure
from scipy.ndimage import zoom

rng = np.random.default_rng(42)

class Plot3D(QtWidgets.QWidget):
    """
    Class for the 3D plotter UI
    """
    def __init__(self):
        super(Plot3D, self).__init__()
        screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = screen_properties.width()
        self.screen_height = screen_properties.height()
        if sys.platform == "linux" or sys.platform == "linux2":
            self.homedir = os.environ["HOME"]
        else: self.homedir = "C:/"
        self.version = "plot3d 1.00"
        try:
            QtGui.QIcon.setThemeName("ubuntu-mono-dark")
        except:
            pass
        self.window_heigth = 450
        self.window_width = 685
        self.setWindowTitle(self.version)
        self.setGeometry((self.screen_width-self.window_width)//2,
                         (self.screen_height-self.window_heigth)//2,
                         self.window_width, self.window_heigth)
        # initialization of required parameter
        self.compression = 1
        self.plot_types = ["voxel", "surface"]
        self.measurements = {}
        self.measurements_zero_border = {}
        self.thresholds = {}
        self.face_colors = {}
        self.plots = {}
        self.time_button = time.time()
        self.predefined_colors = ["#ff0000", "#0000ff", "#00ff00", "#00ffff",
                                  "#ff00ff", "#ffff00", "#3300ff", "#0033ff",
                                  "#ff3300", "#ff0033", "#33ff00", "#00ff33",
                                  "#ff0099", "#9900ff", "#0099ff", "#ff9900",
                                  "#ff0099", "#99ff00", "#00ff99", "#ff3399",
                                  "#ff9933", "#33ff99", "#99ff33", "#3399ff",
                                  "#9933ff", "#3333ff", "#33ff33", "#3333ff",
                                  "#9999ff", "#99ff99", "#99999ff"]
        self.setStyleSheet("QWidget { color: black; background-color:white;}"\
                           +"background-color:white;"\
                           +"border-width: 10px;"\
                           +"QLabel {font-size: 11px;} "\
                           +"QLineEdit {font-size: 11px; max-height: 18px;} "\
                           +"QCheckBox {font-size: 11px; max-height: 18px;} "\
                           +"QPushButton {font-size: 11px; max-height: 18px;}"\
                           +"QRadioButton {font-size: 11px; max-height: 18px;} "\
                           +"QComboBox {font-size: 11px; max-height: 18px;} "\
                           +"QTextEdit {font-size: 11px}")
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.azim = 30
        self.elev = 30
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.zlim = (0.0, 1.0)
        self.time = time.time()
        # initialising the GUI
        self.__init__GUI()
        self.__init__plot()
        self.__init__layout()

    def __init__GUI(self):
        # define widget use to hold button, label, etc. widgets and be placed
        # in Splitter
        self.load_widget = QtWidgets.QWidget()
        self.parameter_widget = QtWidgets.QWidget()
        self.plot_widget = QtWidgets.QWidget()
        self.layout_load_widget = QtWidgets.QGridLayout()
        self.layout_parameter_widget = QtWidgets.QGridLayout()
        self.layout_plot_widget = QtWidgets.QGridLayout()
        self.load_widget.setLayout(self.layout_load_widget)
        self.parameter_widget.setLayout(self.layout_parameter_widget)
        self.plot_widget.setLayout(self.layout_plot_widget)
        # define Statusbar
        self.statusBar = QtWidgets.QLabel("", self)
        self.layout_plot_widget.addWidget(self.statusBar, 0, 2, 1, 1)
        # define labels
        self.label_stepsize = QtWidgets.QLabel("compression", self)
        # define buttons
        self.button_load_measurement = QtWidgets.QPushButton("load measurement", self)
        self.button_load_measurement.clicked.connect(self.load_measurement)
        self.button_load_measurement.setStyleSheet("QWidget {background-color:lightblue}")
        self.button_clear_all = QtWidgets.QPushButton("clear all", self)
        self.button_clear_all.clicked.connect(self.clear_all)
        self.button_clear_all.setStyleSheet("QWidget {background-color:red}")
        self.button_clear_all.hide()
        self.button_display_data = QtWidgets.QPushButton("display confocal data", self)
        self.button_display_data.clicked.connect(self.plot)
        self.button_display_data.setStyleSheet("QWidget {background-color:lightblue}")
        self.button_display_data.hide()
        self.layout_load_widget.addWidget(self.button_load_measurement, 0, 0, 1, 1)
        self.layout_load_widget.addWidget(self.button_clear_all, 0, 2, 1, 1)
        self.layout_parameter_widget.addWidget(self.button_display_data, 20, 0, 1, 2)
        # define spinpox
        self.spinbox_compression = QtWidgets.QDoubleSpinBox(self)
        self.spinbox_compression.setRange(1, 10)
        self.spinbox_compression.setDecimals(1)
        self.spinbox_compression.setSingleStep(0.1)
        self.layout_parameter_widget.addWidget(self.spinbox_compression, 2, 2, 1, 2)
        self.layout_parameter_widget.addWidget(self.label_stepsize, 2, 0, 1, 2)
        # define combos
        self.combo_plot_type = QtWidgets.QComboBox(self)
        self.combo_plot_type.addItems(self.plot_types)
        self.combo_plot_type.setStyleSheet("QWidget {background-color:lightgreen}")
        self.layout_load_widget.addWidget(self.combo_plot_type, 0, 1, 1, 1)

    def __init__plot(self):
        """
        define the layout of the plot frame
        """
        self.figure_3d_confocal = figure.Figure(dpi = 80)
        self.canvas_3d_confocal = pltqt.FigureCanvasQTAgg(self.figure_3d_confocal)
        self.canvas_3d_confocal.setParent(self)
        self.toolbar_3d_confocal = pltqt.NavigationToolbar2QT(self.canvas_3d_confocal, self)
        self.toolbar_3d_confocal.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        # establish connections with User action
        self.canvas_3d_confocal.mpl_connect("button_release_event", self.store_azim_elev)
        # create an axis
        self.ax_canvas_3d_confocal = self.figure_3d_confocal.add_subplot(111, projection="3d")
        self.ax_canvas_3d_confocal.grid(False)
        self.ax_canvas_3d_confocal.set_xlim(self.xlim)
        self.ax_canvas_3d_confocal.set_ylim(self.ylim)
        self.ax_canvas_3d_confocal.set_zlim(self.zlim)
        self.ax_canvas_3d_confocal.xaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.yaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.zaxis.set_pane_color((1, 1, 1, 1))
        # histogram figure
        self.figure_histogram = figure.Figure(dpi = 80)
        self.canvas_histogram = pltqt.FigureCanvasQTAgg(self.figure_histogram)
        self.canvas_histogram.setParent(self)
        self.canvas_histogram.setMaximumHeight(200)
        self.toolbar_histogram = pltqt.NavigationToolbar2QT(self.canvas_histogram, self)
        self.toolbar_histogram.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        # establish connections with User action
        # create an axis
        self.ax_canvas_histogram = self.figure_histogram.add_subplot(111)
        self.layout_plot_widget.addWidget(self.canvas_3d_confocal, 0, 0, 1, 1)
        self.layout_plot_widget.addWidget(self.toolbar_3d_confocal, 1, 0, 1, 1)
        self.layout_parameter_widget.addWidget(self.canvas_histogram, 0, 0, 1, 4)
        self.layout_parameter_widget.addWidget(self.toolbar_histogram, 1, 0, 1, 4)

    def __init__layout(self, ):
        # left splitter
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_splitter.setFrameStyle(0)
        self.left_splitter.addWidget(self.load_widget)
        self.left_splitter.addWidget(self.plot_widget)
        # right splitter
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.right_splitter.setMaximumWidth(200)
        self.right_splitter.setFrameStyle(0)
        self.right_splitter.setStretchFactor(0, 0)
        self.right_splitter.addWidget(self.parameter_widget)
        # central splitter
        self.plot_3d_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.plot_3d_splitter.setMaximumSize(self.screen_width, self.screen_height)
        self.plot_3d_splitter.addWidget(self.left_splitter)
        self.plot_3d_splitter.addWidget(self.right_splitter)
        self.layout.addWidget(self.plot_3d_splitter, 0, 0, 1, 1)

    def show_plot3d(self):
        self.show()
        self.activateWindow()

    def load_measurement(self):
        """
        This function loads the measurement data
        """
        file_path = QtWidgets.QFileDialog(self).getOpenFileNames(filter="(*.npy)")
        # try to open given file, except return (1, 1, 1) array
        for file in file_path[0]:
            try: measurement = np.load(file)
            except:
                print("no .npy-file!, check input")
                measurement = np.zeros((1, 1, 1))
                return
            # the file name should have the format element_line_furtherinformation.npy
            name = file.split("/")[-1].split("_")
            # retrieve the element and line
            try: name = name[0]+"_"+name[1]
            except:
                name = name[0]
                print(f"name not in standard format, please check\n name:\t {name}")
            # definean identification number of the loaded data
            nr = len(self.measurements)
            # define a threshold for plotting purpose, 15 % of unique
            # net peak intensity values
            threshold = np.percentile(np.unique(np.round(measurement, 1)), 20)
            # adding a border to the loaded data
            measurement_zero_border = self.add_zero_border(measurement)
            # file the dict containing the data
            self.measurements.update({nr:measurement})
            self.measurements_zero_border.update({nr:measurement_zero_border})
            shape = measurement.shape
            self.xlim = (0.0, shape[0])
            self.ylim = (0.0, shape[1])
            self.zlim = (0.0, shape[2])
            # set color to measurement from predefined colors, if more then 6 measurements are loaded
            # a random color is defined
            if nr > (len(self.predefined_colors)-1):
                self.face_colors.update({nr: list(rng.sample(3))})
            else:
                self.face_colors.update({nr: self.predefined_colors[nr]})
            # add the threshold values and the corresponding widgets to the treshold dict
            self.thresholds.update({nr:[QtWidgets.QPushButton(name, self),
                                        QtWidgets.QPushButton("", self),
                                        QtWidgets.QLineEdit(str(np.round(threshold, 2)), self),
                                        name,
                                        threshold,
                                        QtWidgets.QCheckBox("", self)]})
            # connect element button with click to histogram plot
            self.thresholds[nr][0].clicked.connect(partial(self.plot_histogram, nr))
            # connect element button with right click to delete the loaded
            # measurement
            self.thresholds[nr][0].setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.thresholds[nr][0].customContextMenuRequested.connect(partial(self.button_right_click, nr))
            # if threshold is changed in lineedit, plot hitogram with adjusted
            # display of threshold
            self.thresholds[nr][2].returnPressed.connect(partial(self.plot_histogram, nr))
            self.layout_parameter_widget.addWidget(self.thresholds[nr][0], nr+3, 0, 1, 1)
            self.layout_parameter_widget.addWidget(self.thresholds[nr][1], nr+3, 1, 1, 1)
            self.layout_parameter_widget.addWidget(self.thresholds[nr][2], nr+3, 2, 1, 1)
            self.layout_parameter_widget.addWidget(self.thresholds[nr][5], nr+3, 3, 1, 1)
            bins = np.arange(0, np.max(measurement), 0.1)
            histogram = np.histogram(measurement, bins = bins)
            self.thresholds["%d_histogram"%nr] = histogram
            self.thresholds[nr][5].setChecked(True)
            self.thresholds[nr][1].setStyleSheet("QPushButton {background-color : %s}"%(self.face_colors[nr]))
            self.thresholds[nr][0].show()
            self.thresholds[nr][1].show()
            self.thresholds[nr][2].show()
            self.thresholds[nr][5].show()
            # shift the plot button corresponding to the number of loaded measurements
            self.button_display_data.show()

    def add_zero_border(self, array):
        """
        This function adds a zero border to the array for better plot results
        """
        add = np.zeros((array.shape[0], 1, array.shape[2]))
        array = np.hstack((np.hstack((add, array)), add))
        add = np.zeros((array.shape[0], array.shape[1], 1))
        array = np.dstack((np.dstack((add, array)), add))
        add = np.zeros((1, array.shape[1], array.shape[2]))
        array = np.vstack((np.vstack((add, array)), add))
        return array

    def resize_array(self, array, compression_factor):
        """ This function shrinks the given array by a factor """
        zoom_value = tuple([(1/compression_factor) if i != 1 else 1 for i in array.shape ])
        array = zoom(array, zoom_value, order = 5)
        return array

    def button_right_click(self, nr, event):
        """
        remove the selected measurement

        Parameters
        ----------
        nr : int
            number of loaded measurement in dictionary
        Returns
        -------
        None
        """
        menu = QtWidgets.QMenu(self)
        remove_measurement_action = menu.addAction("remove")
        action = menu.exec_(self.thresholds[nr][0].mapToGlobal(event))
        if len(self.measurements) == 1:
            self.clear_all()
            return
        if action == remove_measurement_action:
            self.thresholds[nr][0].hide()
            self.thresholds[nr][1].hide()
            self.thresholds[nr][2].hide()
            self.thresholds[nr][5].hide()
            self.thresholds.pop(nr)
            self.measurements.pop(nr)
            self.face_colors.pop(nr)
            self.measurements_zero_border.pop(nr)

    def clear_all(self):
        self.measurements = {}
        self.measurements_zero_border = {}
        for nr in self.thresholds:
            if type(nr) != str:
                self.thresholds[nr][0].hide()
                self.thresholds[nr][1].hide()
                self.thresholds[nr][2].hide()
                self.thresholds[nr][5].hide()
        self.thresholds = {}
        self.face_colors = {}
        self.plots = {}
        self.button_display_data.hide()
        self.figure_3d_confocal.delaxes(self.ax_canvas_3d_confocal)
        self.ax_canvas_3d_confocal = self.figure_3d_confocal.add_subplot(111, projection="3d")
        self.ax_canvas_3d_confocal.grid(False)
        self.ax_canvas_3d_confocal.xaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.yaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.zaxis.set_pane_color((1, 1, 1, 1))
        self.canvas_3d_confocal.draw()

    def clear_plot(self, ):
        self.figure_3d_confocal.delaxes(self.ax_canvas_3d_confocal)
        self.ax_canvas_3d_confocal = self.figure_3d_confocal.add_subplot(111, projection="3d")
        self.ax_canvas_3d_confocal.grid(False)
        self.ax_canvas_3d_confocal.xaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.yaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.zaxis.set_pane_color((1, 1, 1, 1))
        self.canvas_3d_confocal.draw()

    def color_definition(self, nr):
        array = self.measurements[nr]
        threshold = float(self.thresholds[nr][2].text())
        if threshold > np.max(array):
            threshold = 0.8*np.max(array)
            self.thresholds[nr][2].setText(str(round(threshold, 3)))
        face_color = self.face_colors[nr]
        self.thresholds[nr][1].setStyleSheet("QPushButton {background-color : %s}"%(face_color))
        try:
            self.thresholds[nr][1].clicked.disconnect()
        except:
            pass
        self.thresholds[nr][1].clicked.connect(partial(self.color_selection, nr))

    def color_selection(self, measurement_nr):
        color = QtWidgets.QColorDialog.getColor()
        self.thresholds[measurement_nr][1].setStyleSheet("QPushButton {background-color : %s}"%color.name())
        self.refresh_plot(measurement_nr, color.name())

    def plot(self):
        self.button_clear_all.show()
        # clear plot
        self.figure_3d_confocal.delaxes(self.ax_canvas_3d_confocal)
        self.ax_canvas_3d_confocal = self.figure_3d_confocal.add_subplot(111, projection="3d")
        self.ax_canvas_3d_confocal.grid(False)
        # decide between surface or voxel
        if self.combo_plot_type.currentText() == "voxel":
            self.plot_voxel()
        else:
            self.init_plot_surface()

    def plot_histogram(self, nr):
        self.ax_canvas_histogram.clear()
        self.thresholds[nr][4] = float(self.thresholds[nr][2].text())
        self.ax_canvas_histogram.set_title("Histogram of %s"%self.thresholds[nr][0].text(), fontsize = 10)
        self.ax_canvas_histogram.set_xlabel("Net Fluorescence / cps", fontsize = 8)
        self.ax_canvas_histogram.set_ylabel("# of occurence", fontsize = 8)
        occ, bins = self.thresholds["%d_histogram"%nr]
        self.ax_canvas_histogram.vlines(self.thresholds[nr][4], 0, np.max(occ),
                                       colors = "red", linestyles = "dashdot",
                                       lw = 0.5)
        self.ax_canvas_histogram.bar(bins[:-1], occ, color = self.face_colors[nr], log = True)
        self.figure_histogram.tight_layout()
        self.canvas_histogram.draw()

    def refresh_plot(self, measurement_nr, color):
        if self.thresholds[measurement_nr][4] != float(self.thresholds[measurement_nr][2].text()):
            self.thresholds[measurement_nr][4] = float(self.thresholds[measurement_nr][2].text())
        self.face_colors[measurement_nr] = color
        self.plot()

    def store_azim_elev(self, event):
        self.azim = self.ax_canvas_3d_confocal.azim
        self.elev = self.ax_canvas_3d_confocal.elev
        self.xlim = self.ax_canvas_3d_confocal.get_xlim3d()
        self.ylim = self.ax_canvas_3d_confocal.get_ylim3d()
        self.zlim = self.ax_canvas_3d_confocal.get_zlim3d()

    def norm_and_bool_array(self, array, threshold):
        array[array<threshold] = 0
        array_normed = (array / np.max(array))
        array_bool = np.greater(array_normed, 0).astype(bool)
        return array_normed, array_bool

    def cuboid_data(self, o, size=(1, 1, 1)):
        """
        This function creates the form of the cubes
        """
        X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
             [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 1]],
             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(o)
        return X

    def plotCubeAt(self, positions, sizes=None, colors=None, **kwargs):
        if not isinstance(colors, (list, np.ndarray)):
            colors = ["C0"]*len(positions)
        if not isinstance(sizes, (list, np.ndarray)):
            sizes = [(1, 1, 1)]*len(positions)
        g = []
        for p, s, c in zip(positions, sizes, colors):
            g.append(self.cuboid_data(p, size=s) )
        return Poly3DCollection(np.concatenate(g),
                                facecolors=np.repeat(colors, 6, axis=0), **kwargs)

    def plot_voxel(self):
        # read out the data compression from the spinbox
        self.compression = self.spinbox_compression.value()
        # add a progressbar to visualize progress
        progress_bar = QtWidgets.QProgressBar(self)
        self.layout_parameter_widget.addWidget(progress_bar, 21, 0, 1, 4)
        progress_bar.show()
        self.ax_canvas_3d_confocal.clear()
        try: del self.voxels, self.voxel_bool
        except: pass
        # create iterator to check if self.voxels was already created
        i = 0
        # now iterate over every measurement and plot
        for nr, measurement in self.measurements.items():
            # update progress in progress bar
            progress_bar.setValue(nr/len(self.measurements)*100)
            if self.thresholds[nr][5].checkState().value != 2:
                continue
            if self.compression != 1:
                measurement = self.resize_array(measurement, self.compression)
            # read out the threshold of the measurement
            threshold = float(self.thresholds[nr][2].text())
            self.thresholds[nr][4] = threshold
            # create an indices array
            x, y, z = np.indices(measurement.shape)
            # now norm measurement to rgb 1 and create bool array of measurement
            measurement_normed, measurement_bool = self.norm_and_bool_array(measurement, threshold)
            positions = np.c_[x[measurement_bool==1], y[measurement_bool==1], z[measurement_bool==1]]
            # read out color of measurement
            color = self.face_colors[nr]
            # if the first iteration, create the voxels and the voxel_bool array
            if i == 0:
                self.voxel_bool = np.copy(measurement_bool)
                self.voxels = np.zeros(measurement_bool.shape, dtype = object)
            else:
                self.voxel_bool += measurement_bool
            self.color_definition(nr)
            for position in positions:
                x, y, z = position
                alpha_tmp = measurement_normed[x, y, z]
                rgb = np.array([int(color[k:k+2], 16)/255 for k in range(1, len(color)-1, 2)])
                rgba = np.append(rgb, alpha_tmp)
                if np.sum(self.voxels[x, y, z]) == 0:
                    self.voxels[x, y, z] = rgba
                else:
                    # if it is not the initial color, add both rgba values,
                    # norm them by themselves and divide them by 2
                    rgba_old = self.voxels[x, y, z]
                    norm_factor = (1-rgba_old[3])*rgba[3]+rgba_old[3]
                    rgb_new = ((1-rgba_old[3])*rgba[3]*rgba[:3]+rgba_old[3]*rgba_old[:3]) / norm_factor
                    rgba_new = np.append(rgb_new, norm_factor)
                    self.voxels[x, y, z] = rgba_new
            i += 1
        self.ax_canvas_3d_confocal.voxels(self.voxel_bool,
                                          facecolors = self.voxels,
                                          shade = False, )
        progress_bar.hide()
        self.ax_canvas_3d_confocal.set_xlim(self.xlim)
        self.ax_canvas_3d_confocal.set_ylim(self.ylim)
        self.ax_canvas_3d_confocal.set_zlim(self.zlim)
        self.ax_canvas_3d_confocal.grid(False)
        self.ax_canvas_3d_confocal.view_init(self.elev, self.azim)
        self.ax_canvas_3d_confocal.xaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.yaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.zaxis.set_pane_color((1, 1, 1, 1))
        self.figure_3d_confocal.tight_layout()
        self.canvas_3d_confocal.draw()

    def make_mesh(self, image, threshold=-300, step_size=1):
        verts, faces, norm, val = measure.marching_cubes_lewiner(image, threshold, step_size=step_size, allow_degenerate=True)
        return verts, faces

    def plot_surface(self, measurement_nr, face_color = [0.52, 0.52, 0.52]):
        self.ax_canvas_3d_confocal.set_xlim(self.xlim)
        self.ax_canvas_3d_confocal.set_ylim(self.ylim)
        self.ax_canvas_3d_confocal.set_zlim(self.zlim)
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        self.plots[measurement_nr][2] = Poly3DCollection(self.plots[measurement_nr][0][self.plots[measurement_nr][1]], linewidths=0.00, alpha=0.7)
        self.plots[measurement_nr][2].set_facecolor(self.face_colors[measurement_nr])
        self.plots[measurement_nr][3] = self.ax_canvas_3d_confocal.add_collection3d(self.plots[measurement_nr][2])

    def init_plot_surface(self):
        self.statusBar.setText("plotting in progress")
        QtWidgets.QApplication.processEvents()
        # read out of entry data
        self.compression = int(np.round(self.spinbox_compression.value()))
        self.spinbox_compression.setValue(self.compression)
        progress_bar = QtWidgets.QProgressBar(self)
        self.layout_parameter_widget.addWidget(progress_bar, 21, 0, 1, 4)
        progress_bar.show()
        for i, _ in enumerate(self.measurements):
            progress_bar.setValue(i/len(self.measurements)*100)
            self.color_definition(i)
            self.thresholds[i][1].clicked.connect(partial(self.color_selection, i))
            v, f = self.make_mesh(self.measurements[i],
                                  threshold = float(self.thresholds[i][2].text()),
                                  step_size = self.compression)
            self.plots[i] = [v, f, None, None] # [verts, faces, plot_array, plot]
            self.plot_surface(i)
        progress_bar.hide()
        self.ax_canvas_3d_confocal.grid(False)
        self.ax_canvas_3d_confocal.view_init(self.elev, self.azim)
        self.figure_3d_confocal.tight_layout()
        self.ax_canvas_3d_confocal.xaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.yaxis.set_pane_color((1, 1, 1, 1))
        self.ax_canvas_3d_confocal.zaxis.set_pane_color((1, 1, 1, 1))
        self.canvas_3d_confocal.draw()
        self.statusBar.setText("")

def main():
    app = QtWidgets.QApplication(sys.argv)
    abs_correction = Plot3D()
    abs_correction.show()
    app.exec_()

if __name__ == "__main__":
    main()
