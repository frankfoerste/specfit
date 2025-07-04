import os
from PyQt6 import QtGui, QtCore, QtWidgets
import matplotlib.backends.backend_qt5agg as pltqt
import matplotlib.figure as figure
import numpy as np
import natsort as ns
import specfit_GUI_functions as sfunc

class FitThresholdPopup(QtWidgets.QWidget):
    popup_threshold = QtCore.pyqtSignal(object)
    def __init__(self, folder_path):
        super(FitThresholdPopup, self).__init__()
        self.folder_path = folder_path
        self.threshold = 0
        self.spec = {}
        self.roi_indicees = (0,1)
        self.loadtype = ""
        self.use_spectra = False
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.popup_heigth = 500
        self.popup_width = 520
        self.setWindowTitle("Fit Treshold")
        self.setGeometry((self.screen_width-self.popup_width)//2,
                         (self.screen_height-self.popup_heigth)//2,
                         self.popup_width, self.popup_heigth)
        ### define label and entries ###
        self.label_mincount = QtWidgets.QLabel("mincounts", self)
        self.label_mincount.setGeometry(5, 460, 80, 20)
        self.entry_mincounts = QtWidgets.QLineEdit("0", self)
        self.entry_mincounts.setGeometry(90, 460, 80, 20)
        self.__init__plot()

    def __init__plot(self):
        """
        define the layout of the plot frame
        """
        self.figure_matplot = figure.Figure(dpi = 80)
        self.canvas_matplot= pltqt.FigureCanvasQTAgg(self.figure_matplot)
        self.canvas_matplot.setParent(self)
        self.canvas_matplot.move(5,0)
        self.toolbar_matplot = pltqt.NavigationToolbar2QT(self.canvas_matplot, self)
        self.toolbar_matplot.setStyleSheet("color: black; background-color:DeepSkyBlue; border: 1px solid #000")
        self.toolbar_matplot.setGeometry(0,400, 510, 50)

    def show_frp(self):
        self.show()

    def define_counts(self):
        if self.loadtype == "folder":  # other loadtypes get the spectra from specfit_main
            dictpath = os.path.join(self.folder_path,os.path.join("data","spectra.pickle"))
            print(self.folder_path)
            if os.path.exists(dictpath):
                self.spec = sfunc.open_dict_pickle(dictpath)
                self.use_spectra = True
            else:
                print("use_counts") # TODO
                counts = np.load( os.path.join(self.folder_path,os.path.join("data","counts.npy")))
                sum_counts = counts
                self.use_spectra = False
        if self.use_spectra:
            sum_counts = []
            sorted_keys = ns.natsorted(self.spec.keys())
            for key in sorted_keys:
                sum_counts.append(np.sum(self.spec[key][self.roi_indicees[0]:self.roi_indicees[1]]))
        return sum_counts

    def plot_counts(self):
        sum_counts = self.define_counts()
        if self.loadtype in ["angle_file"]:
            ylabel = "Counts in ROI"
        if self.loadtype in ["msa_file", "file", "folder"]:
            if os.path.exists( os.path.join(self.folder_path,os.path.join("data","spectra.pickle"))):
                ylabel = "Counts per second in ROI"
            else:
                ylabel = "Counts per second"
        self.ax_canvas_matplot = self.figure_matplot.add_subplot(111)
        self.ax_canvas_matplot.set_xlabel("Measurement")
        self.ax_canvas_matplot.set_ylabel(ylabel)
        if len(sum_counts)>1:self.ax_canvas_matplot.plot(sum_counts)
        else: self.ax_canvas_matplot.plot(sum_counts, marker = "*")
