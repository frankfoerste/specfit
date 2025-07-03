import os
import time
from PyQt6 import QtGui, QtWidgets, QtCore
import numpy as np
import matplotlib.colors as col
from functools import partial

class SpecFitFitSettings(QtWidgets.QWidget):
    def __init__(self, parent = None, _parent = None):
        # initalize main window
        super(SpecFitFitSettings, self).__init__(parent)
        self.parent = _parent
        self.__version__ = u"Fit Settings - 1.0"
        self.bg_color = 'white'   
        self.specfit_fit_settings_layout = QtWidgets.QGridLayout()
        self.specfit_fit_settings_layout.setSpacing(0)
        self.setLayout(self.specfit_fit_settings_layout)
        self.row_height = 18
        self.button_width = 80
        self.label_width = 60
        # define geometry and properties
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.window_heigth = 120
        self.window_width = 400
        self.setWindowTitle(self.__version__)
        self.setGeometry((self.screen_width-self.window_width)//2,
                         (self.screen_height-self.window_heigth)//2,
                         self.window_width, self.window_heigth)
        self.standard_style_sheet = 'QWidget {color: black; background-color:white; font-size: 10px} '\
                           +'QLineEdit {max-height: 18px; border: 1px solid grey; border-radius: 6px; padding: 0 4px;} '\
                           +'QLineEdit:focus {max-height: 18px; border: 1px solid red; border-radius: 6px; padding: 0 4px;} '\
                           +'QCheckBox {max-height: 18px} '\
                           +'QPushButton {max-height: 18px; border: 1px solid grey; border-radius: 6px; padding: 0 4p;} '\
                           +'QPushButton:hover {max-height: 18px; border: 1px solid red; border-radius: 6px; padding: 0 4p;} '\
                           +'QRadioButton {max-height: 18px}'
        self.setStyleSheet(self.standard_style_sheet)
        # define used variables
        self.font_bold = QtGui.QFont()  #: a bold font
        self.font_bold.setPixelSize(11)
        self.font_bold.setBold(True)
        self.time_press = time.time()  #: time to check if mouse was double clicked 
        self.minimum = 'local'  #: variable used to switch between 'local' and 'global' minimum optimization
        self.plot_iterations = True  #: variable used to decide if iteration results should be plotted or not
        self.colors = list(col.TABLEAU_COLORS.values())*10  #: list of colors used for plotting
        self.__init__UI()
        self.__init_connections()

    def __init__UI(self):
        # define labels
        self.label_fit_parameter = QtWidgets.QLabel(u'fit parameter', self)
        self.label_fit_parameter.setFont(self.font_bold)
        self.label_save_parameter = QtWidgets.QLabel(u'save parameter', self)
        self.label_save_parameter.setFont(self.font_bold)
        self.label_a0 = QtWidgets.QLabel('a0', self)
        self.label_a1 = QtWidgets.QLabel('a1', self)
        self.label_fano = QtWidgets.QLabel('fano', self)
        self.label_FWHM = QtWidgets.QLabel('el. noise', self)
        self.label_strip_cycles = QtWidgets.QLabel('str cycles', self)
        self.label_strip_width = QtWidgets.QLabel('str width', self)
        self.label_smooth_cycles = QtWidgets.QLabel('smooth cycles', self)
        self.label_smooth_width = QtWidgets.QLabel('smooth width', self)
        self.label_calc_minima_order = QtWidgets.QLabel('min order', self)
        self.label_PU_factor = QtWidgets.QLabel('pile-up factor', self)
        self.label_PU_threshold = QtWidgets.QLabel('pile-up threshold', self)
        self.label_Escape_factor = QtWidgets.QLabel('escape factor', self)
        self.label_Escape_threshold = QtWidgets.QLabel('escape threshold', self)
        self.label_horizontal_separator_1 = QtWidgets.QLabel('', self)
        self.label_horizontal_separator_2 = QtWidgets.QLabel('', self)
        self.label_horizontal_separator_1.setFixedHeight(2)
        self.label_horizontal_separator_2.setFixedHeight(2)
        self.label_horizontal_separator_1.setStyleSheet('QLabel {background-color : black}')
        self.label_horizontal_separator_2.setStyleSheet('QLabel {background-color : black}')
        self.specfit_fit_settings_layout.addWidget(self.label_fit_parameter, 0, 0, 1, 5)
        self.specfit_fit_settings_layout.addWidget(self.label_a0, 4, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_a1, 5, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_fano, 6, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_FWHM, 7, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_strip_cycles, 4, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_strip_width, 5, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_smooth_cycles, 6, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_smooth_width, 7, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_calc_minima_order, 8, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_PU_factor, 9, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_PU_threshold, 10, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_Escape_factor, 11, 2, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_Escape_threshold, 12, 0, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.label_horizontal_separator_1, 1, 0, 1, 5)
        self.specfit_fit_settings_layout.addWidget(self.label_horizontal_separator_2, 15, 0, 1, 5)
        self.specfit_fit_settings_layout.addWidget(self.label_save_parameter, 14, 0, 1, 1)
        # define checkboxes
        self.check_calc_minima = QtWidgets.QCheckBox('calc minima', self)
        self.check_PU_Lines = QtWidgets.QCheckBox('fit pile-up', self)
        self.check_Escape_Lines = QtWidgets.QCheckBox('fit escape', self)
        self.check_batch_fitting = QtWidgets.QCheckBox('fit in batches', self)
        self.check_save_background = QtWidgets.QCheckBox('save background', self)
        self.check_save_fitted_spectrum = QtWidgets.QCheckBox('save fitted spectrum', self)
        self.check_save_storage = QtWidgets.QCheckBox('save data in .npy', self)
        self.check_calc_minima.setCheckState(QtCore.Qt.CheckState.Checked)
        self.check_PU_Lines.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.check_Escape_Lines.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.check_batch_fitting.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.check_save_background.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.check_save_fitted_spectrum.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.check_save_fitted_spectrum.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.specfit_fit_settings_layout.addWidget(self.check_calc_minima, 8, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_PU_Lines, 9, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_Escape_Lines, 11, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_batch_fitting, 13, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_save_background, 16, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_save_fitted_spectrum, 17, 0, 1, 2)
        self.specfit_fit_settings_layout.addWidget(self.check_save_storage, 18, 0, 1, 2)
        # define entry fields
        self.entry_a0 = QtWidgets.QLineEdit('', self)
        self.entry_a1 = QtWidgets.QLineEdit('', self)
        self.entry_fano = QtWidgets.QLineEdit('', self)
        self.entry_FWHM = QtWidgets.QLineEdit('', self)
        self.entry_strip_cycles = QtWidgets.QLineEdit('5', self)
        self.entry_strip_width = QtWidgets.QLineEdit('60', self)
        self.entry_smooth_cycles = QtWidgets.QLineEdit('0', self)
        self.entry_smooth_width = QtWidgets.QLineEdit('1', self)
        self.entry_calc_minima_order = QtWidgets.QLineEdit('15', self)
        self.entry_PU_factor = QtWidgets.QLineEdit('0.1', self)
        self.entry_PU_threshold = QtWidgets.QLineEdit('1e-8', self)
        self.entry_Escape_factor = QtWidgets.QLineEdit('0.8', self)
        self.entry_Escape_threshold = QtWidgets.QLineEdit('1e-3', self)
        self.specfit_fit_settings_layout.addWidget(self.entry_a0, 4, 1, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_a1, 5, 1, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_fano, 6, 1, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_FWHM, 7, 1, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_strip_cycles, 4, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_strip_width, 5, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_smooth_cycles, 6, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_smooth_width, 7, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_calc_minima_order, 8, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_PU_factor, 9, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_PU_threshold, 10, 1, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_Escape_factor, 11, 3, 1, 1)
        self.specfit_fit_settings_layout.addWidget(self.entry_Escape_threshold, 12, 1, 1, 1)
        
    def __init_connections(self, ):
        self.entry_a0.textChanged.connect(partial(self.update_entries, self.parent.entry_a0))
        self.entry_a1.textChanged.connect(partial(self.update_entries, self.parent.entry_a1))
        self.entry_fano.textChanged.connect(partial(self.update_entries, self.parent.entry_fano))
        self.entry_FWHM.textChanged.connect(partial(self.update_entries, self.parent.entry_FWHM))
        self.entry_strip_cycles.textChanged.connect(partial(self.update_entries, self.parent.entry_strip_cycles))
        self.entry_strip_width.textChanged.connect(partial(self.update_entries, self.parent.entry_strip_width))
        self.entry_smooth_cycles.textChanged.connect(partial(self.update_entries, self.parent.entry_smooth_cycles))
        self.entry_smooth_width.textChanged.connect(partial(self.update_entries, self.parent.entry_smooth_width))
        self.entry_calc_minima_order.textChanged.connect(partial(self.update_entries, self.parent.entry_calc_minima_order))
        self.check_calc_minima.stateChanged.connect(lambda state: self.parent.check_calc_minima.setChecked(state == QtCore.Qt.CheckState.CheckState.Checked))
        
    def update_entries(self, entry, text):
        entry.setText(text)
    
    def set_PU_Escape_parameters(self,):
        if self.check_PU_Lines.isChecked():
            self.parent.s.PU_factor = float(self.entry_PU_factor.text())
            self.parent.s.PU_threshold = float(self.entry_PU_threshold.text())
            self.parent.s.calc_PU = True
        else:
            self.parent.s.calc_PU = False
        if self.check_Escape_Lines.isChecked():
            self.parent.s.Escape_factor = float(self.entry_Escape_factor.text())
            self.parent.s.Escape_threshold = float(self.entry_Escape_threshold.text())
            self.parent.s.calc_Escape = True
        else:
            self.parent.s.calc_Escape = False
        
    def show_sfs(self, _parent = None):
        self.parent = _parent
        self.show()
        self.activateWindow()
    
    def get_parent_directory(self, path):
        """
        function to return parent directory path
        """
        return os.path.abspath(os.path.join(path, os.pardir))
