# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from PyQt5 import QtWidgets, QtGui
import sys, os
import matplotlib.backends.backend_qt5agg as pltqt
import matplotlib.figure as figure
import spx_functions as spx
from glob import glob

class display_meas_points(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(display_meas_points, self).__init__(parent)
        screen_properties = QtWidgets.QDesktopWidget().screenGeometry()
        screen_width = screen_properties.width()
        screen_height = screen_properties.height()
        if sys.platform == 'linux' or sys.platform == 'linux2':
            self.homedir = os.environ['HOME']
        else: self.homedir = 'C:/'
        self.version = 'measurement range 1.00'
        try: QtGui.QIcon.setThemeName('ubuntu-mono-dark')
        except: pass
        self.window_heigth = 450
        self.window_width = 685
        self.setWindowTitle(self.version)
        self.setGeometry((screen_width-self.window_width)//2,
                         (screen_height-self.window_heigth)//2,
                         self.window_width, self.window_heigth)
        self.positions = []
        self.__init__GUI()
        self.__init__plot()

    def __init__GUI(self):
        self.button_load = QtWidgets.QPushButton('load', self)
        self.button_load.setGeometry(5, 5, 100, 20)
        self.button_load.clicked.connect(self.load_data)
        self.button_load.setStyleSheet('QWidget {background-color:lightblue}')

    def __init__plot(self):
        '''define the layout of the plot frame '''
        
        self.figure_meas_points = figure.Figure(dpi = 80)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas_meas_points = pltqt.FigureCanvasQTAgg(self.figure_meas_points)
        self.canvas_meas_points.setParent(self)
        self.canvas_meas_points.move(5,30)

        # this is the Navigation widget for matplotlib plots
        self.toolbar_meas_points = pltqt.NavigationToolbar2QT(self.canvas_meas_points, self)        
        self.toolbar_meas_points.setGeometry(0,380, 590, 50)
        
        ### establish connections with User action
#        self.canvas_meas_points.mpl_connect('button_release_event',self.store_azim_elev)

    def show_display_meas_points(self):
        self.show()
        self.activateWindow()
    
    def reset_2_default(self):
        try: self.figure_meas_points.delaxes(self.ax_canvas_meas_points)
        except: pass
        self.positions = []
        
    def load_data(self):
        self.reset_2_default()
        folderpath = QtWidgets.QFileDialog.getExistingDirectory()         ### function for a folder-GUI
        file_path_list = glob(folderpath+'/*spx')
        self.retrieve_positions_spx(file_path_list)
        
    def retrieve_positions_spx(self, file_path_list):
        """ This function reads out the position of all spx files in the given list
        """
        for i in range(len(file_path_list)):
            self.positions.append(spx.spx_tensor_position(file_path_list[i]))
        self.positions = np.asarray(self.positions)
        self.display_meas_points()
    
    def display_meas_points(self):
        # create an axis
        self.ax_canvas_meas_points = self.figure_meas_points.add_subplot(111, projection='3d')
        self.ax_canvas_meas_points.grid(False)
#        self.ax_canvas_meas_points.set_xlim(self.xlim)
#        self.ax_canvas_meas_points.set_ylim(self.ylim)
#        self.ax_canvas_meas_points.set_zlim(self.zlim)
        self.ax_canvas_meas_points.xaxis.set_pane_color((1,1,1,1))
        self.ax_canvas_meas_points.yaxis.set_pane_color((1,1,1,1))
        self.ax_canvas_meas_points.zaxis.set_pane_color((1,1,1,1))
        
        self.ax_canvas_meas_points.scatter(self.positions[:,0], 
                                           self.positions[:,1], 
                                           self.positions[:,2])
        self.canvas_meas_points.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    pmp = display_meas_points()
    pmp.show()
    app.exec_()

if __name__ == '__main__':
    main()
