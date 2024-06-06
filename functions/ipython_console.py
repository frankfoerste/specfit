# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:16:19 2017

@author: frank
"""

##############################################################################
''' import of used modules '''
##############################################################################

from PyQt5.QtWidgets import QWidget
from qtconsole.rich_ipython_widget import RichIPythonWidget
from qtconsole.inprocess import QtInProcessKernelManager

##############################################################################
''' GUI Programm '''
##############################################################################

class IPythonConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ipython_widget = RichIPythonWidget()
        self.ipython_widget.kernel_manager = QtInProcessKernelManager()
        self.ipython_widget.kernel_manager.start_kernel(show_banner=False)
        self.ipython_widget.kernel_manager.kernel.gui = 'qt'
        self.ipython_widget.kernel_client = self.ipython_widget.kernel_manager.client()
        self.ipython_widget.kernel_client.start_channels()

    def exit_console(self, msg):
        """Custom command to exit the IPython console."""
        self.ipython_widget.kernel_client.stop_channels()
        self.ipython_widget.kernel_manager.shutdown_kernel()
        self.ipython_widget.close()
        
    def show(self,):
        self.ipython_widget.kernel_manager.kernel.shell.run_cell('import numpy as np')
        self.ipython_widget.kernel_manager.kernel.shell.run_cell('import matplotlib.pyplot as plt')
        self.ipython_widget.kernel_manager.kernel.shell.run_cell('"to enter the SpecFit class, use specfit. e.g. specfit.row_height"')
        self.ipython_widget.show()