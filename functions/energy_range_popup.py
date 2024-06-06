#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:35:08 2018

@author: frank
"""


from PyQt5 import QtCore, QtWidgets




##############################################################################
''' Energy Popup '''
############################################################################## 
   
class energy_range_popup(QtWidgets.QWidget):
    
    popup_energy = QtCore.pyqtSignal(int)
    def __init__(self):
        super(energy_range_popup, self).__init__()
        self.screen_properties = QtWidgets.QDesktopWidget().screenGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.popup_heigth = 70
        self.popup_width = 260
        self.setWindowTitle('specFit')
        self.setGeometry((self.screen_width-self.popup_width)//2,
                         (self.screen_height-self.popup_heigth)//2,
                         self.popup_width, self.popup_heigth)
        self.init_UI()
        
    def init_UI(self):
        
        ### initialize widgets ###
        
        self.label_message = QtWidgets.QLabel('What energy range are you interested?', self)
        self.label_energy_low = QtWidgets.QLabel('0 keV -', self)       
#        self.entry_low = QtWidgets.QLineEdit('',self)
        self.entry_high = QtWidgets.QLineEdit('',self)
        self.button_close = QtWidgets.QPushButton('Okay',self)
        
        ### set properties of widgets ###
        
        self.label_message.move(5,0)
        self.label_message.setFixedSize(250,20)
        self.label_energy_low.move(85,20)
        self.label_energy_low.setFixedSize(50,20)
#        self.entry_low.move(5,20)
#        self.entry_low.setFixedSize(30,20)
        self.entry_high.move(140,20)
        self.entry_high.setFixedSize(30,20)
        self.button_close.move(100,40)
        self.button_close.setFixedSize(50,20)
    
        self.button_close.clicked.connect(self.emit_energy_range)
   
        self.show()
        
    def emit_energy_range(self):
        self.energy = int(self.entry_high.text())
        self.popup_energy.emit(self.energy)
        self.close()
