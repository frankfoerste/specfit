#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:43:36 2018

@author: gesa
"""

import numpy as np
from PyQt5 import QtGui, QtWidgets
from copy import deepcopy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import traceback


class lines_widget(QtWidgets.QWidget):
    '''tool to define own lines '''
    def __init__(self):
    
        super(lines_widget, self).__init__()
        self.screen_properties = QtWidgets.QDesktopWidget().screenGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.popup_heigth = 350
        self.popup_width = 350
        self.setWindowTitle('User defined Lines')
        self.setGeometry((self.screen_width-self.popup_width)//2,
                         (self.screen_height-self.popup_heigth)//2,
                         self.popup_width, self.popup_heigth)
        
        self.number_lines = 0
        self.ident_list= [] #every line get a unique id, easier to debug
        self.ident_nr = 0
        
        self.popt_list= []#optimal parameter
        self.label_list= []
        self.x_list = [] #energy
        
        self.spec = []
        self.spec_label = '' #for example angle
        self.opt_param_not_found_dict = {}
        self.spec_loaded = False # determine wether a spectrum was loaded or not


       # self.spec = np.array([1,2,3,4,6,8,6,5,3,2,1])
        self.a0 = 0
        self.a1 = 1

        
        self.init_UI()
        
        
    def init_UI(self):
        #define labels
        self.label_line_label = QtWidgets.QLabel('label: ', self)
        self.label_line_start = []
        self.label_line_end = []
        
        #define entries
        self.entry_label_list = []
        self.entry_start_list = []
        self.entry_end_list = []
        
        #define buttons
        self.button_remove_list = []
        self.button_new_line = QtWidgets.QPushButton('new line',self)
        self.button_new_line.clicked.connect(self.add_new_line)
        self.button_check_lines = QtWidgets.QPushButton('check lines',self)
        self.button_check_lines.clicked.connect(self.check_lines)
        
        
        self.label_line_label.setGeometry(0,0,100,20)
        self.button_new_line.setGeometry(0,30,100,20)
        self.button_check_lines.setGeometry(120,30,100,20)
        
        
        self.add_new_line()
        


    def display_lines_widget(self):
        self.show()
        

    def get_channel_of_energy(self,energy): 
            energy = np.array(energy).astype(float)
            return np.rint((np.divide(np.subtract(energy,self.a0),self.a1))).astype(int)
        
    def get_energy_of_channel(self,channel):
            channel = np.array(channel).astype(float)
            return  np.add(np.multiply(self.a1,channel) , self.a0)


        
    def add_new_line(self):
        
        self.number_lines +=1
        n= self.number_lines
        self.ident_nr +=1
        self.ident_list.append(self.ident_nr)

        self.label_line_start.append(QtWidgets.QLabel('start: ', self))
        self.label_line_end.append(QtWidgets.QLabel('end: ', self))    
        self.entry_label_list.append(QtWidgets.QLineEdit('',self))
        self.entry_start_list.append(QtWidgets.QLineEdit('',self))
        self.entry_end_list.append(QtWidgets.QLineEdit('',self))    
        self.button_remove_list.append(QtWidgets.QPushButton(QtGui.QIcon.fromTheme('window-close'),'',self))
        
        self.label_line_start[-1].setGeometry(110,n*30,50,20)
        self.label_line_start[-1].show()
        self.label_line_end[-1].setGeometry(210,n*30,50,20)
        self.label_line_end[-1].show()
        
        self.entry_label_list[-1].setGeometry(0,n*30,100,20)
        self.entry_label_list[-1].show()
        self.entry_start_list[-1].setGeometry(150,n*30,50,20)
        self.entry_start_list[-1].show()
        self.entry_end_list[-1].setGeometry(240,n*30,50,20)
        self.entry_end_list[-1].show()
        
        self.button_remove_list[-1].setGeometry(310,n*30,20,20)
        ident = deepcopy(self.ident_nr)
        self.button_remove_list[-1].clicked.connect(lambda: self.remove_line((ident)))
        self.button_remove_list[-1].show()
        self.button_new_line.move(0,n*30+30)
        self.button_check_lines.move(120,n*30+30)
        

    def new_filled_line(self,label,start,end):
        self.add_new_line()
        self.entry_label_list[-1].setText(label)
        self.entry_start_list[-1].setText(start)
        self.entry_end_list[-1].setText(end)
        
        
    def remove_line(self,remove_id):
        self.number_lines -= 1
        i = self.ident_list.index(remove_id)
        
        for line_list in [self.label_line_start, self.label_line_end, self.entry_label_list, self.entry_start_list, self.entry_end_list, self.button_remove_list]:
            line_list[i].setVisible(False)
            self.line_list = line_list.remove(line_list[i])
        self.ident_list.remove(remove_id)
        ##bring all lines one up
        for line_under_i in range(i, self.number_lines):
            for line_list in [self.label_line_start, self.label_line_end, self.entry_label_list, self.entry_start_list, self.entry_end_list, self.button_remove_list]:
                line_list[line_under_i].move(line_list[line_under_i].x(), line_list[line_under_i].y()-30)
                    
        self.button_new_line.move(0,self.number_lines*30+30)
        self.button_check_lines.move(120,self.number_lines*30+30)
            
        
    def reset_2_default(self):
        ''' delate all lines'''
        self.number_lines = 0
        n= self.number_lines
        self.ident_nr = 0
        self.ident_list = []
        
        
          #i am sure there is a better way for this
        try:
             for hide_list in [self.label_line_start,self.label_line_end, self.entry_start_list, self.entry_end_list, self.entry_label_list, self.button_remove_list]:
                 for element in hide_list:
                     element.hide()
      
        except AttributeError: # if not yet defined
            pass

        self.label_line_start = []
        self.label_line_end= []  
        self.entry_label_list= []
        self.entry_start_list= []
        self.entry_end_list= []    
        self.button_remove_list= []
        
        self.button_new_line.move(0,n*30+30)
        self.button_check_lines.move(120,n*30+30)
        
        
    def get_label(self, label_entry, start_entry, end_entry):
        '''read the entries and return a user defined labels, define label as Start_end if not defined
        return None if no lines are defined'''
        label = label_entry.text()
        if (label == ''):
                label = '%s_%s'%(start_entry.text(), end_entry.text())
        return label
    
    
    def get_label_list(self):
        ''' return the labels of all filled entries '''
        label_list = []
        for label, start, end in zip(self.entry_label_list, self.entry_start_list,self.entry_end_list):
            if  start.text() != '' and end.text()!='':
                label_list.append(self.get_label(label,start,end))
        return label_list
    
    
    def get_start_list(self):
        start_list = []
        for label, start, end in zip(self.entry_label_list, self.entry_start_list,self.entry_end_list):
            if  start.text() != '' and end.text()!='':
                start_list.append(start.text())
        return start_list
    
    
    def get_end_list(self):
        end_list = []
        for label, start, end in zip(self.entry_label_list, self.entry_start_list,self.entry_end_list):
            if  start.text() != '' and end.text()!='':
                end_list.append(end.text())
        return end_list
        
                        
    def load_spec(self,spec,a0,a1,label):
        self.spec = np.array(spec)
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.spec_label = label
        self.spec_loaded = True

        
    def gaussian(self,e,a,e0,sigma):
        ''' normalized gaussian with a factor a'''
        return a/(np.sqrt(2*np.pi)*sigma) * np.exp(-(e - e0)**2/(2*sigma**2))
#        return a* 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(e - e0)**2/(2*sigma**2))
        
        
    def check_lines(self):    
        '''check if gaussian fits look alright, not necessary! '''
        if len(self.spec) == 0:
            print('please load a measurement first')
        self.gaussian_fit()
        self.show_gaussian_fits()

                
    def show_gaussian_fits(self):
        ''' plot the gaussian fits and the data in new windows to check if the fit looks ok '''
        for popt,label,x in zip(self.popt_list,self.label_list,self.x_list):
            plt.close(label)
            plt.figure(label)
            plt.plot(x,self.spec[self.get_channel_of_energy(x)], 'b+:', label = 'data')
            plt.plot(x, self.gaussian(x,*popt),'ro:', label = 'fit')
            plt.legend()
            plt.show()
    
        
    def gaussian_fit(self):         
        ''' 
        try a gaussian fit for every user defined line, return the labels of sucessfully fitted lines and its parameters
        '''       
       
        self.popt_list = []
        self.label_list = []
        self.x_list = []
        for label_entry, start_entry, end_entry in zip( self.entry_label_list, self.entry_start_list, self.entry_end_list):
            if  start_entry.text() != '' and end_entry.text()!='':
                label= self.get_label(label_entry, start_entry, end_entry )           
                start_ch = self.get_channel_of_energy(start_entry.text())
                end_ch = self.get_channel_of_energy(end_entry.text())
                x = self.get_energy_of_channel(np.array(range(start_ch, end_ch+1)))
                selected_area = self.spec[self.get_channel_of_energy(x)]
                mean = self.get_energy_of_channel(start_ch + (-start_ch+end_ch)/2)
                sigma = 0.2
                maximum = np.max(selected_area)
                try:
                    popt,pcov = curve_fit(self.gaussian,x,selected_area,p0=[maximum,mean,sigma])
                    self.popt_list.append(popt)
                    self.label_list.append(label)
                    self.x_list.append(x)
                except RuntimeError: 
                    print('gaussian fit not possible, take default parameters for ', label)
                    #traceback.print_exc()
                    self.popt_list.append([maximum,mean, sigma])
                    self.gaussian_impossible_handler(label)

                    
                    
    def gaussian_impossible_handler(self, label):
        '''build a dict: linelabel -> [list of angles where gaussian parameters could not be found]'''
        if self.spec_label != 'test': #do not store in dict for checkfit
            if label not in self.opt_param_not_found_dict.keys():
                self.opt_param_not_found_dict[label] = []
            self.opt_param_not_found_dict[label].append(str(self.spec_label))


    def get_user_defined_lines_data(self):
         '''returns an array in Form [[label1,start1,end1],[label2,start2,end2]] '''   
         return (np.array((self.get_label_list(),self.get_start_list(), self.get_end_list())).T).tolist()


    def build_output_rows(self):
        user_lines_list = []
        try:
            for popt in self.popt_list:
                e = self.get_energy_of_channel(np.array(range(0,len(self.spec))))
                line_spectra = self.gaussian(e,*popt) #take standard values
                e0 = popt[1]
                sigma = popt[2]
                line_spectra[:min(self.get_channel_of_energy(e0-4*sigma), len(line_spectra))]= 0
                line_spectra[max(self.get_channel_of_energy(e0+4*sigma),0):]= 0
                user_lines_list.append(line_spectra)
        except:
            print('could not build a line')
            traceback.print_exc()
        return user_lines_list
            
        
    def get_user_defined_lines(self):
        '''
        returns the lines in a format that specfit can understand
        '''
        self.gaussian_fit()
        line_spectra_list = self.build_output_rows()
        self.label_list = self.get_label_list()
        if (len(self.label_list) != len(line_spectra_list)):
            print('len label list', len(self.label_list))
            print('len line spectra list', len(line_spectra_list))
        assert (len(self.label_list) == len(line_spectra_list))
        return self.label_list, line_spectra_list
            
        
    def plot_output(self,line_spectra_list):
         ''' for testing/debugging'''
         for label,line_spectra,popt in zip(self.label_list, line_spectra_list, self.popt_list):
             a = popt[0]
             plt.figure('%s-a=%s'%(label,a))
             plt.plot(self.get_energy_of_channel(range(len(line_spectra))),np.multiply(line_spectra,a), 'ro:', label = 'fit')
             plt.plot(self.get_energy_of_channel(range(len(line_spectra))),self.spec, 'b+:', label = 'data')
         plt.show()

            
    def print_gaussian_fit_error_results(self):
        if len(self.opt_param_not_found_dict.keys())>0:
            print('--------------------------------------------------------------')
            for k in self.opt_param_not_found_dict.keys():
                print('%s: %s spectra could not be fitted'%(k,len(self.opt_param_not_found_dict[k])))
                print(self.opt_param_not_found_dict[k])
            print('------------------------------------------------------------------')
