# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:25:09 2017

@author: gesa
"""

# -*- coding: utf-8 -*-

###############################################################################
# Force matplotlib to not use any Xwindows backend.
import numpy as np
import matplotlib.pyplot as plt
plt.ioff() #again force to not show images
import sys
import collections #to use ordered dicts
import os
###############################################################################

def _cast_to_int(x):
    try:
        new_int = int(_cast_to_float(x))
    except:
        print('some error with casting of Data - maybe empty String?')
        sys.exit()
    return new_int


def _cast_to_float(x):
    x= x.strip()
    if 'e' in str(x):
        new_float = float(x.split('e')[0])*10**int((float(x.split('e')[1])))
    else:
        new_float = float(x)
    return new_float


def txt2spec_para(file_path):
    ''' 
    This function reads out the spectrum of a .txt-file and reads out the
    detector parameters given in the .txt-file.
    
    Parameters
    ----------
    file_path : str
        complete folder path of the .txt-file.
    
    Returns
    -------
    dict containing the spectrum
    list containing hard coded detector parameters 

    '''
    

    tfile = open(file_path,'r')
    file_content = tfile.readlines()
    
    angles = []
    spectra_tmp = []    
    
    for i,anglespec in enumerate(file_content):
        #anglespec.strip()
        anglespec = anglespec.strip('\n').strip().split(' ')
        cur_angle = '%s'%(_cast_to_float(anglespec[0]))
        angles.append(cur_angle)
        help_anglespec = (anglespec[1:]) #remove angle
        spectra_tmp.append([])
        for x in help_anglespec:
            spectra_tmp[i].append(_cast_to_float(x))
    tfile.close()

    ### split spectra into portions
    spectra = collections.OrderedDict()
    for i,angle in enumerate(angles):
            spectra[angle] = np.array(spectra_tmp[i] )
    #detector paametrs:
    #0 is a0
    #1 is a1
    #2 Fano
    #3 FWH
    #4 life time?
    # maximalenergie die betrachtet werden soll....? siehe show_Roi
    #6 gating time
    
                    
   # parameters = [0.044748349590333983, 0.0090837427460835011, 0.493, 0.074071083612883247]
	#[0.08534,0.009,0.115,0.15353084747784951] #veronika3
   # parameters = [0.0360483374766,0.0090438203411  , 0.115, 0.06] 
   #calib from Text:a0 +a1*x
    parameters =[ 0.04369487, 0.0090333623, 0.4927366167856802,0.074025388749522497,1,0,280e-9, 1]
    parameters[5] = np.add(np.multiply(len(spectra_tmp[i])-1,parameters[1]),parameters[0])
    return spectra, parameters
    
    
def txt2channels(file_path):
    ''' 
    This function reads out the number of channels of a .txt-file .
    '''  
    tfile = open(file_path,'r')
    channels = len(tfile.readline().split()) -1 #remove angle information    
    tfile.close()
    return channels    
    
    
def txt2positions(file_path):
    tfile = open(file_path, 'r')
    ticks = []
    for line in tfile:
        string_tick = line.split(' ')[0]
        ticks.append(_cast_to_float(string_tick))
    return np.array(np.meshgrid(1,1,ticks)).T.reshape(-1,3)
     
     
def txt2num_spec(file_path):
    tfile = open(file_path, 'r')
    lines = tfile.readlines()
    
    return len(lines)
    
    
def plot_spectra_for_angle(energy,spectra,bg,fit,angle,save_folder_path): 
    '''plot and save an image with spectra,background and spectra-fit   '''
    
    angle = _cast_to_float(angle)
    if not os.path.exists(save_folder_path +'/ang_images'):
        os.mkdir(save_folder_path +'/ang_images')
        
    savepath = save_folder_path +'/ang_images/fit_plot_{0:0.4f}.png'.format(angle)
    fit_with_bg = np.add(fit,bg)
    
    fig,ax =plt.subplots()
    try:    
        plt.yscale('log')
        ax.plot(energy,spectra, 'b.')
        ax.plot(energy,bg,'g-')
        ax.plot(energy,fit_with_bg,'r-')
    except:
        plt.yscale('linear')
        ax.plot(energy,spectra, 'b.')
        ax.plot(energy,bg,'g-')
        ax.plot(energy,fit_with_bg,'r-')
    
    plt.xlabel = 'Energy in keV'
    plt.ylabel = 'intenity/arb. un.'
    #ymin = min(i for i in spectra if i >5)/10
    #ymin = 1
    #plt.ylim(ymin, np.max(spectra)*5)
    plt.savefig(savepath)
    plt.close(fig)
    
def save_numpy_arrays(energy,spectra,bg,fit,angle,save_folder_path):    
    angle = _cast_to_float(angle)
    fit_with_bg = np.add(fit,bg)
    
    if not os.path.exists(save_folder_path +'/ang_npz'):
        os.mkdir(save_folder_path +'/ang_npz')
    
    outfile = save_folder_path +'/ang_npz/en_sp_bg_fit_{0:0.4f}.png'.format(angle)
    np.savez(outfile,energy,spectra,bg,fit_with_bg)
    
    
def plot_angle_line_intensities(results,angles,savepath,n=0):
    ''' plot for every result-line the angles on x-axis and the intenity on x axis
    cut n angles on both sides -cause they look ugly
    Needs:
        results - list of all results dim: numberlines x numberspecptra
        angles -list of all angles
        savepath - path of allsaved.txts is used to save the images
        n - cut off the n first and last angles     
    '''
    #do not cut more than you have
    if n >= len(angles)/2:
        print('ERROR, you are cutting 2*%s angles in the plot! Thats more than you have... n is set zu 0.'%(n))
        n = None
    else:
    #cut the first and the last    
        angles_cut = angles[n:-n]  
    
        
    #cast all to floast   
    angles_cut = np.vectorize(_cast_to_float)(angles_cut)
    angles = np.vectorize(_cast_to_float)(angles)
    
    for r,sp in zip(results,savepath):
        #try: r_cut = np.array(r[n:-n])
        #except: pass #if n=None
        r = np.array(r)

        #plot and save it!     
        try:
            #fig,ax =plt.subplots()
            #ax.plot(angles_cut,r_cut,linestyle = 'None',marker = '.')
            #plt.xlabel = 'emission angle /°'
            #plt.ylabel = 'intenity/(photons x sr^(-1))'
            #plt.savefig(sp+'cut%s.png'%n)
           # plt.close()
            
            fig,ax =plt.subplots()
            ax.plot(angles,r,linestyle = 'None',marker = '.')
            plt.xlabel = 'emission angle /°'
            plt.ylabel = 'intenity/(photons x sr^(-1))'
            plt.savefig(sp+'.png')
            plt.close()        
            
        except: #most likely an dimension error
            print('ERROR')
            print(angles,r)
            print(len(angles), len(r))
