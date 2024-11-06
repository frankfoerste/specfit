#!/usr/bin/python
"""
"""
# from numba import jit
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import argrelextrema, convolve
import re
import os, sys
import xraylib as xrl

##############################################################################
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SpecFit(object):
    """
    """
#   first self.load_spec(spectrum) for initialise self.meas_load, spectrum:=[[int,int,int,...,]]
#   then self.set_ROI(ROI) for initialise self.ROI and self.Bins and self.Meas
#   than you have to calculate the Background with self.strip()
#   PU_Lines not works yet
#   M-Lines dont work yet
#   escape is fitted but nt counted as intensity
    
    def __init__(self):

        # parameters without preset
        self.xraylib = True#use xralib instead of TUB, True or False
        self.xraylib_linesets = self.get_all_xrlib_linesets()# [Ka_Lines, Kb_Lines,K_Lines, L1_Lines, L2_Lines, L3_Lines] or list of empty lists
        self.Z = [] #list of all added lines, [[29,['K-L3', 'K-L2', 'K-L1'],K-linie],[]]
        self.Lines = [] #list of dicts [{'I':1,'Z':29, 'edge':'K-line', 'g_sum':0.8, 'lines':[{'E':2.5, 'g':0.1},{'E':2.6, 'g':0.2},{}]},{...}]
        self.user_defined_lines = []# for every user defined line a list with relative intensities per channel
        self.udl_label_list = [] #a list with the labels of every user defined line
        self.result_udl = []# a list storing the intensity results for every user defined line
        self.PU_Elements = [] #stores the Z Values of all added elements, z values appear mulitple times
        self.life_time = 0 #used to calculate pileups..pu seems to be to low
        self.real_time = 0 #used to calculate pileups..pu seems to be to low
        self.ROI = [0, 1]#ROI in channel, used for linear and nonlinear fit, not for stripping

        # parameters with preset
        self.gating_time = 8e-3 #used for pileuplines
        self.strip_cycles = 30
        self.strip_width = 60
        self.smooth_cycles =  2
        self.smooth_width = 3
        self.minchange = 1e-3# threshold for non-linear fit
        self.order = 15 # test parameter for linear deconvolution
        self.minima = [] # list containing all minima. If only minima on sum_spec, self.minima becomes np.ndarray
        self.Det = [-0.96, 0.01, 0.112, 0.0488]  # a0,     a1,    Fano,  FWHMel
        self.calc_PU = 0.1
        self.PU_factor = 1
        self.PU_threshold = 1E-8
        self.calc_Escape = 0.8
        self.Escape_factor = 0.8
        self.Escape_threshold = 1E-3

        # attributes
        self.PU_Lines = None#stores the PU lines list as a list of dicts [{E:g},{E:g}]
        self.meas_load = None #stores the spectrum as a list
        self.Strip = None #stores the stripped background
        self.fit_in_progress = False #used to avoid recalculating spectra Matrix
        
        #unsused params
        
#        self.PolynomeOrder = 2        
#        self.Ecal = True        
#        self.background_polynome = []       
#        self.offset = 0
#        self.corrlim = 0        
        
 
    def get_all_xrlib_linesets(self):
        ''' return List of Ka_Lines, Kb_Lines,K_Lines, L1_Lines, L2_Lines, L3_Lines if xraylib installed, else return empty list '''
        try:
            # selection = (r'KL.*_LINE|KM.*_LINE')
            # K-Lines
            Kp=re.compile(r'KL.*_LINE|KM.*_LINE|KN.*_LINE|KO.*_LINE|KP.*_LINE')
            Kap=re.compile(r'KL.*_LINE')
            Kbp=re.compile(r'KM.*_LINE|KN.*_LINE|KO.*_LINE|KP.*_LINE')
            # L-Lines
            Lp=re.compile(r'L1.*_LINE|L2.*_LINE|L3.*_LINE')
            L1p=re.compile(r'L1.*_LINE')
            L2p=re.compile(r'L2.*_LINE')
            L3p=re.compile(r'L3.*_LINE')
            # M-Lines
            Mp = re.compile(r'M1.*_LINE|M2.*_LINE|M3.*_LINE|M4.*_LINE|M5.*_LINE')
            M1p = re.compile(r'M1.*_LINE')
            M2p = re.compile(r'M2.*_LINE')
            M3p = re.compile(r'M3.*_LINE')
            M4p = re.compile(r'M4.*_LINE')
            M5p = re.compile(r'M5.*_LINE')
    
            Xdir=dir(xrl)
            self.K_Lines=[getattr(xrl,l) for l in Xdir if Kp.match(l)]
            self.Ka_Lines=[getattr(xrl,l) for l in Xdir if Kap.match(l)]
            self.Kb_Lines=[getattr(xrl,l) for l in Xdir if Kbp.match(l)]

            self.L_Lines=[getattr(xrl,l) for l in Xdir if Lp.match(l)]
            self.L1_Lines=[getattr(xrl,l) for l in Xdir if L1p.match(l)]
            self.L2_Lines=[getattr(xrl,l) for l in Xdir if L2p.match(l)]
            self.L3_Lines=[getattr(xrl,l) for l in Xdir if L3p.match(l)]
            
            self.M_Lines = [getattr(xrl,l) for l in Xdir if Mp.match(l)]
            self.M1_Lines = [getattr(xrl,l) for l in Xdir if M1p.match(l)]
            self.M2_Lines = [getattr(xrl,l) for l in Xdir if M2p.match(l)]
            self.M3_Lines = [getattr(xrl,l) for l in Xdir if M3p.match(l)]
            self.M4_Lines = [getattr(xrl,l) for l in Xdir if M4p.match(l)]
            self.M5_Lines = [getattr(xrl,l) for l in Xdir if M5p.match(l)]
            
            self.xrl_lines = [self.K_Lines, self.Ka_Lines, self.Kb_Lines,
                              self.L_Lines, self.L1_Lines, self.L2_Lines, self.L3_Lines,
                              self.M_Lines, self.M1_Lines, self.M2_Lines, self.M3_Lines, self.M4_Lines, self.M5_Lines]
            # xrl_lines = [Ka_Lines, Kb_Lines,K_Lines, L1_Lines, L2_Lines, L3_Lines, L_Lines,
            #              M3_Lines, M4_Lines, M5_Lines]
        except:
            self.xrl_lines = [[],[],[],
                          [],[],[],[],
                          [],[],[],[],[],[]]
            # xrl_lines = [[],[],[],
            #              [],[],[],[],
            #              [],[],[]]
        
        return self.xrl_lines

        
    def load_spec(self, spectrum, add = False): #spectrum has to be a list: [.,.,.]
        '''load spec in self.measure load '''
        if add:
            try: 
                self.meas_load += np.array(spectrum, 'd')
            except (TypeError, AttributeError): 
                self.meas_load += np.array(spectrum, 'd')
        else:
            if type(spectrum) == np.ndarray:
                self.meas_load = spectrum
            else:
                self.meas_load = np.array(spectrum, 'd')


    def clear_spec(self):
        self.meas_load = None


    def get_spec(self):
        return list(self.meas_load)

#    def background_polynome(self, x):
#        P = 0
#        for i in range(len(self.background_polynome)):
#            P += self.background_polynome[i]*pow(x, i)
#        return P


    def get_background(self):
        if self.Strip != None: strip = self.Strip
        else:
            try: strip = list(np.zeros(len(self.meas_load)))
            except TypeError: return None
        
        #if self.background_polynome:
        #    for _c in self.Bins:
        #        strip[_c] = self.background_polynome(self.get_E(_c))
        return strip


    def get_fit_spec(self):
        try: 
            return self.calc_spec()
        except (TypeError, AttributeError):
            try: return list(np.zeros(len(self.meas_load)))
            except TypeError:return None


    def get_result(self, print_results = False):
        Result = {}
        for line in self.Lines:
            try: element = xrl.AtomicNumberToSymbol(line['Z'])
            except: element = xrl.SymbolToAtomicNumber(line['Z'])
            Result['%s_%s'%(element, line['edge'])] = {
                'I':line['I'],
                'Z':line['Z'],
                'Edge':line['edge'],
                }
        for udl_I, label in zip(self.result_udl,self.udl_label_list):
            Result[label] = {'I': udl_I}
        if print_results:
            print('Results:\t', Result)
        return Result


    def set_ROI(self, ROI = None):
        '''
        the variables ROI, Bins and Meas are defined. 
        If no ROI is set the variables are None
        otherwise the ROI is set to the energy-channels as [ROI_E_channel_low, ROI_E_channel_high]
        Bins are the corresponding energies as array
        Meas are the corresponding intensity values as array
        If user defined lines are present their ROIs are stored in 
        self.user_defined_lines_ROI

        Parameters
        ----------
        ROI : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if not ROI:
            self.ROI = None
            self.Bins = None
            self.Meas = None
            return
        if isinstance(ROI[0], float):
            self.ROI = [np.round(((E - self.Det[0])/self.Det[1]),0).astype(int) for E in ROI]
        else: self.ROI = ROI
        self.Bins = np.arange(self.ROI[0], self.ROI[1])
        self.Meas = self.meas_load[self.ROI[0]:self.ROI[1]]
        try:
            self.user_defined_lines_ROI = []
            for i,udl in enumerate(self.user_defined_lines):
                self.user_defined_lines_ROI.append(udl[self.ROI[0]:self.ROI[1]]) 
        except: print('somethings wrong with user defined lines, ROI definition')
             
        
    def strip(self, calc_minima = True, only_minima_on_sum_spec = True): 
        '''
        To erase the background of an energy dispersive x-ray spectrum the 
        spectrum is smoothed and the so generated background is stripped off

        Parameters
        ----------
        calc_minima : boolean, optional
            if True local intensity minima are calculated. Default is True
        only_minima_on_sum_spec : boolean, optional
            if True, the local intensity minima will only calculated from the 
            sum spectrum

        Returns
        -------
        spec : np.ndarray
        '''
        ### define stipping function if dimension of loaded spec is one
        def __strip_jit(spec, cycles=3, width=75, smooth=False):
            '''
            if possible use jit to strip the spectrum

            Parameters
            ----------
            spec : np.ndarray
                spectrum
            cycles : int, optional
                number of smoothing iterations. The default is 3.
            width : int, optional
                Number of channels to calculate mean value. The default is 75.
            smooth : boolean, optional
                if true, spectrum was smoothed. Not implemented anymore. The default is False.

            Returns
            -------
            spec : np.ndarray
            '''
            kernel = np.ones(width) / width
            for i in range(cycles):
                spec_tmp = spec.copy()
                smoothed_spec = convolve(spec_tmp, kernel, mode = 'same')
                for j in range(len(spec)):
                    if (spec_tmp[j] >= smoothed_spec[j]) or smooth:
                        spec[j] = smoothed_spec[j]
            return spec
        
        ### define stipping function if dimension of loaded spec is greater one
        def __strip_array(spec, cycles = 3, width = 75,):
            spec_tmp = np.copy(spec)
            _hw = int(width / 2)
            # start = time()
            for i in range(cycles):
                for j in range(_hw, spec.shape[-1] - _hw):
                    spec_slice = spec_tmp[...,j - _hw : j + _hw]
                    mean = np.mean(spec_slice,axis = -1)
                    # if spec_tmp[..., j + _hw] >= mean:
                    spec[..., j] = mean
                    # spec[..., j+ _hw][np.greater(spec[..., j+ _hw],mean)] = mean[np.greater(spec[..., j+ _hw],mean)]
            return spec
        
        
        ### copy self.meas_load to not alter it
        Spec = np.copy(self.meas_load)
        ### prior stripping check if spectrum shall be smoothed
        if self.smooth_cycles > 0:
            Spec = __strip_jit(Spec, cycles = self.smooth_cycles, 
                               width = self.smooth_width, smooth = True)
        ### if no smoothing required check if loaded Spec's dimension is bigger
        ###  than 1, apply array stripping
        if np.ndim(Spec) > 1:
            self.Strip = __strip_array(Spec, cycles = self.strip_cycles, 
                                       width = self.strip_width,)
        else:
            if calc_minima:
                ### check if local intensity minima shall only be calculated on
                ### sum spectrum or every time
                if only_minima_on_sum_spec:
                    ### on default type(self.minima) == list, on first minima
                    ### calculation the type is changed to np.ndarray, this way no
                    ### extra minima calculation is performed
                    if type(self.minima) != np.ndarray:
                        self.minima = argrelextrema(Spec[self.ROI[0]:self.ROI[1]], np.less, order = self.order)[0]
                else:
                    self.minima = argrelextrema(Spec[self.ROI[0]:self.ROI[1]], np.less, order = self.order)[0]
                ### if there are minima, linear interpolate between the minima and
                ### keep lowest intensity, return smoothed and stripped spectrum
                if len(self.minima) != 0:
                    ### form linear interpolation over all channel and intensity
                    ### minima
                    interpolation = np.interp(np.arange(0,len(Spec[self.ROI[0]:self.ROI[1]])),
                                    xp = np.append(0,np.append(self.minima,len(Spec[self.ROI[0]:self.ROI[1]]))),
                                    fp = np.append(Spec[self.ROI[0]:self.ROI[1]][0],
                                                   np.append(Spec[self.ROI[0]:self.ROI[1]][self.minima],Spec[self.ROI[0]:self.ROI[1]][-1])))
                    ### mask where the interpolation is smaller than the spectrum
                    mask = Spec[self.ROI[0]:self.ROI[1]] > interpolation
                    ### replace spectrum who comply with mask with interpolation
                    Spec[self.ROI[0]:self.ROI[1]][mask] = interpolation[mask]
                    ### smooth if required
                    self.Strip = __strip_jit(Spec, cycles = self.strip_cycles, width = self.strip_width)
                else:
                    self.minima = [0,len(Spec[self.ROI[0]:self.ROI[1]])]
                    interpolation = np.interp(np.arange(0,self.minima[1]),
                                    xp = self.minima,
                                    fp = [Spec[self.ROI[0]], Spec[self.ROI[1]]])
                    mask = Spec[self.ROI[0]:self.ROI[1]] > interpolation
                    Spec[self.ROI[0]:self.ROI[1]][mask] = interpolation[mask]
                    self.Strip = __strip_jit(Spec, cycles = self.strip_cycles, width = self.strip_width)
            else:
                self.Strip = __strip_jit(Spec, cycles = self.strip_cycles, width = self.strip_width)
        return Spec


    def nl_det(self):
        '''
        This function calculates the PileUp spectrum
        '''
        self.NLDet = np.zeros(len(self.meas_load))
        if self.PU_Lines:
            energy = self.energy
            for _l in self.PU_Lines:
                try:
                    self.NLDet += self.det_resp(_l['E'], energy)*_l['g']*self.Det[1]
### Leona: Müssten hier die verschieden Breiten der Typen berücksichtig werden?:
#                    self.NLDet += self.det_resp(_l['E'], energy,T=_l['type'])*_l['g']*self.Det[1]

                except TypeError:
                    print('SpecFit.nl_det() laeuft:')
                    print('_l: ', _l)
                    print('self.PU_Lines: ', self.PU_Lines)
                    print('_l[g]*self.Det[1]: ', _l['g']*self.Det[1])
                    print('self.Det[1]: ', self.Det[1])
        

    def get_lines(self, Zlist):
        ''' needs Z as defined in addlines, 
        [[29,['K-L3', 'K-L2', 'K-L1'],K-linie],[]]
        returns LL which is a list of dicts
        [{'I':1,'Z':29, 'edge':'K-line', 'g_sum':0.8, 'lines':[{'E':2.5, 'g':0.1},{'E':2.6, 'g':0.2},{}]},{...}]'''
        LL = []
        g_sum = {}
        ### calculate self.energy
        self.get_E()
        ### create LL dictionary
        for Z in Zlist:
            _Z, lines, name = Z
            sum_name = '%s-%s'%(_Z,name)
            g_sum[sum_name] = 0
            lines_list = []
            for _l in lines:
                factor = 1
                if sum_name[-1] == 'L':
                    if _l in self.L1_Lines:
                        factor = xrl.JumpFactor(_Z, xrl.L1_SHELL) * xrl.FluorYield(_Z, xrl.L1_SHELL)    
                    elif _l in self.L2_Lines:
                        factor = xrl.JumpFactor(_Z, xrl.L2_SHELL) * xrl.FluorYield(_Z, xrl.L2_SHELL)
                    elif _l in self.L3_Lines:
                        factor = xrl.JumpFactor(_Z, xrl.L3_SHELL) * xrl.FluorYield(_Z, xrl.L3_SHELL)      
                try:
                    g_sum[sum_name] += (xrl.RadRate( _Z, _l)*factor)
                    if not(xrl.LineEnergy(_Z,_l) == 0.0 or xrl.RadRate( _Z, _l)==0.0):
                        lines_list.append({ 'E':xrl.LineEnergy(_Z,_l), 'g':xrl.RadRate( _Z, _l)*factor, 'type' : 'Fluorescence'})
                except ValueError: pass                    
            LL.append({
                'I':1,
                'Z':_Z,
                'edge':name,
                'g_sum': g_sum[sum_name], 
                'lines': lines_list                  
                })
        return LL


    def addLines(self, Z_and_Lines):
        '''expected dict{'element':(True/False, ['linename',...], Z), '':...,}
        example: {'Cu': (True, ['K-line', 'L1'], 29)}
        build list Z which has the form: [[29,['K-L3', 'K-L2', 'K-L1'],K-linie],[]]'''
        self.Z = []
        self.PU_Elements = []
        entry = []
        for element in Z_and_Lines:
            if not Z_and_Lines[element][0]: 
                print('False- element is not fittet')
                continue
            for line in Z_and_Lines[element][1]:
                entry = [Z_and_Lines[element][2]] #'z.B. L1'
                with HiddenPrints(): #bloc print, cause xraylib prints a lot of waste
                    entry.append(self.get_xraylib_lineset(line))
                entry.append(line)
                #Pile-up Linien nicht berücksichtigen, da sie später separat hinzugefügt werden
                if entry[1] != None:
                    self.Z.append(entry)

        self.Lines = self.get_lines(self.Z)
        for i in range(len(self.Lines)):
            self.PU_Elements.append(self.Lines[i]['Z']) #[29,29,21,..]
        self.PU_Lines = self.pile_up_lines(self.Lines, self.PU_Elements)

        
    def get_xraylib_lineset(self,lineset): #getlineset of line #[Ka_Lines, Kb_Lines,K_Lines, L1_Lines, L2_Lines, L3_Lines,L_Lines]
        d = {'K': 0, 'Ka': 1, 'Kb': 2, 'K-line': 0,
             'L': 3, 'L1': 4, 'L2': 5, 'L3': 6,
             'M': 7, 'M1': 8, 'M2': 9, 'M3': 10, 'M4': 11, 'M5': 12}
        # d = {'Ka': 0 , 'Kb':1, 'K':2, 'K-line':2, 'L1':3, 'L2':4, 'L3':5, 'L':6,
        #      'M3': 7, 'M4': 8, 'M5': 9}
        if not lineset in d:
            print ('Caution! line not in dict %s'%(lineset))
        else:
            return self.xraylib_linesets[d[lineset]]
        
        
    def get_det_width(self, E):         # in rms
        Fano, elFWHM = self.Det[2], self.Det[3]
        if Fano < 0 or elFWHM < 0 or E <0: return 0.001
        a = (elFWHM/2.35)**2 + E * 3.65e-3 * Fano ### / (3.65e-3)
#        w = 3.65e-3
#        a = ((elFWHM/2.35)*w)**2 + E * w * Fano ###  formula to have meaningful noise value in electrons
        if a < 0: return 0.001 #don't now why it happens - but it happens.
        try: return np.sqrt(a)
        except ValueError:
            print('math domain error', a)
            return np.sqrt(float(a))
            
        
    def get_E(self,):
        self.energy = np.linspace(self.Det[0], self.Det[0] + self.Det[1]*len(self.meas_load), len(self.meas_load))


    def det_resp(self, Ec, E, T = 'Fluorescence'):
        if T == 'PU':
            s = self.get_det_width(Ec)*1.175
        if T == 'Escape':
            s = self.get_det_width(Ec)*1
        if T == 'Fluorescence':
            s = self.get_det_width(Ec)

        return 1/(np.sqrt(2*np.pi)*s) * np.exp(-((E - Ec)/s)**2 /2) * (4*s > np.abs(Ec-E)).astype(int)
        
        
    def pile_up_lines(self, linelist, Zlist):
        ''' need lineList  ##[{'I':1,'Z':29, 'edge':'K-line', 'g_sum':0.8, 
        'lines':[{'E':2.5, 'g':0.1},{'E':2.6, 'g':0.2},{}]},{...}]
        and Zlist = self.PU_Elements: [29,29,21,..], build [{E:g}{E:g}] '''
        # print('I am now looking for PU Lines')
        # print('PU Elements', self.PU_Elements)
        PUlines = []
        self.dead_time = self.real_time-self.life_time
        gating_time = self.gating_time
        
        self.sum_meas_load = np.sum(self.meas_load[200:])
        ###[200:] damit der Nullpeak nicht mitgenommen wird -> muss angepasst werden, wenn die Energiekanalbreite sich änder
        # print('self.real_time', self.real_time)
        # print('self.life_time', self.life_time)
        # print('self.dead_time', self.dead_time)
        if self.calc_PU:
            for iIdx in range(len(self.PU_Elements)):
                 len_linelist = len(linelist[iIdx]['lines'])
                 linelist.append({'I': 1, 'Z': linelist[iIdx]['Z'], 'edge': linelist[iIdx]['edge']+'+'+linelist[iIdx]['edge'], 'g_sum': 1, 'lines': []})
                 g = 0
                 for j1 in range(len_linelist): #pileups with themselves, 1. order
                     E1 = 2*linelist[iIdx]['lines'][j1]['E']
                     channel_j1 = int(np.round((-self.Det[0]+linelist[iIdx]['lines'][j1]['E'])/self.Det[1]))
                     if E1 < 41.84:
                         g1 = (linelist[iIdx]['lines'][j1]['g'])**2\
                             *gating_time*self.sum_meas_load
                         if (g1*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E6) > self.PU_threshold:
                             linelist[-1]['lines'].append({
                                 'E':E1,
                                 'g':g1,
                                 'type' : 'PU'})
                             g += g1
                     for j2 in range(j1, len_linelist): #pileup with different sub-lines of the same line, 1. order
                         if j1 != j2 and j1<j2:
                             E2 = linelist[iIdx]['lines'][j1]['E'] \
                                         +linelist[iIdx]['lines'][j2]['E']
                             if E2 < 41.84:
                                 g2 = (linelist[iIdx]['lines'][j1]['g']\
                                     *linelist[iIdx]['lines'][j2]['g']\
                                         *gating_time*self.sum_meas_load) ##1.5
                                 if (g2*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E6) > self.PU_threshold:
                                     linelist[-1]['lines'].append({
                                         'E':E2,
                                         'g':g2,
                                         'type' : 'PU'})
                                     g += g2

                     ##pileups with themselves, 2. order
                     # E3 = 3*linelist[iIdx]['lines'][j1]['E']
                     #
                     # if E3 < 41.84:
                     #     g3 = (linelist[iIdx]['lines'][j1]['g'])**3\
                     #         *gating_time*self.sum_meas_load
                     #     #if g3 > self.PU_threshold and (np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)>1E7:
                     #     if (g3*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E7) > self.PU_threshold:
                     #         linelist[-1]['lines'].append({
                     #             'E':E3,
                     #             'g':g3,
                     #             'type' : 'PU'})
                     #         g += g3
                     #
                     # for j2 in range(j1, len_linelist): #pileup with different sub-lines of the same line, 2. order
                     #     if j2 != j1 and j1<j2:
                     #         E4 = 2*linelist[iIdx]['lines'][j1]['E'] \
                     #                 +linelist[iIdx]['lines'][j2]['E']
                     #         if E4 < 41.84:
                     #             g4 = (linelist[iIdx]['lines'][j1]['g'])**2 \
                     #                 *linelist[iIdx]['lines'][j2]['g']\
                     #                     *gating_time*self.sum_meas_load
                     #             #if g4 > self.PU_threshold and (np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)>1E7:
                     #             if (g4*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E7) > self.PU_threshold:
                     #                 linelist[-1]['lines'].append({
                     #                     'E':E4,
                     #                     'g':g4,
                     #                     'type' : 'PU'})
                     #                 g += g4
                     #         E5 = 1*linelist[iIdx]['lines'][j1]['E'] \
                     #             +2*linelist[iIdx]['lines'][j2]['E']
                     #
                     #         if E5 < 41.84:
                     #             g5 = (linelist[iIdx]['lines'][j1]['g'] \
                     #                 *(linelist[iIdx]['lines'][j2]['g'])**2\
                     #                     *gating_time*self.sum_meas_load)
                     #             #if g5 > self.PU_threshold and (np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)>1E7:
                     #             if (g5*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E7) > self.PU_threshold:
                     #                 linelist[-1]['lines'].append({
                     #                     'E':E5,
                     #                     'g':g5,
                     #                     'type' : 'PU'})
                     #                 g += g5
                 linelist[-1]['g_sum'] = g*self.PU_factor
    # To Do: PU calculation of Lines from differen iIdx, needed if Ka and Kb separately or PU between different Elements are of interest
    # =============================================================================
            for kIdx in range(len(self.PU_Elements)):  #pileup with all other lines and element-lines
                 g = 0
                 iIdx=kIdx+1
                 if iIdx < len(self.PU_Elements):
                     if (linelist[iIdx]['edge'] != linelist[kIdx]['edge']) and linelist[kIdx]['Z'] == linelist[iIdx]['Z']:
                         linelist.append({'I': 1, 'Z': linelist[kIdx]['Z'], 'edge': '', 'g_sum': 1, 'lines': []})
                         for j in range(len(linelist[iIdx]['lines'])):
                             for l in range(len(linelist[kIdx]['lines'])):
                                 linelist[-1]['edge'] = linelist[kIdx]['edge']+'+'+linelist[iIdx]['edge']
                                 E6 = linelist[iIdx]['lines'][j]['E'] \
                                             + linelist[kIdx]['lines'][l]['E']
                                 channel_j1 = int(np.round((-self.Det[0] + linelist[iIdx]['lines'][j]['E']) / self.Det[1]))
                                 if E6 < 41.84:
                                     tmp_spec = self.get_spec()
                                     if ((linelist[iIdx]['lines'][j]['type'] != 'PU') and (linelist[kIdx]['lines'][l]['type'] == 'PU')):
                                         g6 = (linelist[kIdx]['lines'][l]['g']**2\
                                               * linelist[iIdx]['lines'][j]['g']\
                                               * gating_time * self.sum_meas_load)
                                     elif ((linelist[iIdx]['lines'][j]['type'] == 'PU') and (linelist[kIdx]['lines'][l]['type'] != 'PU')):
                                         g6 = (linelist[kIdx]['lines'][l]['g'] \
                                               * linelist[iIdx]['lines'][j]['g']**2\
                                               * gating_time * self.sum_meas_load)
                                     elif ((linelist[iIdx]['lines'][j]['type'] == 'PU') and (linelist[kIdx]['lines'][l]['type'] == 'PU')):
                                         g6 = (linelist[kIdx]['lines'][l]['g']**2 \
                                               * linelist[iIdx]['lines'][j]['g']**2 \
                                               * gating_time * self.sum_meas_load)
                                     else:
                                         g6 = (linelist[kIdx]['lines'][l]['g']\
                                             * linelist[iIdx]['lines'][j]['g']\
                                                 *gating_time*self.sum_meas_load)\
                                                  #*sum(tmp_spec)/self.life_time*gating_time)/1
                                     if (g6*(np.sum(self.meas_load[channel_j1-10:channel_j1+10])*self.life_time)/1E7) > self.PU_threshold:
                                         g += g6
                                         linelist[-1]['lines'].append({
                                             'E':E6,
                                             'g':g6,
                                             'type' : 'PU'})

                         linelist[-1]['g_sum'] = g*self.PU_factor
            for line in linelist:
                gSum = line['g_sum']
                for x in line['lines']:
                    x['g'] /= gSum
                line['g_sum'] = 1

# =============================================================================

        # Escape lines
        if self.calc_Escape:
            for _s in self.Lines:
                for _l in _s['lines']:
    #                if _l['E']<10 and _l['E']> 1.832:
                    if _l['E'] < 41.84 and _l['E']> xrl.EdgeEnergy(14, xrl.K_SHELL):
                        # empirical formula to determine escape possibilities:
                        g = (_s['I']*_l['g']/_s['g_sum']*(0.04344/_l['E'] - 0.003484))*gating_time*self.sum_meas_load*self.Escape_factor
                        if g > self.Escape_threshold:
                            _s['lines'].append({
                                'E':_l['E'] - 1.740,
                                'g': g,
                                'type' : 'Escape'})
        # print('linelist', linelist[iIdx])                
        # print('self.Lines', self.Lines)                
        # print('The PU Lines I found:', PUlines)
        # return PUlines


    def calc_M(self):
        ''' 
        This function calculates the Array M. M contains the normalized intensities
        of the gaussian on the correct energy position
        '''
        _s = -1
        energy = self.energy[self.Bins[0]: self.Bins[-1]+1]
        self.M = np.zeros((len(self.Lines)+len(self.user_defined_lines), len(energy)))
        ### to avoid calculating the same thing over and over while fitting
        ### plenty of spectra in specfit-GUI, M is only calculated once when 
        ### self.fit_in_progress = True
        for _s in range(len(self.Lines)):
            ### iterate over all transitions and create a gaussian
            ### distribution for every transition
            g_sum = self.Lines[_s]['g_sum']
            for _l in self.Lines[_s]['lines']:
                self.M[_s] += self.det_resp(_l['E'], energy, T = _l['type'])*_l['g']*self.Det[1]
        for i, udl in enumerate(self.user_defined_lines): ### user defined lines
            self.M[_s+i+1] += udl[self.Bins[0]: self.Bins[-1]+1] 
        self.M = self.M.swapaxes(0,1)
   

    def linfit(self, onedim = True):
        '''
        This function performs a linear fit with the given measurement parameter.
        No change of the parameter takes place.
        Procedure:
            1) calculate striped spectrum (striped_spec)
            2) calculate normalized gaussian (M)
            3) perform least square fit with M and the on ROI range shortened 
               striped spectrum (NetSpec)
            4) store the fitted intensities in the Lines dict (Lines) as results
        '''
        def residual_calibration(params, NetSpec, M):
                '''
                This function calculates the residual (the difference of fit and data)
                for the complete calibration
                '''
                M_shape = M.shape
                resid = np.zeros(NetSpec.shape)
                for i in range(M_shape[-1]):
                    resid = np.sum(NetSpec - np.sum(M*params.reshape(M.shape[:3]+(1,M.shape[4])), axis = -1),axis = -1)
                resid = np.nan_to_num(resid)
                return resid.flatten().astype(np.float64)
            
        if self.fit_in_progress == False:
            self.nl_det()
            self.calc_M()
        if onedim:
            self.NetSpec = self.meas_load[self.Bins[0]:self.Bins[-1]+1] - self.Strip[self.Bins[0]:self.Bins[-1]+1]
            lsq_linear_result = lsq_linear(self.M, self.NetSpec, bounds = (0, float('inf')))
        else:
            self.NetSpec = self.meas_load[...,self.Bins[0]:self.Bins[-1]+1] - self.Strip[..., self.Bins[0]:self.Bins[-1]+1]
            self.M = np.broadcast_to(self.M, self.NetSpec.shape[:-1]+self.M.shape)
            
                
            lsq_linear_result = least_squares(residual_calibration, 
                                   np.ones(self.M.shape[:3]+(self.M.shape[4],)).flatten().astype(np.float64), 
                                   jac = '3-point',
                                   method = 'dogbox',
                                   args = (self.NetSpec, self.M))
        I, resid = lsq_linear_result.x, lsq_linear_result.fun
        for _s in range(len(self.Lines)):
            self.Lines[_s]['I'] = I[_s]
        self.result_udl = []
        for i in range(len(self.user_defined_lines)):
            # self.result_udl.append(I[len(self.Lines)+i])
            self.result_udl.append(np.sum(self.M[:,len(self.Lines)+i])/I[len(self.Lines)+i])
        return resid


    def fit(self, minchange = None , full = True):
        '''nonlinear fit, changes parameters till change in std is smaller than minchange  '''
        if minchange == None: minchange = float(self.minchange)
        print('nonlinear fit, Parameters will change. params: [a0, a1, Fano, FWHM_0 ]', self.Det[:4])
        print('fitting-threshold: %s'%minchange)
        self.strip()
        self.Resid = self.linfit()
        resid_std = np.sqrt(np.divide(sum(np.square(self.Resid)),len(self.Resid),dtype=float))
#        oldresid = float(resid_std)
        oldresid = float(resid_std*1.1)
        iteration = 0
        while np.abs((oldresid - resid_std)/oldresid) > minchange: 
            iteration += 1
            if full: self.nl_fit()
            oldresid = float(resid_std)
            Resid = self.linfit()
            resid_std = np.sqrt(np.divide(sum(np.square(Resid)),len(Resid),dtype=float))
            print('%s. fit iteration, residchange = %s'%(iteration,( (oldresid - resid_std)/oldresid))) 
            print('[a0 , a1, fano, el. noise]')
            print(self.Det[:4])
            print('----------------------------')
        print('final parameters: ', self.Det)
        
        return self.Det, resid_std


#    @profile
    def calc_spec(self, print_results = False):
        '''return spectrum+background as a list '''
        spec = self.Strip + self.NLDet
        for _s in self.Lines:
            ### iterate over all transitions and create a gaussian
            ### distribution for every transition
            for _l in _s['lines']:
                if _l['type'] == 'PU':
                    spec += self.det_resp(_l['E'], self.energy, _l['type']) * _s['I'] * _l['g'] * self.PU_factor * self.Det[1]
                else:
                    spec += self.det_resp(_l['E'], self.energy, _l['type'])*_s['I']*_l['g']*self.Det[1]

        for udl,r_udl in zip(self.user_defined_lines,self.result_udl):
            spec += udl    
        if print_results:
            integral = np.trapz(spec-self.Strip-self.NLDet, x = np.arange(0, 4096, 1))
            self.get_result()
        return spec


#    @jit
    def nl_fit(self):
        '''internal function to perform just one non linear fit '''
        EBins = self.Det[0] + self.Det[1]*self.Bins
#        bounds = ([-2, 0, 0.1, 0.02, 0], 
#                  [1, 1, 0.12, 0.2, 1e9])
        bounds = ([-2, 0, 0.00, 0.00, 0],           # larger bounds necessary for ccd spectra, where Fano is increased due to PEE
                  [200, 100, 200.00, 200.00, 1e20])
        # if not (bounds[0][2] < self.Det[2] < bounds[1][2]):
        #     self.Det[2] = 0.101
        # if not (bounds[0][3] < self.Det[3] < bounds[1][3]):
        #     self.Det[3] = 0.03
        nl_fit_min = least_squares(self.residual, 
                                   self.Det, 
                                   jac = '3-point',
                                   method = 'dogbox',
                                   # bounds = bounds,
                                   args = (self.Meas, EBins))['x']
#        nl_fit_min = least_squares(self.residual, 
#                                   self.Det, 
#                                   jac = '3-point',
#                                   method = 'dogbox',
#                                   bounds = bounds,
#                                   args = (self.Meas, EBins))['x']
        self.Det = nl_fit_min


    def residual(self, par, meas, bins):
        '''calc residuum of meas and caculated spec '''
#        meas[meas == 0] = 1
        self.Det = par
        self.get_E()
        spec = self.calc_spec()[self.ROI[0]:self.ROI[1]]
        res = (meas - spec)
        return res


    def get_result_keys(self):
        '''return the Labels of the keys in correct order '''
        result_keys = []
        for line in self.Lines:
            try: result_keys.append('%s_%s'%(xrl.AtomicNumberToSymbol(line['Z']),line['edge']))
            except: result_keys.append('%s_%s'%(xrl.SymbolToAtomicNumber(line['Z']),line['edge']))
        return result_keys
