 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:49:28 2018

@author: frank
"""
''' This function creates an energy data-file called lineE.dat' it contains
the fluorescence energies for elements in the range of Z = [11,98] and E = [0,50].
The data file is used to select an element-line directly from the plot.'''

import traceback
import xraylib as xrl
import numpy as np

### determine lines who are interesting
lines = {'Ka' : xrl.KA1_LINE,
         'Kb' : xrl.KB1_LINE,
         'L1' : xrl.L1M3_LINE,
         'L2' : xrl.L2M4_LINE,
         'L3' : xrl.L3M5_LINE}
elements = np.arange(1,98)

energies = {}

for line in lines:
    for element in elements:
        if (element>97) or (element<11):
            continue
        else:
            energy = xrl.LineEnergy(element, line)
            if energy < 50:
                energies[energy] = [element, elements[element], lines[line], line, xrl.CS_FluorLine_Kissel_Cascade(element, line, 20.167)]

with open('lineE.dat', 'w') as tofile:
    for key, value in energies.iteritems():
        tofile.write('%f\t%s\t%d\t%s\t%d\t%f\n'%(key, value[0], value[1], value[3], value[2], value[4]))