.. _ref-detailed-description:

Detailed GUI Description
========================

This document describes the different aspects of the GUI and their functionalities.
For a detailed description of the fitting process please refer to :ref:`detailed fitting description <ref-detailed-fitting-description>`.

.. _ref-menus:
.. list-table:: List of Menus
   :widths: 15 85
   :header-rows: 0
   :class: action-icons

   * - .. image:: _static/icons/folder-blue-open-icon.png
          :width: 32px
     - **Open Folder**
       
       Opens a folder containing spectra.

   * - .. image:: _static/icons/file-open.png
          :width: 32px
     - **Open File**
       
       Opens a spectrum or composite file.

   * - .. image:: _static/icons/angle-open.png
          :width: 32px
     - **Open Angle Resolved Measurement**
       
       Opens an angle resolved measurement, e.g. GE-XRF or GI-XRF.

   * - .. image:: _static/icons/batch_fitting.png
          :width: 32px
     - **Batch Fitting**
       
       Fitting of multiple files and folders with the same fitting parameters.

   * - .. image:: _static/icons/settings-open.png
          :width: 32px
     - **Open Settings File**
       
       Load a settings file to automatically set fitting parameters.

   * - .. image:: _static/icons/settings-save.png
          :width: 32px
     - **Save Settings File**
       
       Save the set of fitting parameters.

   * - .. image:: _static/icons/check-fit.png
          :width: 32px
     - **Check Fit**
       
       Test the fitting parameters on the loaded spectrum.

   * - .. image:: _static/icons/clear-plot.png
          :width: 32px
     - **Clear Fit**
       
       Clears the plotted fit to display the original spectrum.

   * - .. image:: _static/icons/clear-elements.png
          :width: 32px
     - **Clear Elements**
       
       Clears the element selection.

   * - .. image:: _static/icons/show-ROI.png
          :width: 32px
     - **Show Energy ROI**
       
       Opens the energy region of interest (ROI) investigation window.

   * - .. image:: _static/icons/plot-3d.png
          :width: 32px
     - **Plot 3D**
       
       Opens the overview 3D plot window.

   * - .. image:: _static/icons/fit-and-save.png
          :width: 32px
     - **Fit and Save**
       
       Starts the fitting procedure and stores the evaluated intensities.

   * - .. image:: _static/icons/bug.png
          :width: 32px
     - **Debugging Shell**
       
       Opens a debugging IPython shell with access to **SpecFit** parameters.

   * - .. image:: _static/icons/exit.png
          :width: 32px
     - **Exit**
       
       Closes the application.

.. _ref-actions:
.. list-table:: List of Actions
   :widths: 15 85
   :header-rows: 0
   :class: action-icons

   * - .. image:: _static/icons/folder-blue-open-icon.png
          :width: 32px
     - **Open Folder**
       
       Opens a folder containing spectra.

   * - .. image:: _static/icons/file-open.png
          :width: 32px
     - **Open File**
       
       Opens a spectrum or composite file.

   * - .. image:: _static/icons/angle-open.png
          :width: 32px
     - **Open Angle Resolved Measurement**
       
       Opens an angle resolved measurement, e.g. GE-XRF or GI-XRF.

   * - .. image:: _static/icons/batch_fitting.png
          :width: 32px
     - **Batch Fitting**
       
       Fitting of multiple files and folders with the same fitting parameters.

   * - .. image:: _static/icons/settings-open.png
          :width: 32px
     - **Open Settings File**
       
       Load a settings file to automatically set fitting parameters.

   * - .. image:: _static/icons/settings-save.png
          :width: 32px
     - **Save Settings File**
       
       Save the set of fitting parameters.

   * - .. image:: _static/icons/check-fit.png
          :width: 32px
     - **Check Fit**
       
       Test the fitting parameters on the loaded spectrum.

   * - .. image:: _static/icons/clear-plot.png
          :width: 32px
     - **Clear Fit**
       
       Clears the plotted fit to display the original spectrum.

   * - .. image:: _static/icons/clear-elements.png
          :width: 32px
     - **Clear Elements**
       
       Clears the element selection.

   * - .. image:: _static/icons/show-ROI.png
          :width: 32px
     - **Show Energy ROI**
       
       Opens the energy region of interest (ROI) investigation window.

   * - .. image:: _static/icons/plot-3d.png
          :width: 32px
     - **Plot 3D**
       
       Opens the overview 3D plot window.

   * - .. image:: _static/icons/fit-and-save.png
          :width: 32px
     - **Fit and Save**
       
       Starts the fitting procedure and stores the evaluated intensities.

   * - .. image:: _static/icons/bug.png
          :width: 32px
     - **Debugging Shell**
       
       Opens a debugging IPython shell with access to **SpecFit** parameters.

   * - .. image:: _static/icons/exit.png
          :width: 32px
     - **Exit**
       
       Closes the application.
      


In :ref:`Figure 1 <ref-GUI>` the GUI for the fluorescence transformation is displayed. To transform fluorescence load the fluorescence.

.. _ref-GUI-load:
.. figure:: _static/images/factor_transformation_GUI_load.png
   :align: center
   :width: 100%

   Figure 2: Displayed is the loading instance in the GUI for the Factor Transformation.

When the button is pressed, a window pops up (see :ref:`Figure 2 <ref-GUI-load>`). By entering an evaluations ID the XRFDB is queried and possible Evaluations are displayed. Select on an verify with OK.
The conditions are loaded automatically.
!!WARNING
The target is not stored in any document, so it is read out from the ID. If this fails, it has to be selected manually!!
WARNING!!
The new conditions should be entered. When finished, the selected conditions should be verified and the fluorescence transformed by pressing "transform fluorescence".
The transformed fluorescence intensities are plotted alongside the loaded intensities.
