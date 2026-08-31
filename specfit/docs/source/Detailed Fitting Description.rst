.. _ref-detailed-fitting-description:

Sensitivity (Sobol) Analysis
============================
The Sobol method is a variance-based sensitivity analysis method. It is used to determine the contribution of each input parameter to the output variance. The Sobol method decomposes the output variance into contributions from each input parameter and their interactions. This allows for a comprehensive understanding of how input parameters influence the output.
The Sobol method is particularly useful for complex models with multiple input parameters, as it can identify.

To perform a Sobol sensitivity analysis, the class *ICDD_evaluation* in the **Evaluation_ICDD.py** script is utilised.
The class is initiated

   .. code-block:: python

      ICDD = ICDD_evaluation()

With the initiated class all base data has to be read out and registered in the class in order to further analyse the sensitivity.

   .. code-block:: python

      ICDD.readout_base_data(
         limit_sample=samples,
         limit_parameter=parameters,
         element=element,
         min_conc=1e-6,
         mA=0.1,
         normalisation="sum",
         batchsize=100
         )

Now the specific data for the sensitivity analysis can be read out with the following commands:

   .. code-block:: python

      ICDD.readout_data(entry="kvs")
      ICDD.readout_data(entry="mas")
      ICDD.readout_data(entry="atms")
      ICDD.readout_data(entry="sources")
      ICDD.readout_data(entry="filters")

Now all the data is available for analysis and the Sobol sensitivity analysis can be performed:

   .. code-block:: python

      ICDD.all_sobol(
         plot_intens=True,
         plot_sobol=True
         )

.. _ref-sensitivity_analysis:

.. figure:: _static/images/20260629_ST_plot.png
   :width: 100%
   :align: center

   Figure 1: Plotted are the sensitivities of the different input parameters (red-kv, blue-filter, green-source, black-atm) for the specific elements (given as atomic numbers).

In :ref:`Figure 1 <ref-sensitivity_analysis>` the evaluated sensitivities of the different input parameters (red-kv, blue-filter, green-source, black-atm) for the specific elements (given as atomic numbers).
The dashed black vertical line indicates the switch from K-line to L-line fluorescence simulation. The influence of the tube voltage is depicted in red (kv), the source filters in blue (filters), the tube target material in green (source) and the atmosphere in black (atm). The atmosphere has a high influence for low energy fluorescence due to absorption in air. The atmosphere is dominant for elements below Argon and decreases fast for heavier elements. It can also be observed that for lower fluorescence line energy the filter is dominantly influential. This is true until about 8 keV where the target material is influential. The reason are the fluorescence energies of the target material Tungsten (W) at around 8 keV making fluorescence generation very efficient in this energy regime. The filters are getting dominant from 14 keV to 20 keV where the highest fluorescence energy from the target material Rh is located. After this the fluorescence is only excited by the bremsstrahlung, which is highly dependent by the tube acceleration voltage, which explains the rising influence of the voltage.
The mean of all parameters is plotted as a horizontal dashed line in the corresponding color.

SpecFit deconvolution
---------------------
Listed are the functions of the specfit_deconvolution Class

.. autoclass:: functions.specfit_deconvolution.SpecFit
    :members:
    :undoc-members: