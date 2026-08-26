.. SpecFit documentation master file

Documentation for the X-ray Fluorescence Deconvolution Software **SpecFit**
===========================================================================
This is the documentation of the X-ray fluorescence (XRF) deconvolution software
`SpecFit <https://github.com/frankfoerste/specfit/>`__. SpecFit is an
advanced deconvolution tool to evaluate up to 3D XRF datasets. The evaluated
results are returned in a structured *.hdf5* file containing the evaluate
fluorescence intensities for each selected fluorescence line.

It supports various data formats ranging from widely used files lice *.MCA* or
structured files like *.hdf5* to company specific formats like *Bruker*'s
*.bcf* or *.spx* file. The complete list of currently supported files can be
found under :ref:`supported files <ref-Installation>`.

This documentation covers the overall functionality of SpecFit and a clear 
code documentation.

Technical overviews of the functions provided are listed.

For the installation of SpecFit please refer to the :ref:`Installation <ref-Installation>`
section.

For an overview of the functionality and purpose of **SpecFit** please refer
to :ref:`Overview and Purpose <ref-overview>`.

A hands-on fitting procedure cook-book can be found under :ref:`HowTo <ref-how-to>`.

A detailed description of the GUI functionalities can be found under :ref:`Detailed GUI Description <ref-detailed-description>`.

For a detailed description of the fitting procedures, please refer to :ref:`Detailed fitting description <ref-detailed-fitting-description>`.

A list of supported file formats can be found under :ref:`Supported Formats <ref-supported-formats>`.

The content of the documentation is listed here:

Contents
--------

.. toctree::
   :maxdepth: 2

   How to install SpecFit <Installation>
   How to install the virtual Python environment <Installation Python>
   Overview and Purpose <Overview and Purpose>
   HowTo <HowTo>
   Detailed GUI Description <Detailed Description>
   Detailed Fitting Description <Detailed Fitting Description>
   Supported Formats <Supported Formats>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
