.. _ref-Installation-Python:

Python Installation
===================
To install Python here to most convenient approach with Anaconda is explained.
You can download an installer for your OS via this webpage: `Anaconda Installer <https://repo.anaconda.com/archive/>`__.

After downloading please install Anaconda on your system.

We will install a virtual Python environment for the XRSAI module. In order to
do this please open a new instance of a command line program.

You will see **(base)** at the front of the line. This is the Python environment
which you are at the moment running on. We will install the **(specfit)**
environment. You install a new virtual environment with

   .. code-block:: bash

      conda create --name specfit python=3.13

Once the installation is completed, you can change the Python environment to
**specfit** using

   .. code-block:: bash

      conda activate specfit

The block in the beginning of the line should now show **(specfit)** and you
successfully set up the Python requirements to run SpecFit.

You can proceed to install the SpecFit package under :ref:`Installation <ref-Installation>`.