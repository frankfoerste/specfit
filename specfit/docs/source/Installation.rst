.. _ref-Installation:

Installation
============
To install SpecFit please perform the following steps:

1. For the best expirience You need to have Git installed on your system. You
can download an installer from here `Git Installer <https://git-scm.com/install/windows>`__.
You also need a valid python installation on your system. The easiest way to
install python and manage your python environments is using a Python
distribution like `Anaconda <https://repo.anaconda.com/archive/>`__ or likewise.

2. Open a command line programm (Command Prompt) and move to your repository
folder:

   .. code-block:: bash

      cd move/to/repository/folder

3. Download the repository specfit from the GitHub repository under `SpecFit GitHub repository <https://github.com/frankfoerste/specfit.git>`__
using the following command

   .. code-block:: bash

      git clone https://github.com/frankfoerste/specfit.git

4. Prior installing the repository, make sure you are in the desired Python
environment. See the :ref:`ref-Installation-Python` section for detailed Python
installation details. If required activate the **(specfit)** Python environment,
change into the specfit directory and install the module using PIP:

   .. code-block:: bash

      conda activate specfit
      cd specfit
      pip install -e .

This automatically installs all required dependencies of SpecFit. After the
installation is completed you have successfully set up SpecFit.