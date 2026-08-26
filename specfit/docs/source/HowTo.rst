.. _ref-how-to:

How To
======
Additionally to the Sobol sensitivity analysis, the class *ICDD_evaluation* in the **Evaluation_ICDD.py** script provides further evaluation methods.
Namely:

   * Kernel PCA: see `KernelPCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA>`__.
   * PCA: see `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA>`__.
   * PCA: see `TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE>`__.
   * PCA: see `UMAP <https://umap-learn.readthedocs.io/en/latest/>`__.

ICDD Evaluation Class
---------------------
The class ICDD_evaluation allows the loading of the data and different evaluation methods.

.. autoclass:: Evaluation_ICDD.ICDD_evaluation
    :members:
    :undoc-members: