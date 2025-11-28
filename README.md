Surface Plasmon Resonance fitter
==============================

Do not upload Jupyter notebooks to the git repository. Instead do the following when working with notebooks:

Before push: ``jupytext --to py notebooks/*.ipynb``

After pull: ``jupytext --to ipynb notebooks/*.py``

