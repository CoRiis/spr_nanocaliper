# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (stable)
#     language: python
#     name: stable
# ---

# %% [markdown] tags=[]
# # RP0000/LI0000(ProjectNumber): TITLE

# %% [markdown] tags=[]
# ## Summary
# Executive summary goes here
#
# __Objectives__:
#
# * BLA BLA
#
# __Author__: AAAA, BBBB & CCCC
#
# __Modelling__:
#  - Input: 
#  - Output: 
#
# __Background material__:

# %% [markdown]
# ## Imports and Parameter Configurations
#
# ### Project/User Configurations

# %%
project_no = dsi23

# %% [markdown]
# ### Library Imports
# Both internal and external modules importes

# %%
# %load_ext autoreload
# %autoreload 2

import sys, os
sys.path.append(f'../../src/{project_no}/')
import novopy
from novodataset.dataset import DataSet, SAR4MLDataSet, DesignSet

# %% [markdown]
# ### Helper Functions
# In this section goes custom ad-hoc functions. Keeping it here ensures a clean notebook and makes it easier for refactoring at later stage in the process.

# %% [markdown]
# ## Data Import

# %%

# %% [markdown]
# ## Analysis Sections 

# %%

# %% [markdown]
# ## Conclusion & Future Directions

# %% [markdown]
#

# %% [markdown]
# ## AUTHOR & DATE WATERMARK
# Auto update of the user executing the notebook with timestamps, and git repo (if present) locked.
# Author here corresponds to the user executing the notebook and might be different from the list of authors in the top taking part in the development of the notebook.

# %%
user = novopy.services.utils.get_username(on_domino=True)

# %%
# %reload_ext watermark
# %watermark --author $user --time --date --timezone --python --updated --iversion --gitrepo --gitbranch
