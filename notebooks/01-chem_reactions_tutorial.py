# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Author: NCIR
#
# # Walkthrough of classes in chem_reactions.py
#
# This notebook covers the three classes `Reaction`, `Reactions`, and `ODE`from `chem_reactions.py`

# +
import os

os.sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.dsi23.utils.chemical_reactions import Reaction, Reactions, ODE

# %load_ext autoreload
# %autoreload
# -

# ### Reaction
# This class defines a single chemical reaction.

# +
# Each reaction must have a name and string describing the reaction.
r = Reaction(name='ass', reaction_string='A + B -> AB')

# We can then access the name and the reaction string:
print("Name:", r.name)
print("Reaction string:", r.reaction_string)

# If a reaction_string with a reaction symbol '>' is given ,the reaction class also computes the reactants and the netto stoichiometry
print("Reactants:", r.r_stoich)
print("Netto stoichiometry:", r.net_stoich)

print("\nIf no reaction symbol:")
reaction_string = 'A + B'
r = Reaction(name='reactant', reaction_string=reaction_string)
print("Reaction string:", reaction_string)
print("Name:", r.name)
print("Reaction string:", r.reaction_string)
print("Reactants:", r.r_stoich)
print("Netto stoichiometry:", r.net_stoich)

# +
# From the Reaction, we can extract the number of molecules by name
r = Reaction(name='ass', reaction_string='A + B -> AB')

# Get molecules in the reactions
print('get_molecules_names:', r.get_molecule_names())

print('\nGet the number of molecules of the reactants')
# Get number of molecules of the reactants
for m in r.get_molecule_names():
    print(m, ':', r.get_mol_r_stoich(m))
    
print('\nGet the number of molecules of the netto stoichiometry')
# Get number of molecules of the reactants
for m in r.get_molecule_names():
    print(m, ':', r.get_mol_net_stoich(m))
# -

# ### Reactions
#
# This class provides an easy way to keep track of multiple reactions

# Initializing the class creates three dictionaries in the class
rs = Reactions()
rs

# We can add a reaction to the reactions
rs.add(name='ass', reaction_string='A+B->AB')
rs

# and another one
rs.add(name='dis', reaction_string='AB->A+B')
rs

# The class make sure, we do not override anything or add duplicates
# Here we add a reaction with name "dis", but it is not added since there
# already is a reaction with the name "dis".
rs.add(name='dis', reaction_string='CD->C+D')
rs

# An overall overview of the stoichiometry
rs.print_stochiometry()

# There are two set of get-functions in the class:
#
#
#

help(rs.get_conc_as_dict)

help(rs.get_conc_as_vec)

# Examples:

rs.get_conc_as_dict([0.1, 0.2])

rs.get_conc_as_vec({'A': 1})

# Likewise, we have the following get-functions for the rates:

rs.get_rates_as_dict([0.1, 0.2])

rs.get_rates_as_vec({'ass': 0.1, 'new': 0, 'dis': 0.5})

# We can also get a single reaction by its name

rs['ass']

# The number of reactions

len(rs)

# Iterate over the reactions

for r in rs:
    print(r)

# ### ODE

# +
# %load_ext autoreload
# %autoreload

from src.dsi23.utils.chemical_reactions import Reaction, Reactions, ODE
# -

ode = ODE(reactions=rs)
ode


