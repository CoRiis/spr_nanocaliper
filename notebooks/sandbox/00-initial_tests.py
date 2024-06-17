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

# +
import numpy as np
import os

os.sys.path.append(os.path.dirname(os.path.abspath('../')))

from src.dsi23.utils.chemical_reactions import Reaction, Reactions, ODE


# +
# Test if the get_mol_r_stoich() method returns the correct number of molecules in the reactant.
def test_get_mol_r_stoich():
    r = Reaction("test", "2H2 + O2 > 2H2O")
    assert r.get_mol_r_stoich("H2") == 2
    assert r.get_mol_r_stoich("O2") == 1
    assert r.get_mol_r_stoich("H2O") == 0


# Test if the get_mol_net_stoich() method returns the correct number of molecules in the netto stoichiometry.
def test_get_mol_net_stoich():
    r = Reaction("test", "2H2 + O2 > 2H2O")
    assert r.get_mol_net_stoich("H2") == -2
    assert r.get_mol_net_stoich("O2") == -1
    assert r.get_mol_net_stoich("H2O") == 2


# Test if the get_molecule_names() method returns a set of all the molecules in the reaction.
def test_get_molecule_names():
    r = Reaction("test", "2H2 + O2 > 2H2O")
    assert r.get_molecule_names() == {"H2", "O2", "H2O"}


# Test if the __repr__() method returns the correct string representation of the Reaction object.
def test_repr():
    r = Reaction("test", "2H2 + O2 > 2H2O")
    assert repr(r) == "Name: test. Reaction string: 2H2 + O2 > 2H2O"


# Test if the string_to_reaction() method returns the correct dictionary of stoichiometric coefficients for a given reaction string.
def test_string_to_reaction():
    r = Reaction("test", "2H2 + O2 > 2H2O")
    assert r.string_to_reaction(
        "2H2 + O2 > 2H2O") == ({"H2": 2, "O2": 1}, {"H2O": 2, 'H2': -2, 'O2': -1})
    assert r.string_to_reaction(
        "H2 + O2 > H2O") == ({"H2": 1, "O2": 1}, {"H2O": 1, "H2": -1, "O2": -1})
    assert r.string_to_reaction(
        "2H2O > 2H2 + O2") == ({"H2O": 2}, {"H2": 2, "O2": 1, "H2O": -2})


def run_tests():
    test_get_mol_r_stoich()
    test_get_mol_net_stoich()
    test_get_molecule_names()
    test_repr()
    test_string_to_reaction()
    print("Passed")


run_tests()
# Run all the tests in this script
# if __name__ == '__main__':
#    pytest.main()

# +
def test_get_rates_as_dict():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    rates = {'R1': 0.5, 'R2': 1.0}
    expected_output = {'R1': 0.5, 'R2': 1.0}
    assert reactions.get_rates_as_dict(list(rates.values())) == expected_output


def test_get_rates_as_vec():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    rates = {'R1': 0.5, 'R2': 1.0}
    expected_output = np.array([0.5, 1.0])
    assert np.allclose(reactions.get_rates_as_vec(rates), expected_output)


def test_get_conc_as_dict():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    reactions.molecule_i = {'A': 0, 'C': 1, 'B': 2}
    conc = {'A': 1.0, 'C': 2.0, 'B': 0.5}
    expected_output = {'A': 1., 'C': 2., 'B': .5}
    assert reactions.get_conc_as_dict(list(conc.values())) == expected_output


def test_get_conc_as_vec():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    reactions.molecule_i = {'A': 0, 'C': 1, 'B': 2}
    conc = {'A': 1.0, 'C': 2.0, 'B': 0.5}
    expected_output = np.array([1.0, 2.0, 0.5])
    assert np.allclose(reactions.get_conc_as_vec(conc), expected_output)


def run_tests():
    test_get_rates_as_dict()
    test_get_rates_as_vec()
    test_get_conc_as_dict()
    test_get_conc_as_vec()
    print("Passed")


run_tests()


# +
def test_get_dyn_conc_as_vec():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)

    # Setting dyn_molecule_i manually because the class is randomly shuffling the order..
    ode.dyn_molecule_i = {"A": 0, "B": 1, "C": 2}

    conc_dict = {"A": 1.0, "B": 2.0}
    conc_vec = ode.get_dyn_conc_as_vec(conc_dict)
    assert np.allclose(conc_vec, np.array([1.0, 2.0, 0.0]))


def test_get_dyn_conc_as_dict():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)

    # Setting dyn_molecule_i manually because the class is randomly shuffling the order..
    ode.dyn_molecule_i = {"A": 0, "B": 1, "C": 2}

    conc_vec = np.array([1.0, 2.0, 3.0])
    conc_dict = ode.get_dyn_conc_as_dict(conc_vec)
    assert conc_dict == {"A": 1.0, "B": 2.0, "C": 3.0}


def test_get_dyn_rates_as_vec():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)
    rates_dict = {"R1": 0.5}
    rates_vec = ode.get_dyn_rates_as_vec(rates_dict)
    assert np.allclose(rates_vec, np.array([0.5, 0]))


def test_get_dyn_rates_as_dict():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)
    rates_vec = np.array([0.5])
    rates_dict = ode.get_dyn_rates_as_dict(rates_vec)
    assert rates_dict == {"R1": 0.5, "R2": 0.0}


def test_add_concentration_function():
    reactions = Reactions()
    ode = ODE(reactions)
    ode.add_concentration_function("A", lambda t: 1.0)
    assert ode.concentration_functions == {}

    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)
    ode.add_concentration_function("A", lambda t: 1.0)
    assert "A" in ode.concentration_functions


def test_add_reaction_function():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)
    ode.add_reaction_function("r1", lambda t: 0.5)
    assert ode.reaction_functions == {}

    ode = ODE(reactions)
    ode.add_reaction_function("R1", lambda t: 0.5)
    assert "R1" in ode.reaction_functions


def test_dy_dt_np():
    reactions = Reactions()
    reactions.add('R1', 'A -> B')
    reactions.add('R2', '2B -> C')
    ode = ODE(reactions)

    y = ode.get_dyn_conc_as_vec({"A": 0, "B": 1, "C": 2})
    t = 0
    rates = ode.get_dyn_rates_as_vec({"R1": 0, "R2": 2})
    dydt = ode.dy_dt_np(y, t, *rates)
    assert ode.get_dyn_conc_as_dict(dydt) == {"A": 0, "B": -4, "C": 2}


def run_tests():
    test_get_dyn_conc_as_vec()
    test_get_dyn_conc_as_dict()
    test_get_dyn_rates_as_vec()
    test_get_dyn_rates_as_dict()
    test_add_concentration_function()
    test_add_reaction_function()
    test_dy_dt_np()


run_tests()
print("Passed")
# -


