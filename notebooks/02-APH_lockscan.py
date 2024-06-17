#!/usr/bin/env python
# coding: utf-8
# +
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import lognormal

os.sys.path.append(os.path.dirname(os.path.abspath('.')))

from src.dsi23.utils.chemical_reactions import Reactions, ODE, get_located_fun
from src.dsi23.utils.sprfitter import SPRFitter, bound_lognormal
from src.dsi23.utils.data import read_SPR_data
from src.dsi23.utils.viz import plot_curves

# %load_ext autoreload
# %autoreload
# -


USE='np'


model_name = 'model_1'

# +
relative_path = '../'
data_path = relative_path + 'data/raw/'
results_path = relative_path + 'results/' + model_name + '/'
output_folder = results_path + 'IgG4_Ab1/'

init_ass = 0.00018
dis = 0.0017
cutoff = 400.0
n_optimization_restarts = 1
verbose = True  # plot

fileconfigs_IgG4_Ab1 = [
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0001.ipynb',
        'filename': 'APh6 0.3 Ab0001.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0001_2_c020323.ipynb',
        'filename': "APh6 to ab0001 IgG1_2_c020323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0001_3_c020323.ipynb',
        'filename': "APh6 to ab0001 IgG1_3_c020323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0001_1_newgoodchip_280323.ipynb',
        'filename': "APh12 0.5 to ab0001 IgG4_1_new chip 280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0001_2_newgoodchip_280323.ipynb',
        'filename': "APh12 0.5 to ab0001 IgG4_2_new chip 280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0001_3_28032.ipynb',
        'filename': "APh12 0.5 to ab0001 IgG4_3_280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0001_1_c010523.ipynb',
        'filename': "APh18 0.1 to ab0002 IgG4_1_c010523.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0001_2_c010523.ipynb',
        'filename': "APh18 0.1 to ab0002 IgG4_2_c010523.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0001_3_c010523.ipynb',
        'filename': "APh18 0.1 to ab0002 IgG4_3_c010523.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    }
]

fileconfigs_IgG4GS_Ab5 = [
    # IgG4GS_Ab5
    # APH6
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0005.ipynb',
        'filename': 'APh6 0.3 Ab0005.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0005_3_c020323.ipynb',
        'filename': "APh6 to ab0005 IgG4GS_3_c020323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0005_2_c020323.ipynb',
        'filename': "APh6 to ab0005 IgG4GS_2_c020323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    # APH12
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0005_4_28032.ipynb',
        'filename': "APh12 0.5 to ab0005 IgG4GS_4_new chip 280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0005_2_280323.ipynb',
        'filename': "APh12 0.5 to ab0005 IgG4GS_2_280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0005_3_28032.ipynb',
        'filename': "APh12 0.5 to ab0005 IgG4GS_3_280323.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    # AP18
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0005_firstdata.ipynb',
        'filename': "APH180.42_ab0005.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0004GS_2_c010523.ipynb',
        'filename': "APh18 0.1 to ab0005 IgG4GS_2_c010523.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0004GS_4_c01052.ipynb',
        'filename': "APh18 0.1 to ab0005 IgG4GS_4_c010523.xlsx",
        'cutoff': cutoff,
        'init_ass': init_ass
    }
]

fileconfigs_IgG1_Ab2 = [
    # APH6
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0002.ipynb',
        'filename': 'APH6 0.05nM to Ab0002.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0002.ipynb',
        'filename': 'Extracted data biacore APH6 0.42 in combination with bivalent AB.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0002_3_c020323.ipynb',
        'filename': 'APh6 to ab0002 IgG1_3_c020323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0002_4_c020323.ipynb',
        'filename': 'APh6 to ab0002 IgG1_4_c020323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH6-Ab0002_5_c020323.ipynb',
        'filename': 'APh6 to ab0002 IgG1_5_c020323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    # APH12
    {
        'notebook': 'test_locking_JFGB-APH12.ipynb',
        'filename': 'APh12 400ass time biacore data.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0002_2_newgoodchip_280323.ipynb',
        'filename': 'APh12 0.5 to ab0002 IgG1_2_new chip 280323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0002_3_2_newgoodchip_280323.ipynb',
        'filename': 'APh12 0.5 to ab0002 IgG1_3_2_new chip 280323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0002_5_newgoodchip_280323.ipynb',
        'filename': 'APh12 0.5 to ab0002 IgG1_5_new chip 280323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH12-Ab0002_4_newgoodchip_280323.ipynb',
        'filename': 'APh12 0.5 to ab0002 IgG1_4_new chip 280323.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    # APH18
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0002_2_c010523.ipynb',
        'filename': 'APh18 0.1 to ab0002 IgG1_2_c010523.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0002_3_c010523.ipynb',
        'filename': 'APh18 0.1 to ab0002 IgG1_3_c010523.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
    {
        'notebook': 'test_locking_JFGB-APH18-Ab0002_4_c010523.ipynb',
        'filename': 'APh18 0.1 to ab0002 IgG1_4_c010523.xlsx',
        'cutoff': cutoff,
        'init_ass': init_ass
    },
]

fileconfigs = fileconfigs_IgG4_Ab1

print(len(fileconfigs))
# -


if not os.path.exists(results_path):
        os.mkdir(results_path)


# +
class ModelConfig1:
    def add_kinetics(self, kinetics):
        kinetics.add("ass", "A+B -> AB1")
        kinetics.add("dis", "AB1 -> A+B")
        kinetics.add("avid_ass", "AB1 -> AB2")
        kinetics.add("avid_dis", "AB2 -> AB1")
        kinetics.add("ass2", "AB1+A -> A2B2")
        kinetics.add("dis2", "A2B2 -> A+AB1")

    def configure_sprfitter(self, sprfitter, dis):
        sprfitter.add_molecule_to_predict('AB1', 1)
        sprfitter.add_molecule_to_predict('A2B2', 2)
        sprfitter.add_molecule_to_predict('AB2', 1)
        sprfitter.fix_rate('dis', 0.0017)

    def get_guesses(self, ass, dis):
        return {'ass': ass, 'dis': dis, 'avid_ass': 100 *
                            0.002, 'avid_dis': 100 * 0.002, 'ass2': 0.0002, 'dis2': 0.002}


class ModelConfig2:
    def add_kinetics(self, kinetics):
        kinetics.add("ass", "A+B -> AB")
        kinetics.add("dis", "AB -> A+B")
        kinetics.add("cross_ass", "AB+B -> AB2")
        kinetics.add("cross_dis", "AB2 -> AB+B")

    def configure_sprfitter(self, sprfitter, dis):
        sprfitter.add_molecule_to_predict('AB', 1)
        sprfitter.add_molecule_to_predict('AB2', 1)
        sprfitter.fix_rate('dis', dis)

    def get_guesses(self, ass, dis):
        return {'ass': ass, 'dis': dis, 'cross_ass': 10 * ass, 'cross_dis': 2 * dis}


class ModelConfigFactory:
    @staticmethod
    def get_model_config(model_name):
        if model_name == 'model_1':
            return ModelConfig1()
        elif model_name == 'model_2':
            return ModelConfig2()
        else:
            print(f"Model {model_name} is not defined.")
            return None


# -

# exp_i = 7
fc_lst = [fileconfigs_IgG1_Ab2]  # , fileconfigs_IgG4_Ab1, fileconfigs_IgG4GS_Ab5]
# ,'Insilico_scan_results/NATW_model/IgG4_Ab1/','Insilico_scan_results/NATW_model/IgG4GS_Ab5/']
of_lst = [results_path + 'IgG1_Ab2']
for fileconfigs, output_folder in zip(fc_lst, of_lst):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for exp_i in range(len(fileconfigs[:1])):
        all_data = read_SPR_data(data_path + fileconfigs[exp_i]['filename'])

        # remove lowest concentration!
        all_removed = False
        while not all_removed:
            remove_i = None
            for i, d in enumerate(all_data):
                # print("doing {}".format(i))
                if d.metadata['concentration'] < 12:
                    remove_i = i
                    break
            if remove_i is None:
                all_removed = True
            else:
                all_data.pop(remove_i)
        for i, d in enumerate(all_data):
            print("Concentration in sample {0} is {1}".format(
                i, d.metadata['concentration']))

        nmdls = len(all_data)

        if verbose:
            print(nmdls)
            plot_curves(all_data, fileconfigs[exp_i]['cutoff'])

        kinetics = Reactions(use=USE)
        model = ModelConfigFactory.get_model_config(model_name)
        model.add_kinetics(kinetics)

        mdls = [ODE(kinetics) for s in range(0, nmdls)]
        for i in range(0, nmdls):
            mdls[i].add_concentration_function("A", get_located_fun(
                [0., fileconfigs[exp_i]['cutoff']], 0.1, all_data[i].metadata["concentration"], use=USE))

        sprfitter = SPRFitter(all_data, mdls)
        model.configure_sprfitter(sprfitter, dis)

        mdl = mdls[0]
        ass = fileconfigs[exp_i]['init_ass']
        guess_rates_dict = model.get_guesses(ass, dis)
        max_signal = np.max([np.max(data.curve.y) for data in all_data])

        sprfitter.print_every = 1000
        filename = os.path.join(
            output_folder, fileconfigs[exp_i]['filename'][:-5].replace(' ', '_') + '.txt')
        if not os.path.exists(filename):
            header = 'chi2\tB_initial\tB_final'
            for k in guess_rates_dict.keys():
                header = header + '\t' + k + '_initial\t' + k + '_final'
            with open(filename, 'w') as file:
                file.write(header + '\n')

        for i in range(n_optimization_restarts):
            this_guess = dict()
            for k, v in guess_rates_dict.items():
                thisbounds = None if k not in sprfitter.bound_rates else sprfitter.bound_rates[k]
                sigma = np.log(
                    100) / 3 if thisbounds is None else (np.log(thisbounds[1]) - np.log(thisbounds[0])) / 3
                if model_name == 'model_1':
                    this_guess[k] = bound_lognormal(np.log(v), sigma, thisbounds)
                elif model_name == 'model_2':
                    this_guess[k] = bound_lognormal(np.log(v) * 2, sigma, thisbounds)
                else:
                    print('Model not defined')
                    raise NotImplementedError
            guess_B = lognormal(np.log(max_signal), np.log(1.1), sprfitter.bound_B)
            print('============')
            print('Iteration {0}'.format(i))
            print('------------')
            res = sprfitter.minimize(
                guess_B, this_guess, method='Nelder-Mead', options={'maxfev': 400})
            final_B = res.get('B')
            final_rates = res.get('rates_dict')
            chi2 = res.get('chi2')
            s = str(chi2) + '\t' + str(guess_B) + '\t' + str(final_B)
            for k in guess_rates_dict.keys():
                s = s + '\t' + str(this_guess[k]) + '\t' + str(final_rates[k])
            with open(filename, 'a') as file:
                file.write(s + '\n')
