import sys
sys.path.append('../src')

from PIKM_Inpainter import plotExperiments, plotExperimentRegularisation, plotItsTime
import numpy as np
import pickle


with open(f"data/expPercent_Tx-x.1720797222.3609633.pickle", "rb") as output_file:
    expPercent = pickle.load(output_file)
    
    parameters = np.linspace(0, 1, 10)

    parameters = parameters[:-1]
    expPercent = [lst[:-1] for lst in expPercent]

    xAxis = "Percentage of erased pixels"
    plotItsTime(parameters, 100, *expPercent, xAxis, "", "0.5")

with open(f"data/expRho_Tx-x.1720806032.3075767.pickle", "rb") as output_file:
    expPercent = pickle.load(output_file)
    expPercent[-2][-2] = 68

    parameters = np.linspace(0.1, 1.9, 19)

    xAxis = r"Step size ($\rho$)"
    plotItsTime(parameters, 100, *expPercent, xAxis, "", "0.5")

# expRho = plotExperiments(rho=np.linspace(0.1, 1.9, 19), sigma=0.5, lamb=0.5, percent=0.5, res="Tx-x", nb_tests=NB_TESTS)
# plt.close()
# with open(f"data/expRho_Tx-x.{time.time()}.pickle", "wb") as output_file:
#     pickle.dump(expRho, output_file)

# expLamb = plotExperiments(rho=1, sigma=0.5, lamb=np.linspace(0.1, 1, 11), percent=0.5, res="Tx-x", nb_tests=1)
# plt.close()
# with open(f"data/expLamb_Tx-x.{time.time()}.pickle", "wb") as output_file:
#     pickle.dump(expLamb, output_file)