import sys
sys.path.append('../src')

from PIKM_Inpainter import plotExperiments, plotExperimentRegularisation
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

NB_TESTS = 1

expPercent = plotExperiments(rho=1, sigma=0.5, lamb=0.5, percent=np.linspace(0, 1, 10), res="Tx-x", nb_tests=NB_TESTS)
plt.close()
with open(f"data/expPercent_Tx-x.{time.time()}.pickle", "wb") as output_file:
    pickle.dump(expPercent, output_file)

expRho = plotExperiments(rho=np.linspace(0.1, 1.9, 19), sigma=0.5, lamb=0.5, percent=0.5, res="Tx-x", nb_tests=NB_TESTS)
plt.close()
with open(f"data/expRho_Tx-x.{time.time()}.pickle", "wb") as output_file:
    pickle.dump(expRho, output_file)

expLamb = plotExperiments(rho=1, sigma=0.5, lamb=np.linspace(0.1, 1, 11), percent=0.5, res="Tx-x", nb_tests=1)
plt.close()
with open(f"data/expLamb_Tx-x.{time.time()}.pickle", "wb") as output_file:
    pickle.dump(expLamb, output_file)