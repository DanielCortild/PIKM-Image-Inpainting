import sys
sys.path.append('../src')
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from PIKM_Inpainter.getInpainted import getInpainted2
# getInpainted(rho=1.8, sigma=.5, lamb=0.8, percent=.5, res="Tx-x",
#              nrows=3, ncols=2, figsize=(5, 8),
#              nplotrows=2, nplotcols=2, plotfsize=(5, 5), plotsanchor=-0.08)
getInpainted2(rho=1.8, sigma=.5, lamb=0.8, percent=.5, res="Tx-x", nplotrows=1, nplotcols=2, plotfsize=(8, 2))