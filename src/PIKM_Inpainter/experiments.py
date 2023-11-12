# !/usr/bin/env python
# encoding: utf-8
"""
experiments.py - Implements the experiments about the different parameters
~ Daniel Cortild, 11 November 2023
"""

# External Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Internal Imports
from .Image import Image
from .Mask import Mask
from .Algorithm import Algorithm

# Running Algoroithm
def runAlgorithm(rho, lamb, sigma, percent, tolerance, max_it, method):
    """
    Runs the algorithm given different parameters.
    Input:
        rho         Step size parameter
        lamb        Regularisation parameter
        sigma       Relaxation parameter
        percent     Percentage of erased pixels
        tolerance   Stopping tolerance
        max_it      Maximum number of iterations
        method      Acceleration method selected
    Output:
        its         Iterations required to run the algorithm
        time        Time required to run the algorithm
    """
    image = Image(name="Image", dims=(512, 512))
    image.load_image("Venice.jpeg")
    mask = Mask(percentage=percent, dims=(512, 512))
    image_corrupt = mask.mask_image(image.image)
    _, _, _, _, its, time, _ = Algorithm(image_corrupt, mask, rho, lamb, sigma, 
                                         tolerance, max_it, method).run(False, False)
        
    return its, time

# Error Plots
def plotItsTime(parameters, max_it, its_S, its_H, its_N, its_R, 
                  time_S, time_H, time_N, time_R, xlabel, title):
    """
    Plots the results of an experiment given the results.
    Input:
        parameters      The list of paramaters computed against
        max_it          Maximum number of iterations
        its_S           List of iterations required for static
        its_H           List of iterations required for heavy-ball
        its_N           List of iterations required for Nesterov
        its_R           List of iterations required for reflected
        time_S          List of times required for static
        time_H          List of times required for heavy-ball
        time_N          List of times required for Nesterov
        time_R          List of times required for reflected
        xlabel          The X-label for the plot
        title           The plot title
    Output:
        None
    """
    # Post analyse the data
    max_time = max(max(time_S), max(time_H), max(time_N), max(time_R))
    time_S = [time_S[i] if its_S[i] < max_it else max_time for i in range(len(time_S))]
    time_H = [time_H[i] if its_H[i] < max_it else max_time for i in range(len(time_H))]
    time_N = [time_N[i] if its_N[i] < max_it else max_time for i in range(len(time_N))]
    time_R = [time_R[i] if its_R[i] < max_it else max_time for i in range(len(time_R))]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.5), dpi=600)

    axs[0].title.set_text("Iterations to reach $10^{-3}$ tolerance")
    axs[0].plot(parameters, its_S, label="Static")
    axs[0].plot(parameters, its_H, label="Heavy-Ball")
    axs[0].plot(parameters, its_N, label="Nesterov")
    axs[0].plot(parameters, its_R, label="Reflected")
    axs[0].axhline(y=max_it, color="r", label="Did Not Converge", linestyle="--")
    axs[0].set_ylabel("Iterations")
    axs[0].set_xlabel(xlabel)
    axs[0].legend()

    axs[1].title.set_text("Time to reach $10^{-3}$ tolerance")
    axs[1].plot(parameters, time_S, label="Static")
    axs[1].plot(parameters, time_H, label="Heavy-Ball")
    axs[1].plot(parameters, time_N, label="Nesterov")
    axs[1].plot(parameters, time_R, label="Reflected")
    axs[1].axhline(y=max_time, color="r", label="Did Not Converge", linestyle="--")
    axs[1].set_ylabel("Time in seconds")
    axs[1].set_xlabel(xlabel)
    axs[1].legend()

    plt.show()

def plotExperiments(rho, sigma, lamb, percent):
    """
    Plots the results of an experiment based on different parameters.
    Input:
        rho         Step size parameter (Float or list of floats)
        sigma       Regularisation parameter (Float or list of floats)
        lamb        Relaxation parameter (Float or list of floats)
        percent     Percentage of erased pixels (Float or list of floats)
    Output:
        its_S       List of iterations required for static
        its_H       List of iterations required for heavy-ball
        its_N       List of iterations required for Nesterov
        its_R       List of iterations required for reflected
        time_S      List of times required for static
        time_H      List of times required for heavy-ball
        time_N      List of times required for Nesterov
        time_R      List of times required for reflected
    """
    # Import the Image
    image = Image(name="Image", dims=(512, 512))
    image.load_image("Venice.jpeg")

    # Universal Parameters
    tolerance = 1e-3
    max_it = 100

    # Check parameters
    if sum(map(lambda x: type(x) in [list, np.ndarray], [rho, sigma, lamb, percent])) != 1 and \
       sum(map(lambda x: type(x) in [float, int], [rho, sigma, lamb, percent])) != 3:
        raise ValueError("Only one input should be a list, others should be floats")

    # Get number of tests
    if type(rho) in [list, np.ndarray]:
        N = len(rho)
        parameters = rhos = rho
        sigmas = [sigma] * N
        lambs = [lamb] * N
        percents = [percent] * N
        xAxis = r"Step size ($\rho$)"
        title = "Analysis on the step size"
    elif type(sigma) in [list, np.ndarray]:
        N = len(sigma)
        rhos = [rho] * N
        parameters = sigmas = sigma
        lambs = [lamb] * N
        percents = [percent] * N
        xAxis = r"Regularisation parameter ($\sigma$)"
        title = "Analysis on the regularisation parameter"
    elif type(lamb) in [list, np.ndarray]:
        N = len(lamb)
        rhos = [rho] * N
        sigmas = [sigma] * N
        parameters = lambs = lamb
        percents = [percent] * N
        xAxis = r"Relaxation parameter ($\lambda$)"
        title = "Analysis on the relaxation parameter"
    elif type(percent) in [list, np.ndarray]:
        N = len(percent)
        rhos = [rho] * N
        sigmas = [sigma] * N
        lambs = [lamb] * N
        parameters = percents = percent
        xAxis = "Percentage of erased pixels"
        title = "Analysis on the percentage of erased pixels"
    else:
        raise ValueError("At least one input should be a list")

    # Set empty lists for results
    its_S   = [0] * N
    its_H   = [0] * N
    its_N   = [0] * N
    its_R   = [0] * N
    time_S  = [0] * N
    time_H  = [0] * N
    time_N  = [0] * N
    time_R  = [0] * N

    # Run the Algorithm
    for i, (rho, sigma, lamb, percent) in enumerate(tqdm(zip(rhos, sigmas, lambs, percents))):
        its_S[i], time_S[i] = runAlgorithm(rho, lamb, sigma, percent,
                                  tolerance, max_it, method="static")
        its_H[i], time_H[i] = runAlgorithm(rho, lamb, sigma, percent,
                                    tolerance, max_it, method="heavyball")
        its_N[i], time_N[i] = runAlgorithm(rho, lamb, sigma, percent,
                                    tolerance, max_it, method="nesterov")
        its_R[i], time_R[i] = runAlgorithm(rho, lamb, sigma, percent,
                                    tolerance, max_it, method="reflected")

    # Plot the results
    plotItsTime(parameters, max_it, its_S, its_H, its_N, its_R, 
                time_S, time_H, time_N, time_R, xAxis, title)

    return its_S, its_H, its_N, its_R, time_S, time_H, time_N, time_R

def plotExperimentRegularisation(rho, sigmas, lamb, percent, method):
    """
    Plots different images for different regularisation parameters.
    Input:
        rho         Step size selected
        sigmas      Relaxation parameters selected
        lamb        Regularisation parameter selected
        percent     Percentage of erased pixels selected 
        method      Acceleration method chosen
    Output:
        None
    """
    # Check input
    if (not type(rho) in [float, int]) or (not type(lamb) in [float, int]) or (not type(percent) in [float, int]) or \
        type(sigmas) not in [list, np.ndarray] or method not in ["static", "heavyball", "nesterov", "reflected"]:
        raise ValueError("Input is incorrect.")
    if len(sigmas) != 6:
        raise ValueError("List `sigmas` should contain 6 elements exactly.")

    # Universal Parameters
    tolerance = 1e-3
    max_it = 100

    # Import the Image
    image = Image(name="Image", dims=(512, 512))
    image.load_image("Venice.jpeg")
    mask = Mask(percentage=percent, dims=(512, 512))
    image_corrupt = mask.mask_image(image.image)

    # Create Algorithm Instances
    sols = [None] * len(sigmas)
    for i, sigma in enumerate(tqdm(sigmas)):
        sols[i] = Algorithm(image_corrupt, mask, rho, lamb, sigma, 
                            tolerance, max_it, method=method).run()[0]

    # Plot Original and Result
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), dpi=100)

    axs[0,0].title.set_text("Original Image")
    axs[0,0].imshow(image.image, vmin=0, vmax=1)
    axs[0,0].set_axis_off()
    axs[0,0].set_facecolor("white")

    axs[0,1].title.set_text("Corrupt Image")
    axs[0,1].imshow(image_corrupt, vmin=0, vmax=1)
    axs[0,1].set_axis_off()
    axs[0,1].set_facecolor("white")

    axs[0,2].title.set_text(r"Recovered with $\sigma=0.0625$")
    axs[0,2].imshow(sols[0], vmin=0, vmax=1)
    axs[0,2].set_axis_off()
    axs[0,2].set_facecolor("white")

    axs[0,3].title.set_text(r"Recovered with $\sigma=0.25$")
    axs[0,3].imshow(sols[1], vmin=0, vmax=1)
    axs[0,3].set_axis_off()
    axs[0,3].set_facecolor("white")

    axs[1,0].title.set_text(r"Recovered with $\sigma=0.5$")
    axs[1,0].imshow(sols[2], vmin=0, vmax=1)
    axs[1,0].set_axis_off()
    axs[1,0].set_facecolor("white")

    axs[1,1].title.set_text(r"Recovered with $\sigma=1$")
    axs[1,1].imshow(sols[3], vmin=0, vmax=1)
    axs[1,1].set_axis_off()
    axs[1,1].set_facecolor("white")

    axs[1,2].title.set_text(r"Recovered with $\sigma=4$")
    axs[1,2].imshow(sols[4], vmin=0, vmax=1)
    axs[1,2].set_axis_off()
    axs[1,2].set_facecolor("white")

    axs[1,3].title.set_text(r"Recovered with $\sigma=16$")
    axs[1,3].imshow(sols[5], vmin=0, vmax=1)
    axs[1,3].set_axis_off()
    axs[1,3].set_facecolor("white")

    plt.show()