# !/usr/bin/env python
# encoding: utf-8
"""
getInpainted.py - Returns the inpainted images
~ Daniel Cortild, 11 November 2023
"""

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Internal Imports
from .Image import Image
from .Mask import Mask
from .Algorithm import Algorithm

def getInpainted(rho, sigma, lamb, percent, res="rel", nrows=2, ncols=3, figsize=(10, 5)):
    """
    Returns the inpainted images.
    Input:
        rho         Step size
        sigma       Regularisation parameter
        lamb        Relaxation parameter
        percent     Percentage of erased pixels
        res         Type of residual ("rel" or "Tx-x")
        nrows       Number of rows (Default: 2)
        ncols       Number of columns (Default: 3)
        figsize     Figsize (Default: (10, 5))
    Output:
        None
    """
    # Import the Image
    image = Image(name="Image", dims=(512, 512))
    image.load_image("Venice.jpeg")
    mask = Mask(percentage=percent, dims=(512, 512))
    image_corrupt = mask.mask_image(image.image)

    # Set the parameters
    tolerance = 1e-3 if res == "rel" else 0.5
    max_it = 100

    # Create Algorithm Instance
    sol_S, hist_S, hist_T_S, hist_F_S, its_S, time_S, F_S = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="static", res=res).run()
    sol_H, hist_H, hist_T_H, hist_F_H, its_H, time_H, F_H = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="heavyball", res=res).run()
    sol_N, hist_N, hist_T_N, hist_F_N, its_N, time_N, F_N = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="nesterov", res=res).run()
    sol_R, hist_R, hist_T_R, hist_F_R, its_R, time_R, F_R = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="reflected", res=res).run()

    # Plot Original and Result
    fig = plt.figure(figsize=figsize, dpi=300)

    ax = fig.add_subplot(nrows, ncols, 1)
    ax.title.set_text("Original Image")
    ax.imshow(image.image, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax = fig.add_subplot(nrows, ncols, 2)
    ax.title.set_text("Corrupt Image")
    ax.imshow(image_corrupt, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax = fig.add_subplot(nrows, ncols, 3)
    ax.title.set_text("Non-Inertial")
    ax.imshow(sol_S, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax = fig.add_subplot(nrows, ncols, 4)
    ax.title.set_text("Heavy Ball")
    ax.imshow(sol_H, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax = fig.add_subplot(nrows, ncols, 5)
    ax.title.set_text("Nesterov")
    ax.imshow(sol_N, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax = fig.add_subplot(nrows, ncols, 6)
    ax.title.set_text("Reflected")
    ax.imshow(sol_R, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_facecolor("white")

    # Save plot as file
    try:
        t = time.time()
        print(f"Saving file to ./plots/{t}.png")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(f"plots/{t}.png", bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"ERROR: {e}")

    # Error Plots
    fig = plt.figure(figsize=(8, 2), dpi=300)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.45)
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text("$|X_k-X_{k-1}|^2$")
    l1 = ax.plot([np.linalg.norm(hist_S[i] - hist_S[i-1])**2 for i in range(1, len(hist_S))], 
                        label="Non-Inertial")
    l2 = ax.plot([np.linalg.norm(hist_H[i] - hist_H[i-1])**2 for i in range(1, len(hist_H))], 
                        label="Heavy Ball")
    l3 = ax.plot([np.linalg.norm(hist_N[i] - hist_N[i-1])**2 for i in range(1, len(hist_N))], 
                        label="Nesterov")
    l4 = ax.plot([np.linalg.norm(hist_R[i] - hist_R[i-1])**2 for i in range(1, len(hist_R))], 
                        label="Reflected")
    ax.set_yscale('log')
    ax.set_xlabel("Iteration (k)")

    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text(r"$|T_kX_k-X_{k}|^2$")
    ax.plot([np.linalg.norm(hist_T_S[i] - hist_S[i])**2 for i in range(1, len(hist_S))], 
                label="Non-Inertial")
    ax.plot([np.linalg.norm(hist_T_H[i] - hist_H[i])**2 for i in range(1, len(hist_H))], 
                label="Heavy Ball")
    ax.plot([np.linalg.norm(hist_T_N[i] - hist_N[i])**2 for i in range(1, len(hist_N))], 
                label="Nesterov")
    ax.plot([np.linalg.norm(hist_T_R[i] - hist_R[i])**2 for i in range(1, len(hist_R))], 
                label="Reflected")
    ax.set_yscale('log')
    ax.set_xlabel("Iteration (k)")

    fig.legend([l1, l2, l3, l4], 
            labels=["Non-Inertial", "Heavy Ball", "Nesterov", "Reflected"],
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.3),
                ncol=4)

    # Save plot as file
    try:
        t = time.time()
        print(f"Saving file to ./plots/{t}.png")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(f"plots/{t}.png", bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"ERROR: {e}")