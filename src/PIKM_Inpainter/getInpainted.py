# !/usr/bin/env python
# encoding: utf-8
"""
getInpainted.py - Returns the inpainted images
~ Daniel Cortild, 11 November 2023
"""

# External Imports
import numpy as np
import matplotlib.pyplot as plt

# Internal Imports
from .Image import Image
from .Mask import Mask
from .Algorithm import Algorithm


def getInpainted(rho, sigma, lamb, percent):
    """
    Returns the inpainted images.
    Input:
        rho         Step size
        sigma       Regularisation parameter
        lamb        Relaxation parameter
        percent     Percentage of erased pixels
    Output:
        None
    """
    # Import the Image
    image = Image(name="Image", dims=(512, 512))
    image.load_image("Venice.jpeg")
    mask = Mask(percentage=0.5, dims=(512, 512))
    image_corrupt = mask.mask_image(image.image)

    # Set the parameters
    tolerance = 1e-3
    max_it = 100

    # Create Algorithm Instance
    sol_S, hist_S, hist_T_S, hist_F_S, its_S, time_S, F_S = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="static").run()
    sol_H, hist_H, hist_T_H, hist_F_H, its_H, time_H, F_H = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="heavyball").run()
    sol_N, hist_N, hist_T_N, hist_F_N, its_N, time_N, F_N = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="nesterov").run()
    sol_R, hist_R, hist_T_R, hist_F_R, its_R, time_R, F_R = Algorithm(
                                        image_corrupt, mask, rho, lamb, sigma, 
                                        tolerance, max_it, method="reflected").run()

    # Plot Original and Result
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=200)

    axs[0,0].title.set_text("Original Image")
    axs[0,0].imshow(image.image, vmin=0, vmax=1)
    axs[0,0].set_axis_off()
    axs[0,0].set_facecolor("white")

    axs[0,1].title.set_text("Corrupt Image")
    axs[0,1].imshow(image_corrupt, vmin=0, vmax=1)
    axs[0,1].set_axis_off()
    axs[0,1].set_facecolor("white")

    axs[0,2].title.set_text("Static")
    axs[0,2].imshow(sol_S, vmin=0, vmax=1)
    axs[0,2].set_axis_off()
    axs[0,2].set_facecolor("white")

    axs[1,0].title.set_text("Heavy Ball")
    axs[1,0].imshow(sol_H, vmin=0, vmax=1)
    axs[1,0].set_axis_off()
    axs[1,0].set_facecolor("white")

    axs[1,1].title.set_text("Nesterov")
    axs[1,1].imshow(sol_N, vmin=0, vmax=1)
    axs[1,1].set_axis_off()
    axs[1,1].set_facecolor("white")

    axs[1,2].title.set_text("Reflected")
    axs[1,2].imshow(sol_R, vmin=0, vmax=1)
    axs[1,2].set_axis_off()
    axs[1,2].set_facecolor("white")

    plt.show()

    # Error Plots
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.2), dpi=200)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.45)
    plt.tight_layout()

    axs[0].title.set_text("$|X_k-X_{k-1}|^2$")
    l1 = axs[0].plot([np.linalg.norm(hist_S[i] - hist_S[i-1])**2 for i in range(1, len(hist_S))], 
                        label="Static")
    l2 = axs[0].plot([np.linalg.norm(hist_H[i] - hist_H[i-1])**2 for i in range(1, len(hist_H))], 
                        label="Heavy Ball")
    l3 = axs[0].plot([np.linalg.norm(hist_N[i] - hist_N[i-1])**2 for i in range(1, len(hist_N))], 
                        label="Nesterov")
    l4 = axs[0].plot([np.linalg.norm(hist_R[i] - hist_R[i-1])**2 for i in range(1, len(hist_R))], 
                        label="Reflected")
    axs[0].set_yscale('log')
    axs[0].set_xlabel("Iteration (k)")

    axs[1].title.set_text(r"$|T_kX_k-X_{k}|^2$")
    axs[1].plot([np.linalg.norm(hist_T_S[i] - hist_S[i])**2 for i in range(1, len(hist_S))], 
                label="Static")
    axs[1].plot([np.linalg.norm(hist_T_H[i] - hist_H[i])**2 for i in range(1, len(hist_H))], 
                label="Heavy Ball")
    axs[1].plot([np.linalg.norm(hist_T_N[i] - hist_N[i])**2 for i in range(1, len(hist_N))], 
                label="Nesterov")
    axs[1].plot([np.linalg.norm(hist_T_R[i] - hist_R[i])**2 for i in range(1, len(hist_R))], 
                label="Reflected")
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Iteration (k)")

    axs[2].title.set_text(r"$|X_k-X^*|^2$")
    axs[2].plot([np.linalg.norm(hist_S[i] - image.image)**2 for i in range(1, len(hist_S))], 
                label="Static")
    axs[2].plot([np.linalg.norm(hist_H[i] - image.image)**2 for i in range(1, len(hist_H))], 
                label="Heavy Ball")
    axs[2].plot([np.linalg.norm(hist_N[i] - image.image)**2 for i in range(1, len(hist_N))], 
                label="Nesterov")
    axs[2].plot([np.linalg.norm(hist_R[i] - image.image)**2 for i in range(1, len(hist_R))], 
                label="Reflected")
    axs[2].set_yscale('log')
    axs[2].set_xlabel("Iteration (k)")

    axs[3].title.set_text(r"$|F(X_k)-X^*|^2$")
    axs[3].plot([np.linalg.norm(hist_F_S[i] - F_S(image.image))**2 for i in range(1, len(hist_S))], 
                label="Static")
    axs[3].plot([np.linalg.norm(hist_F_H[i] - F_H(image.image))**2 for i in range(1, len(hist_H))], 
                label="Heavy Ball")
    axs[3].plot([np.linalg.norm(hist_F_N[i] - F_N(image.image))**2 for i in range(1, len(hist_N))], 
                label="Nesterov")
    axs[3].plot([np.linalg.norm(hist_F_R[i] - F_R(image.image))**2 for i in range(1, len(hist_R))], 
                label="Reflected")
    axs[3].set_yscale('log')
    axs[3].set_xlabel("Iteration (k)")

    fig.legend([l1, l2, l3, l4], 
            labels=["Static", "Heavy-Ball", "Nesterov", "Reflected"],
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.18),
                ncol=4)

    plt.show()