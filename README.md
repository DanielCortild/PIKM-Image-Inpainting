# Perturbed Inertial KM Iterations for Image Inpainting

This package provides functions to execute *perturbed inertial Krasnoselskii-Mann iterations* to solve the image inpainting problem.

This code is related to my Bachelor Thesis on the topic of *perturbed inertial Krasnoselskii-Mann iterations*. In this thesis, I devise an inertial framework to accelerate the convergence of the standard KM iterations, and apply this algorithm to the image *inpainting problem*. This problem consists of reconstructing an image after part of it has been deleted (in this case a number of random pixels). The full text is available via [this link](http://dx.doi.org/10.13140/RG.2.2.15082.49601).

After more careful analysis the work gave rise to a paper, on the same topic. The preprint of this work may be found on [ArXiV](https://arxiv.org/abs/2401.16870).

## Installation

The package is available through pip, and may be installed via:

```
pip install PIKM_Inpainter
```

## Setup
In order to run the package, an image is required. The image must imperatively be called `Venice.jpeg`, and be placed in the same folder as the code is executed. Examples may be found in `/tests`.

## Main Usage
To utilize this package, you can call the `getInpainted` function:
```
getInpainted(rho, sigma, lamb, percent, res)
```

### Parameters:
- **rho** (_float_): Step size parameter, in the interval (0,2).
- **sigma** (_float_): Regularisation parameter, positive number.
- **lamb** (_float_): Relaxation parameter, in the interval (0,1).
- **percent** (_float_): Percentage of pixels erased randomly in the image, in the interval (0,1).
- **res** (_string_): Type of residual (`"rel"` for relative residual, `"Tx-x"` for pointwise residual)

### Returns:
- None

## Running Experiments
Experiments are pre-coded in the library, through the function `plotExperiments`.
```
plotExperiments(rho, sigma, lamb, percent, res)
```

### Parameters:
One of the parameters should be a list of parameters, which will determine the experiment.
- **rho** (_float_): Step size parameter, in the interval (0,2).
- **sigma** (_float_): Regularisation parameter, positive number.
- **lamb** (_float_): Relaxation parameter, in the interval (0,1).
- **percent** (_float_): Percentage of pixels erased randomly in the image, in the interval (0,1).
- **res** (_string_): Type of residual (`"rel"` for relative residual, `"Tx-x"` for pointwise residual)

### Output:
* **its_S** (_list_): List of iterations required for static.
* **its_H** (_list_): List of iterations required for heavy-ball.
* **its_N** (_list_): List of iterations required for Nesterov.
* **its_R** (_list_): List of iterations required for reflected.
* **time_S** (_list_): List of times required for static.
* **time_H** (_list_): List of times required for heavy-ball.
* **time_N** (_list_): List of times required for Nesterov.
* **time_R** (_list_): List of times required for reflected.

## Experiment on Regularisation Parameter
A specific type of experiment may be run on the regularisation parameter, through the function `plotExperimentRegularisation`.
```
plotExperimentRegularisation(rho, sigmas, lamb, percent, method, res)
```

### Parameters:
- **rho** (_float_): Step size parameter, in the interval (0,2).
- **sigmas** (_list_): List of regularisation parameters. Array must have 6 regularisation parameters.
- **lamb** (_float_): Relaxation parameter, in the interval (0,1).
- **percent** (_float_): Percentage of pixels erased randomly in the image, in the interval (0,1).
- **method** (_string_): The chosen acceleration method. Must be one of `"static"`, `"heavyball"`, `"nesterov"` or `"reflected"`.
- **res** (_string_): Type of residual (`"rel"` for relative residual, `"Tx-x"` for pointwise residual)

## Example

```
from PIKM_Inpainter import getInpainted
getInpainted(rho=1.8, sigma=.5, lamb=.8, percent=.5, res="Tx-x")
```

In this example we select a step size $\rho=1.8$, a regularisation parameter $\sigma=0.5$, a relaxation parameter $\lambda=0.8$, and a percentage of erased pixels of $50\%$.

This produces the result in the following two figures.

![](https://github.com/DanielCortild/PIKM-Image-Inpainting/blob/master/plots/all_Tx-x.png?raw=true)
![](https://github.com/DanielCortild/PIKM-Image-Inpainting/blob/master/plots/residuals_Tx-x.png?raw=true)