#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 3 for CLaSP 410.

'''

import numpy as np
from matplotlib import pyplot as plt
#  constants here.
sigma = 5.67E-8  # Steffan-Boltzman constant.

# The function below is pulled from Prof. Welling's github repo


def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    debug : boolean, default=False
        Turn on debug output.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # Populate our A matrix piece-by-piece.
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # Diagonal elements are always -2 ('cept at Earth's surface.)
            if i == j:
                A[i, j] = -1*(i > 0) - 1
                # print(f"Result: A[i,j]={A[i,j]}")
            else:
                # This is the pattern we solved for in class!
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # At Earth's surface, epsilon =1, breaking our pattern.
    # Divide by epsilon along surface to get correct results.
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)

    # Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)

    # Convert fluxes to temperatures.
    # Fluxes for all atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25  # Flux at ground: epsilon=1.

    return temps


def main():
    # set albedo to 0.33 because average albedo of Earth is around 0.3, emissivity is set to from 0 to 1 step size 0.5 to better visualize the relationship between emissivity and surface temp
    # q3 experiment 1
    epsilons = np.arange(0, 1.1, 0.05)
    Temps = []
    for eps in epsilons:
        Temps.append(n_layer_atmos(N=1, epsilon=eps, S0=1350, albedo=0.33))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    Temps = np.array(Temps)
    ax.plot(epsilons, Temps[:, 0], label='Earth Surface Temp', color='blue', linestyle='-', linewidth=2, marker='o')
    ax.plot(epsilons, Temps[:, 1], label='1st Atmosphere Surface Temp',
            color='red', linestyle='-', linewidth=2, marker='o')
    ax.axhline(y=288, color='orange', linestyle='--', label='Temp = 288K')
    ax.axvline(x=0.85, color='navy', linestyle='--', label='Epsilon = 0.85')
    ax.set_title('Surface Temp vs. Emissivity')
    ax.set_xlabel('Emissivity (epsilon)')
    ax.set_ylabel('Surface Temp (Kelvin)')
    ax.grid(True)
    ax.legend()
    plt.savefig('q3Exp1.png')

    # q3 experiment 2
    epsilon = 0.255
    num_layers = np.arange(1, 11)
    Earth_temp = []
    for n in num_layers:
        # append only surface temp of the Earth
        Earth_temp.append(n_layer_atmos(N=n, epsilon=epsilon, S0=1350, albedo=0.33)[0])
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    Earth_temp = np.array(Earth_temp)
    ax.plot(Earth_temp, num_layers, label='Altitude', color='blue', linestyle='-', linewidth=2, marker='o')
    ax.axvline(x=288, color='navy', linestyle='--', label='Temp = 288K')
    ax.set_title('Altitude vs. Earth Surface Temp')
    ax.set_xlabel('Earth Surface Temp (Kelvin)')
    ax.set_ylabel('Altitude (layer)')
    ax.grid(True)
    ax.legend()
    plt.savefig('q3Exp2.png')


if __name__ == '__main__':
    main()
