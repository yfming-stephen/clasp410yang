# !/usr/bin/env python3
'''
This file contains tools for calculating timeseries of temperature for
Kangerlussuaq and plotting heat map and seasonal temperature profile
for the ground temperature system.
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7,
                     -21.0,
                     10.7, 8.5, 3.1,
                     -17.,
                     -6.0,
                     -8.4, 2.3, 8.4,
                     -12.0,
                     -16.9])


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    Parameters:
    -----------
    t : array
        Time array
    Returns:
    --------
    t_s : array
        Array of temperature over time
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    t_s = t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()
    return t_s


def Kangerlussuaq(xmax, tmax, dx, dt, c2=1, temp_shift=0, debug=True):
    '''
    Parameters:
    -----------
    xmax : float
        The depth of the ground, unit m
    tmax : float
        The time elapsed, unit s
    dx : float
        Step size in depth, unit m
    dt : float
        Step size in time, unit s
    c2 : float
        Thermal diffusivity, unit m^2/s
    Returns:
    --------
    xgrid, tgrid : ndarray
        Space grid and temperature grid for the heat diffusion model.
    U : ndarray
        Heat matrix
    '''

    # check numerical stability
    if dt > dx**2 / (2*c2):
        raise ValueError('dt is too large. Doesn\'t satisfy numerical stability criterion')
    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros((M, N))

    # Set initial conditions:
    # initial condition is set to 0 for Kangerlussuaq model
    U[:, 0] = 0
    # Set boundary conditions:
    # temp_shift is defaul to 0, and create the system without warming effect
    # when temp_shift is not 0, warming effect is added to the system
    U[0, :] = temp_kanger(np.arange(0, N))+temp_shift
    U[-1, :] = 5

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])

    # Return grid and result:
    return xgrid, tgrid, U
# The base heat diffusion solver written by Prof. Welling in lecture


def heatdiff(xmax, tmax, dx, dt, c2=1, neumann=False, debug=True):
    '''
    Parameters:
    -----------
    xmax : float
        The depth of the ground
    tmax : float
        The time elapsed
    dx : float
        Step size in depth
    dt : float
        Step size in time
    c2 : float
        Thermal diffusivity
    Returns:
    --------
    xgrid, tgrid : ndarray
        Space grid and temperature grid for the heat diffusion model.
    U : ndarray
        Heat matrix
    '''

    # check numerical stability
    if dt > dx**2 / (2*c2):
        raise ValueError('dt is too large. Doesn\'t satisfy numerical stability criterion')
    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros((M, N))

    # Set initial conditions:
    U[:, 0] = 4*xgrid - 4*xgrid**2
    # Set boundary conditions:
    U[0, :] = 0
    U[-1, :] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
        if neumann:
            U[0, j+1] = U[1, j+1]
            U[-1, j+1] = U[-2, j+1]
    # Return grid and result:
    return xgrid, tgrid, U


def plot_groundtemp(time, depth, heat, outname="ground_profile.png",
                    dx=.5, c2=2.5e-7):
    '''
    Create a plot of Kangerlussuaq ground temperature. The figure and axes
    objects are returned for further modification as necessary.

    Parameters
    ----------
    time : 1D numpy array
        Time (in years) of the simulation.
    depth : 1D numpy array
        The depth, in meters, of the ground corresponding to the heat array.
    heat : 2D numpy array
        The result of a the Kangerlussuaq function heat solver.
    outname : string, defaults to "ground_profile.png"
        Set file name for saving result.
    dx, c2 : floats
        Set values for grid step and diffusion coefficient for labeling plot.

    Returns
    -------
    fig : Matplotlib figure object
        Figure object containing the resulting plot.
    ax1, ax2 : Matplotlib axes objects
        Axes object containing the plots.
    '''

    # Extract summer and winter ground temp curves:
    summer = heat[:, 365:].max(axis=1)
    winter = heat[:, 365:].min(axis=1)

    # Create figure with 2 axes.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # LEFT PLOT: Contour of temp vs. depth and time.
    map = ax1.pcolormesh(time, depth, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax1, label='Temperature ($°C$)')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
    ax1.invert_yaxis()

    # RIGHT PLOT: Summer/winter profiles
    ax2.plot(winter, depth, label='Winter', color='blue')
    ax2.plot(summer, depth, label='Summer', color='red', linestyle='--')
    ax2.invert_yaxis()
    ax2.set_xlabel('Temperature($°C$)')
    ax2.set_ylabel('Depth(m)')
    ax2.set_title('Ground Temperature: Kangerlussuaq')
    ax2.legend()
    ax2.grid()

    # Add details to image:
    fig.text(0.5, 0.01, f"Input:  depth = {depth[-1]}m, years = {time[-1]}, " +
             f"dx = {dx}m, dt = 1 day, c2 = {c2}m^2/s", ha="center",
             fontsize=12)

    fig.tight_layout()
    fig.savefig(outname)

    return fig, ax1, ax2


def main():

    # # Get solution using your solver:
    # x, t, heat = heatdiff(1, 0.2, 0.2, 0.02, c2=1, neumann=True, debug=False)
    # print(heat)
    # # Create a figure/axes object
    # fig, axes = plt.subplots(1, 1)
    # # Create a color map and add a color bar.
    # map = axes.pcolor(t, x, heat, cmap='seismic', vmin=0, vmax=1)
    # plt.colorbar(map, ax=axes, label='Temperature ($°C$)')
    # axes.set_xlabel('Time (Years)')
    # axes.set_ylabel('Depth (m)')
    # axes.set_title('Ground Temperature')
    # fig.savefig('heatmap.png')
    # # Set indexing for the final year of results:
    # loc = int(-365/0.002)  # Final 365 days of the result.
    # # Extract the min values over the final year:
    # summer = heat[:, loc:].max(axis=1)
    # winter = heat[:, loc:].min(axis=1)
    # # Create a temp profile plot:
    # fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    # ax2.plot(winter, x, label='Winter', color='blue')
    # ax2.plot(summer, x, label='Summer', color='red')
    # fig.savefig('groundprofile.png')

    # x, t, heat = heatdiff(1, 0.2, 0.2, 0.002, c2=1, neumann=True, debug=False)
    # fig, axes = plt.subplots(1, 1)
    # # Create a color map and add a color bar.
    # map = axes.pcolor(t, x, heat, cmap='seismic', vmin=0, vmax=1)
    # plt.colorbar(map, ax=axes, label='Temperature ($°C$)')
    # axes.set_xlabel('Time (Years)')
    # axes.set_ylabel('Depth (m)')
    # axes.set_title('Ground Temperature')
    # fig.savefig('heatmap_smooth.png')
    # using depth = 100m, time = 5 years (in seconds), dx = 0.5, dt = one day (in seconds), c2 = 2.5e-7m^2/s
    depth = 100
    oneYear = 31536000
    dx = 0.5
    dt = 86400
    c2 = 2.5e-7

    # # Q2
    years = [5*oneYear, 25*oneYear, 100*oneYear, 150*oneYear]
    for y in years:
        x, t, heat = Kangerlussuaq(depth, y, dx, dt, c2, debug=False)
        plot_groundtemp(t/oneYear, x, heat,
                        outname=f'kanger_prof_{int(y/oneYear)}.png')

    # x, t, heat = Kangerlussuaq(depth, 150*oneYear, dx, dt, c2, debug=False)
    # loc = -365
    # summer = heat[:, loc:].max(axis=1)
    # winter = heat[:, loc:].min(axis=1)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    # # change the time unit to year by dividing seconds in a year
    # map = ax1.pcolor(t/oneYear, x, heat, cmap='seismic', vmin=-25, vmax=25)
    # plt.colorbar(map, ax=ax1, label='Temperature ($°C$)')
    # ax1.set_xlabel('Time (Years)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
    # ax1.invert_yaxis()
    # ax2.plot(winter, x, label='Winter', color='blue')
    # ax2.plot(summer, x, label='Summer', color='red', linestyle='--')
    # ax2.invert_yaxis()
    # ax2.axhline(y=1.7, color='green', linestyle='-', label='depth = 1.7m')
    # ax2.axhline(y=52, color='navy', linestyle='-', label='depth = 52m')
    # ax2.set_xlabel('Temperature($°C$)')
    # ax2.set_ylabel('Depth(m)')
    # ax2.set_title('Ground Temperature: Kangerlussuaq')
    # ax2.legend()
    # ax2.grid()
    # fig.text(
    #     0.5, 0.01, f"Input:  depth = {depth}m, years = {150}, dx = {dx}m, dt = 1 day, c2 = {c2}m^2/s", ha="center", fontsize=12)
    # fig.savefig(f'kanger_steady.png')

    # Q3
    shifts = [0.5, 1, 3]
    # x, t, heat = Kangerlussuaq(depth, 150*oneYear, dx, dt, c2, temp_shift=shifts[0], debug=False)
    # loc = -365
    # summer = heat[:, loc:].max(axis=1)
    # winter = heat[:, loc:].min(axis=1)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    # # change the time unit to year by dividing seconds in a year
    # map = ax1.pcolor(t/oneYear, x, heat, cmap='seismic', vmin=-25, vmax=25)
    # plt.colorbar(map, ax=ax1, label='Temperature ($°C$)')
    # ax1.set_xlabel('Time (Years)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
    # ax1.invert_yaxis()
    # ax2.plot(winter, x, label='Winter', color='blue')
    # ax2.plot(summer, x, label='Summer', color='red', linestyle='--')
    # ax2.invert_yaxis()
    # ax2.set_xlabel('Temperature($°C$)')
    # ax2.set_ylabel('Depth(m)')
    # ax2.set_title('Ground Temperature: Kangerlussuaq')
    # ax2.axhline(y=2, color='green', linestyle='-', label='depth = 2m')
    # ax2.axhline(y=51, color='navy', linestyle='-', label='depth = 51m')
    # ax2.legend()
    # ax2.grid()
    # fig.text(
    #     0.5, 0.01, f"Input:  depth = {depth}m, years = {150}, dx = {dx}m, dt = 1 day, c2 = {c2}m^2/s, temp_shift = {shifts[0]}", ha="center", fontsize=12)
    # fig.savefig(f'kanger_shift_{shifts[0]}.png')

    # x, t, heat = Kangerlussuaq(depth, 150*oneYear, dx, dt, c2, temp_shift=shifts[1], debug=False)
    # loc = -365
    # summer = heat[:, loc:].max(axis=1)
    # winter = heat[:, loc:].min(axis=1)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    # # change the time unit to year by dividing seconds in a year
    # map = ax1.pcolor(t/oneYear, x, heat, cmap='seismic', vmin=-25, vmax=25)
    # plt.colorbar(map, ax=ax1, label='Temperature ($°C$)')
    # ax1.set_xlabel('Time (Years)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
    # ax1.invert_yaxis()
    # ax2.plot(winter, x, label='Winter', color='blue')
    # ax2.plot(summer, x, label='Summer', color='red', linestyle='--')
    # ax2.invert_yaxis()
    # ax2.set_xlabel('Temperature($°C$)')
    # ax2.set_ylabel('Depth(m)')
    # ax2.set_title('Ground Temperature: Kangerlussuaq')
    # ax2.axhline(y=2.2, color='green', linestyle='-', label='depth = 2.2m')
    # ax2.axhline(y=49, color='navy', linestyle='-', label='depth = 49m')
    # ax2.legend()
    # ax2.grid()
    # fig.text(
    #     0.5, 0.01, f"Input:  depth = {depth}m, years = {150}, dx = {dx}m, dt = 1 day, c2 = {c2}m^2/s, temp_shift = {shifts[1]}", ha="center", fontsize=12)
    # fig.savefig(f'kanger_shift_{shifts[1]}.png')

    # x, t, heat = Kangerlussuaq(depth, 150*oneYear, dx, dt, c2, temp_shift=shifts[2], debug=False)
    # loc = -365
    # summer = heat[:, loc:].max(axis=1)
    # winter = heat[:, loc:].min(axis=1)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    # # change the time unit to year by dividing seconds in a year
    # map = ax1.pcolor(t/oneYear, x, heat, cmap='seismic', vmin=-25, vmax=25)
    # plt.colorbar(map, ax=ax1, label='Temperature ($°C$)')
    # ax1.set_xlabel('Time (Years)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Ground Temperature: Kangerlussuaq, Greenland')
    # ax1.invert_yaxis()
    # ax2.plot(winter, x, label='Winter', color='blue')
    # ax2.plot(summer, x, label='Summer', color='red', linestyle='--')
    # ax2.invert_yaxis()
    # ax2.set_xlabel('Temperature($°C$)')
    # ax2.set_ylabel('Depth(m)')
    # ax2.set_title('Ground Temperature: Kangerlussuaq')
    # ax2.axhline(y=3, color='green', linestyle='-', label='depth = 3m')
    # ax2.axhline(y=40, color='navy', linestyle='-', label='depth = 40m')
    # ax2.legend()
    # ax2.grid()
    # fig.text(
    #     0.5, 0.01, f"Input:  depth = {depth}m, years = {150}, dx = {dx}m, dt = 1 day, c2 = {c2}m^2/s, temp_shift = {shifts[2]}", ha="center", fontsize=12)
    # fig.savefig(f'kanger_shift_{shifts[2]}.png')


if __name__ == '__main__':
    main()
