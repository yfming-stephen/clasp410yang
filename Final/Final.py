#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Final Project for CLaSP 410.

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import queue
import sys


plt.style.use('fivethirtyeight')
forest_cmap = ListedColormap(['royalblue', 'tan', 'darkgreen', 'crimson'])
disease_cmap = ListedColormap(['orange', 'tan', 'darkgreen', 'crimson'])
np.random.seed(410)


def create_rain_mask(m, n, r):
    '''
    This function creates a rain mask, marking grids that are raining

    Parameters
    ==========
    m, n : integer
        Dimension of the grid (the foreset).
    r : interger
        Dimension of the mask

    Returns
    =======
    mask : nparray
        The rain mask created.

    '''
    y = np.random.randint(0, m - r)
    x = np.random.randint(0, n-r)
    mask = np.zeros((m, n), dtype=bool)
    mask[y:y+r, x:x+r] = True
    return mask


def spread(m, n, pspread, pbare, pstart, wind_dir, wind_speed, prain, rain_mask, init_cell, maxiter=4):
    '''
    This function performs a fire/disease spread simultion
    by plotting out the forest status for each step of the simulation

    Parameters
    ==========
    m, n : integer
        Dimension of the grid (the foreset).
    pspread : float
        Chance fire/disease spreads from 0 to 1 (0 to 100%).
    pbare : float
        Chance the grid is bare from 0 to 1 (0 to 100%).
    pstart : float
        Chance the grid starts fire from 0 to 1 (0 to 100%).
    wind_dir : str
        The direction of the wind ["up", "down", "left", "right"]
    wind_speed : float
        The speed of wind in kilometers per hour
    prain : float
        Chance it is raining in the iteration
    rain_mask : int
        The size of the rain mask, used to create a rxr mask
    init_cell : tuple
        The initial burning cell position
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition

    Returns
    =======
    int 
        The number of steps it takes until there is no fire
    int
        The number of cells still "Forested"

    '''
    # initialize the forest
    grid = np.zeros((m, n))+2
    # random seed for reproducibility
    if pbare != 0.0:
        # set the bare probability for each grid
        bare_prob = np.random.rand(m, n)
        # set grids that have a value higher than the pbare to status bare
        grid = np.where(bare_prob <= pbare, 1, 2)
    if pstart != 0.0:
        # set the start probability for each grid
        start_prob = np.random.rand(m, n)
        # set grids that have a value higher than the pstart and are not bare grids to burning
        starting_cells = np.where((grid != 1) & (start_prob <= pstart))
        grid = np.where((grid != 1) & (start_prob <= pstart), 3, grid)
    else:
        # set the center of the forest to burning
        grid[init_cell] = 3
        starting_cells = (np.array([init_cell[0]]), np.array([init_cell[1]]))

    # plot the inital condition of the forest
    plot_grid(grid, pspread, pbare, pstart, wind_speed, wind_dir, prain, rain_mask, 0)

    after_rain = np.zeros((rain_mask, rain_mask))
    after_rain.fill(-1)
    # Current step's burning grids
    curQ = queue.Queue()

    for y, x in zip(starting_cells[0], starting_cells[1]):
        curQ.put(np.array([y, x]))
    for i in range(1, maxiter):
        # Next step's burning grids
        nextQ = queue.Queue()
        prev_grid = grid.copy()
        # print(i, prev_grid)
        if np.any(after_rain != -1):
            grid[np.where(rm == True)] = after_rain

        # rain effect
        rm = create_rain_mask(m, n, rain_mask)
        if np.random.rand() <= prain:
            after_rain = grid[np.where(rm == True)]
            grid[np.where(rm == True)] = 0
            # set to rain
            # print(np.where(rm == True))
            # print(grid[np.where(rm == True)])
        else:
            rm = np.zeros_like(grid, dtype=bool)
            after_rain = np.zeros((rain_mask, rain_mask))
            after_rain.fill(-1)
        # plot_grid(grid, pspread, pbare, pstart, psurvive, i)
        while not curQ.empty():
            influencing_pos = curQ.get()
            # print(influencing_pos)
            # spreading left
            if grid[influencing_pos[0], influencing_pos[1]] == 3 and influencing_pos[1]-1 >= 0 and prev_grid[influencing_pos[0], influencing_pos[1]-1] == grid[influencing_pos[0], influencing_pos[1]-1] == 2:
                # Add wind effect on spread probability
                if wind_dir == 'left':
                    speed = wind_speed
                elif wind_dir == 'right':
                    speed = -wind_speed
                else:
                    speed = 0
                if np.random.rand() <= pspread*(1+wind_speed/50):
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos-[0, 1])
                    # set the grid to burning
                    grid[influencing_pos[0], influencing_pos[1]-1] = 3
            # spread right
            if grid[influencing_pos[0], influencing_pos[1]] == 3 and influencing_pos[1]+1 < n and prev_grid[influencing_pos[0], influencing_pos[1]+1] == grid[influencing_pos[0], influencing_pos[1]+1] == 2:
                # Add wind effect on spread probability
                if wind_dir == 'right':
                    speed = wind_speed
                elif wind_dir == 'left':
                    speed = -wind_speed
                else:
                    speed = 0
                if np.random.rand() <= pspread*(1+speed/50):
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos+[0, 1])
                    # set the grid to burning
                    grid[influencing_pos[0], influencing_pos[1]+1] = 3
            # spread up
            if grid[influencing_pos[0], influencing_pos[1]] == 3 and influencing_pos[0]-1 >= 0 and prev_grid[influencing_pos[0]-1, influencing_pos[1]] == grid[influencing_pos[0]-1, influencing_pos[1]] == 2:
                # Add wind effect on spread probability
                if wind_dir == 'down':
                    speed = wind_speed
                elif wind_dir == 'up':
                    speed = -wind_speed
                else:
                    speed = 0
                if np.random.rand() <= pspread*(1+speed/50):
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos-[1, 0])
                    # set the grid to burning
                    grid[influencing_pos[0]-1, influencing_pos[1]] = 3
            # spread down
            if grid[influencing_pos[0], influencing_pos[1]] == 3 and influencing_pos[0]+1 < m and prev_grid[influencing_pos[0]+1, influencing_pos[1]] == grid[influencing_pos[0]+1, influencing_pos[1]] == 2:
                # Add wind effect on spread probability
                if wind_dir == 'up':
                    speed = wind_speed
                elif wind_dir == 'down':
                    speed = -wind_speed
                else:
                    speed = 0
                if np.random.rand() <= pspread*(1+speed/50):
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos+[1, 0])
                    # set the grid to burning
                    grid[influencing_pos[0]+1, influencing_pos[1]] = 3

            # set the grid to bare
            if not rm[influencing_pos[0], influencing_pos[1]]:
                grid[influencing_pos[0], influencing_pos[1]] = 1
        # set the cur_burningQ to next
        # print(grid[np.where(rm == True)])

        curQ = nextQ
        # set burning to forest if it is rained in this iteration
        after_rain[np.where(after_rain == 3)] = 2
        # print(i, grid)
        plot_grid(grid, pspread, pbare, pstart, wind_speed, wind_dir, prain, rain_mask, i)
        if curQ.empty():
            print('Burn completed in', i, 'steps')
            return i, np.count_nonzero(grid == 2)
    return maxiter-1, np.count_nonzero(grid == 2)


def plot_grid(forest, pspread, pbare, pstart, wind_speed, wind_dir, prain, rain_mask, iter):
    '''
    This function plots out the forest given forest grid

    Parameters
    ==========
    forest : np.array
        The forest that is going to be plotted
    pspread : float
        Chance fire/disease spreads from 0 to 1 (0 to 100%).
    pbare : float
        Chance the grid is bare from 0 to 1 (0 to 100%).
    pstart : float
        Chance the grid starts fire from 0 to 1 (0 to 100%).
    wind_speed: float
        The speed of the wind in kilometers per hour
    wind_dir : str
        The direction of the wind ["up", "down", "left", "right"]
    prain : float
        Chance it is raining in the iteration
    rain_mask : int
        The side length of the rain mask
    iter : int
        Current iteration number
    '''

    status = ['Rain', 'Bare', 'Forest', 'FIRE!']
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.set_title(f'Forest Status\npspread = {pspread:.2f} pbare = {pbare:.2f} pstart = {
                 pstart:.2f} prain = {prain:.2f} wind speed = {wind_speed:.2f}km/h wind_dir = {wind_dir} iter = {iter}')

    ax.pcolor(forest, cmap=forest_cmap, vmin=0, vmax=3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')

    for i in range(forest.shape[0]):
        for j in range(forest.shape[1]):
            # Get the value at (i, j)
            value = forest[i, j]
            # Add text annotation
            ax.text(j + 0.5, i + 0.60, status[int(value)],
                    ha='center', va='center',
                    color='black', fontsize=16)
            ax.text(j + 0.5, i + 0.40, f'i , j = {j}, {i}',
                    ha='center', va='center',
                    color='black', fontsize=9)
    fig.savefig(f'{pspread:.2f}_{pbare:.2f}_{pstart:.2f}_{wind_speed:.2f}_{
                wind_dir}_{prain:.2f}_{rain_mask}_iter{iter}.png')
    plt.close('all')


def main():
    # Q1 verificaition
    spread(10, 10, pspread=1, pbare=0, pstart=0,
           wind_dir="up", wind_speed=0, prain=0, rain_mask=3, init_cell=(10//2, 10//2), maxiter=4)
    # Q2
    wind_speeds = np.arange(5, 40, 5)
    for speed in wind_speeds:
        spread(20, 20, pspread=1, pbare=0, pstart=0, wind_dir="down", wind_speed=speed,
               prain=0, rain_mask=0, init_cell=(0, 10), maxiter=20)

    # This distance comes from the last iteration with simulation each wind speed from 5 to 35
    distances = [18, 17, 15, 13, 12, 11, 9]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_title(f'Spread Distance vs. Wind Speed (Wind Direciton = down)')
    ax.plot(wind_speeds, distances, marker='o')
    ax.set_xlabel('Wind Speed (km/h)')
    ax.set_ylabel('Y (km)')

    fig.savefig(f'Q2.png')

    # Q3
    rain_masks = np.arange(1, 8, 1)
    steps_taken = []
    forested_left = []
    for mask in rain_masks:
        step, f_left = spread(15, 15, pspread=1, pbare=0.05, pstart=0.04, wind_dir="up", wind_speed=20,
                              prain=1, rain_mask=mask, init_cell=(7, 7), maxiter=20)
        steps_taken.append(step)
        forested_left.append(f_left)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set_title(f'Wildfire Duration vs. Rain Mask Size')
    ax1.plot(rain_masks, steps_taken, marker='o')
    ax1.set_xlabel('Rain Area Side length (km)')
    ax1.set_ylabel('Wildfire Duration (step)')

    ax2.set_title(f'Forest Left vs. Rain Mask Size')
    ax2.plot(rain_masks, forested_left, marker='o')
    ax2.set_xlabel('Rain Area Side length (km)')
    ax2.set_ylabel('Forested Area (km^2)')
    fig.savefig(f'Q3.png')
    return 0


if __name__ == '__main__':
    main()
