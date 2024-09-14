#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import queue
import sys


plt.style.use('fivethirtyeight')
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])


def fire_spread(m, n, pspread, pbare, pstart, maxiter=4):
    '''
    This function performs a fire/disease spread simultion.

    Parameters
    ==========
    m, n : integer
        Dimension of the grid (the foreset).
    pspread : float
        Chance fire spreads from 0 to 1 (0 to 100%).
    pbare : float
        Chance the grid is bare from 0 to 1 (0 to 100%).
    pstart : float
        Chance the grid starts fire from 0 to 1 (0 to 100%).
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition

    This function plots out the forest status for each step of the simulation

    '''

    # initialize the forest
    forest = np.zeros((m, n))+2
    # random seed for reproducibility
    np.random.seed(1)
    if pbare != 0.0:
        # set the bare probability for each grid
        bare_prob = np.random.rand((m, n))
        # set grids that have a value higher than the pbare to status bare
        forest = np.where(bare_prob <= pbare, 1, 2)
    if pstart != 0.0:
        # set the start probability for each grid
        start_prob = np.random.rand((m, n))
        # set grids that have a value higher than the pstart and are not bare grids to burning
        forest = np.where(forest != 1 and start_prob <= pstart, 3, forest)
    else:
        # set the center of the forest to burning
        forest[m//2, n//2] = 3

    # plot the inital condition of the forest
    plot_forest(forest, pspread, pbare, pstart, 0)

    # Current step's burning grids
    cur_burningQ = queue.Queue()
    cur_burningQ.put(np.array([m//2, n//2]))
    for i in range(1, maxiter):
        # Next step's burning grids
        next_burningQ = queue.Queue()
        while not cur_burningQ.empty():
            burning_pos = cur_burningQ.get()
            # spreading left
            if burning_pos[1]-1 >= 0 and forest[burning_pos[0], burning_pos[1]-1] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    next_burningQ.put(burning_pos-[0, 1])
                    # set the grid to burning
                    forest[burning_pos[0], burning_pos[1]-1] = 3
            # spread right
            if burning_pos[1]+1 < n and forest[burning_pos[0], burning_pos[1]+1] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    next_burningQ.put(burning_pos+[0, 1])
                    # set the grid to burning
                    forest[burning_pos[0], burning_pos[1]+1] = 3
            # spread up
            if burning_pos[0]-1 >= 0 and forest[burning_pos[0]-1, burning_pos[1]] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    next_burningQ.put(burning_pos-[1, 0])
                    # set the grid to burning
                    forest[burning_pos[0]-1, burning_pos[1]] = 3
            # spread down
            if burning_pos[0]+1 < m and forest[burning_pos[0]+1, burning_pos[1]] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    next_burningQ.put(burning_pos+[1, 0])
                    # set the grid to burning
                    forest[burning_pos[0]+1, burning_pos[1]] = 3

            # set the grid to bare after burning
            forest[burning_pos[0], burning_pos[1]] = 1
        # set the cur_burningQ to next
        cur_burningQ = next_burningQ
        plot_forest(forest, pspread, pbare, pstart, i)

        if cur_burningQ.empty():
            print('Burn completed in', i, 'steps')
            break


def plot_forest(forest, pspread, pbare, pstart, iter):
    '''
    This function plots out the forest given forest grid

    Parameters
    ==========
    forest : np.array
        The forest that is going to be plotted

    '''
    status = ['','Bare','Forest','FIRE!']
    fig, ax = plt.subplots(1, 1,figsize=(10,10))
    ax.set_title(f'Forest Status\npspread = {pspread} pbare = {pbare} pstart = {pstart} iter = {iter}')
    ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)
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
                    color='black', fontsize=8)
    fig.savefig(f'{pspread}_{pbare}_{pstart}_iter{iter}.png')
    plt.close('all')


def main():
    # No specific intial values given
    if len(sys.argv) < 2:
        nx, ny = 3, 3  # Number of cells in X and Y direction.
        prob_spread = 1.0  # Chance to spread to adjacent cells.
        prob_bare = 0.0  # Chance of cell to start as bare patch.
        prob_start = 0.0  # Chance of cell to start on fire.
        max_iter = 4  # The maximum number of iterations including initial condition
    # Initial values given
    else:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        prob_spread = float(sys.argv[3])
        prob_bare = float(sys.argv[4])
        prob_start = float(sys.argv[5])
        max_iter = int(sys.argv[6])
    fire_spread(ny, nx, prob_spread, prob_bare, prob_start, max_iter)
    return 0


if __name__ == '__main__':
    main()
