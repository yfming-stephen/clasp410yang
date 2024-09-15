#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do the following

Usage: python lab1.py [ny] [nx] [prob_spread] [prob_bare] [prob_start] [maxiter]
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import queue
import sys


plt.style.use('fivethirtyeight')
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
disease_cmap = ListedColormap(['orange','tan', 'darkgreen', 'crimson'])



def spread(m, n, pspread, pbare, pstart, psurvive = -1.0 ,maxiter=4):
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
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition

    Returns
    =======
    int 
        The number of steps it takes until there is no fire

    '''

    # initialize the forest
    grid = np.zeros((m, n))+2
    # random seed for reproducibility
    np.random.seed(2)
    if pbare != 0.0:
        # set the bare probability for each grid
        bare_prob = np.random.rand(m, n)
        # set grids that have a value higher than the pbare to status bare
        grid = np.where(bare_prob <= pbare, 1, 2)
    if pstart != 0.0:
        # set the start probability for each grid
        start_prob = np.random.rand(m, n)
        # set grids that have a value higher than the pstart and are not bare grids to burning
        grid = np.where(grid != 1 and start_prob <= pstart, 3, grid)
    else:
        # set the center of the forest to burning
        grid[m//2, n//2] = 3

    # plot the inital condition of the forest
    plot_grid(grid, pspread, pbare, pstart, psurvive, 0)

    # Current step's burning grids
    curQ = queue.Queue()
    curQ.put(np.array([m//2, n//2]))
    for i in range(1, maxiter):
        # Next step's burning grids
        nextQ = queue.Queue()
        while not curQ.empty():
            influencing_pos = curQ.get()
            # spreading left
            if influencing_pos[1]-1 >= 0 and grid[influencing_pos[0], influencing_pos[1]-1] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos-[0, 1])
                    # set the grid to burning
                    grid[influencing_pos[0], influencing_pos[1]-1] = 3
            # spread right
            if influencing_pos[1]+1 < n and grid[influencing_pos[0], influencing_pos[1]+1] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos+[0, 1])
                    # set the grid to burning
                    grid[influencing_pos[0], influencing_pos[1]+1] = 3
            # spread up
            if influencing_pos[0]-1 >= 0 and grid[influencing_pos[0]-1, influencing_pos[1]] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos-[1, 0])
                    # set the grid to burning
                    grid[influencing_pos[0]-1, influencing_pos[1]] = 3
            # spread down
            if influencing_pos[0]+1 < m and grid[influencing_pos[0]+1, influencing_pos[1]] == 2:
                if np.random.rand() <= pspread:
                    # put the grid into next step burning grids
                    nextQ.put(influencing_pos+[1, 0])
                    # set the grid to burning
                    grid[influencing_pos[0]+1, influencing_pos[1]] = 3
            # if psurvive is included in the model
            if  psurvive != -1 and np.random.rand() <= 1-psurvive:
                # set the grid to death
                grid[influencing_pos[0], influencing_pos[1]] = 0
            else:
                # set the grid to bare
                grid[influencing_pos[0], influencing_pos[1]] = 1
        # set the cur_burningQ to next
        curQ = nextQ
        plot_grid(grid, pspread, pbare, pstart, psurvive, i)

        if curQ.empty():
            print('Burn completed in', i, 'steps')
            return i
    return maxiter-1

def plot_grid(forest, pspread, pbare, pstart, psurvive, iter):
    '''
    This function plots out the forest given forest grid

    Parameters
    ==========
    forest : np.array
        The forest that is going to be plotted
    '''
    if psurvive!=-1:
        status = ['Insky','Immune','Healty','Sick']
    else:
        status = ['','Bare','Forest','FIRE!']
    fig, ax = plt.subplots(1, 1,figsize=(10,10))
    if psurvive!=-1:
        ax.set_title(f'Population Status\npspread = {pspread:.2f} pbare = {pbare:.2f} pstart = {pstart:.2f} iter = {iter}')
    else:
        ax.set_title(f'Forest Status\npspread = {pspread:.2f} pbare = {pbare:.2f} pstart = {pstart:.2f} iter = {iter}')
    if psurvive!=-1:
        ax.pcolor(forest, cmap=disease_cmap, vmin=0, vmax=3)
    else:
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
                    color='black', fontsize=14)
            ax.text(j + 0.5, i + 0.40, f'i , j = {j}, {i}', 
                    ha='center', va='center', 
                    color='black', fontsize=8)
    fig.savefig(f'{pspread:.2f}_{pbare:.2f}_{pstart:.2f}_{psurvive:.2f}_iter{iter}.png')
    plt.close('all')

def varying_fire_spread(m, n, max_iter):
    proba = np.arange(0,1,0.05)
    nsteps = np.zeros_like(proba)
    for i, p in enumerate(proba):
        print(f"Burning for pspread = {p}")
        nsteps[i] = spread(m,n,p,0.0,0.0,maxiter=max_iter)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(proba,nsteps)
    ax.set_xlabel('Pspread')
    ax.set_ylabel('#Step')
    fig.savefig(f'Varying_Pspread.png')
    for i, p in enumerate(proba):
        print(f"Burning for pbare = {p}")
        nsteps[i] = spread(m,n,1.0,p,0.0,maxiter=max_iter)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(proba,nsteps)
    ax.set_xlabel('Pbare')
    ax.set_ylabel('#Step')
    fig.savefig(f'Varying_Pbare.png')

def varying_disease_spread(m, n, max_iter):
    proba = np.arange(0,1,0.05)
    nsteps = np.zeros_like(proba)
    for i, p in enumerate(proba):
        print(f"Burning for pspread = {p}")
        # varying probability of survival
        nsteps[i] = spread(m,n,0.7,0.0,0.0,p,max_iter)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(proba,nsteps)
    ax.set_xlabel('Psurvive')
    ax.set_ylabel('#Step')
    fig.savefig(f'Varying_Pspread.png')
    for i, p in enumerate(proba):
        print(f"Burning for pbare = {p}")
        # varying probability of bare (baren or vaccine or immuned)
        nsteps[i] = spread(m,n,1.0,p,0.0,0.0,max_iter)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(proba,nsteps)
    ax.set_xlabel('Pbare')
    ax.set_ylabel('#Step')
    fig.savefig(f'Varying_Pbare.png')

    for i, p in enumerate(proba):
        print(f"Burning for pbare = {p}")
        # varying probability of bare (baren or vaccine or immuned)
        nsteps[i] = spread(m,n,1.0,p,0.0,p,max_iter)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(proba,nsteps)
    ax.set_xlabel('Psurvive & Pbare')
    ax.set_ylabel('#Step')
    fig.savefig(f'Varying_Psurvive_Pbare_.png')

def main():
    # No specific intial values given
    if len(sys.argv) < 2:
        nx, ny = 3, 3  # Number of cells in X and Y direction.
        prob_spread = 1.0  # Chance to spread to adjacent cells.
        prob_bare = 0.0  # Chance of cell to start as bare patch.
        prob_start = 0.0  # Chance of cell to start on fire.
        prob_surv = -1.0
        max_iter = 4  # The maximum number of iterations including initial condition
    # Initial values given
    else:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        prob_spread = float(sys.argv[3])
        prob_bare = float(sys.argv[4])
        prob_start = float(sys.argv[5])
        prob_surv = float(sys.argv[6])
        max_iter = int(sys.argv[7])
    # spread(ny, nx, prob_spread, prob_bare, prob_start, max_iter)
    # varying_fire_spread(8,8,max_iter)
    varying_disease_spread(10,10,max_iter=100)
    return 0


if __name__ == '__main__':
    main()
