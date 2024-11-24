#!/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6378000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def temp_hot(lats_in):
    '''
    Create a temperature profile for "hot" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       60, 60, 60, 60, 60, 60, 60, 60], dtype=float)

    return T_warm


def temp_cold(lats_in):
    '''
    Create a temperature profile for "cold" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-60, -60, -60, -60, -60, -60, -60, -60, -60, -60,
                       -60, -60, -60, -60, -60, -60, -60, -60], dtype=float)

    return T_warm


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                   debug=False, albedo=0.3, emiss=1, S0=1370):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off
        insolation.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation:
    insol = insolation(S0, lats)

    # Create initial condition:
    Temp = temp_warm(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)

        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        # print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        # print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp
# snowball earth model with dynamic albedo


def dynamic_albedo(temp, freeze_temp=0, melt_temp=5, ice_albedo=0.65, land_albedo=0.25):
    '''
    Calculate albedo based on current temperature in its latitude

    Parameters
    ----------
    temp : np.array
        Temperature at each latitude
    freeze_temp : float
        Temperature for water to freeze
    melt_temp : float
        Temperature for ice to melt
    ice_albedo : float
        Albedo for iced areas
    land_albedo : float
        Albedo for land
    Return
    ------
    albedo : np.array
        Albedo at each latitude
    '''
    albedo = np.zeros_like(temp)
    for i in range(temp.shape[0]):
        if temp[i] <= freeze_temp:
            albedo[i] = ice_albedo
        elif freeze_temp < temp[i] < melt_temp:
            # when the temperature is between fully melted and freezed, give partial albedo higher than land
            albedo[i] = land_albedo + (ice_albedo-land_albedo)*(melt_temp-temp[i]) / (melt_temp-freeze_temp)
        else:
            albedo[i] = land_albedo
    return albedo


def snowball_earth_dynamic(temp, nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                           debug=False, emiss=1, S0=1370):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    temp : funciton
        Takes lats grids and returns an array of temperature based on latitude
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off
        insolation.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation:
    insol = insolation(S0, lats)

    # Create initial condition:
    Temp = temp(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
        albedo = dynamic_albedo(Temp)
        # print(Temp)
        # print(albedo)
        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        # print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        # print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def snowball_earth_solar(init_temp, gamma=0.4, nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                         debug=False, emiss=1, S0=1370):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    init_temp : Numpy array
        An array of temperature based on latitude
    gamma : float
        solar multiplier that represents impact of solar forcing on snowball earth
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off
        insolation.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Generate insolation, factored by gamma
    insol = gamma * insolation(S0, lats)
    # print(insol)

    # Create initial condition:
    Temp = init_temp
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    #
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
        albedo = dynamic_albedo(Temp)
        # print(Temp)
        # print(albedo)
        # Apply insolation and radiative losses:
        # print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        # print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        # print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def q1_validation(tstop=10000):
    '''
    Reproduce example plot in handout.

    Using our DEFAULT values (grid size, diffusion, etc.) plot:
        - Plot initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)

    # Get diffusion + sphercorr + radiative terms:
    lats, t_rad = snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion', color='red')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.', color='orange')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('Warm Earth Validation')
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig('q1_validation.png')


def q2_param(tstop=10000):
    '''
    Find parameters lambda and emissivity that best reproduce 
    the Warm-Earth Equillibrium
    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_warm(lats)
    lamdas = np.arange(0, 160, 5)
    emissivity = np.arange(0, 1, 0.05)
    for lam in lamdas:
        lats, t_rad = snowball_earth(tstop=tstop, lam=lam)

        # Create figure and plot!
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(lats, initial, label='Warm Earth Init. Cond.')
        ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
        ax.set_xlabel('Latitude (0=South Pole)')
        ax.set_ylabel('Temperature ($^{\circ} C$)')

        ax.legend(loc='best')

        fig.tight_layout()
        fig.savefig(f'q2_param_lam={lam}.png')
    for emiss in emissivity:
        lats, t_rad = snowball_earth(tstop=tstop, emiss=emiss, lam=30)
        # Create figure and plot!
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(lats, initial, label='Warm Earth Init. Cond.')
        ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
        ax.set_xlabel('Latitude (0=South Pole)')
        ax.set_ylabel('Temperature ($^{\circ} C$)')

        ax.legend(loc='best')

        fig.tight_layout()
        fig.savefig(f'q2_param_emiss={emiss}.png')


def q2_comb(tstop=10000):
    '''
    Plot out the combination that best reproduces the Warm-Earth Equillibrium

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_warm(lats)
    lats, t_rad = snowball_earth(tstop=tstop, lam=30, emiss=0.74)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('Lambda Emissivity Best Comb')
    fig.text(0.5, 0.005, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'q2_comb.png')


def q3_hot(tstop=10000):
    '''
    Plot out the Hot-Earth Equillibrium

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_hot(lats)
    lats, t_rad = snowball_earth_dynamic(temp_hot, tstop=tstop, lam=30, emiss=0.74)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Hot Earth Init. Cond.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('Hot Earth Temp Profile')
    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'q3_hot.png')


def q3_cold(tstop=10000):
    '''
    Plot out the Cold-Earth Equillibrium

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_cold(lats)
    lats, t_rad = snowball_earth_dynamic(temp_cold, tstop=tstop, lam=30, emiss=0.74)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Hot Earth Init. Cond.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title('Cold Earth Temp Profile')
    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'q3_cold.png')


def q3_freeze(tstop=10000):
    '''
    Plot out the reproduces the Warm-Earth Equillibrium, with albedo 0.6

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    initial = temp_warm(lats)
    lats, t_rad = snowball_earth(tstop=tstop, lam=30, emiss=0.74, albedo=0.6)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Hot Earth Init. Cond.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative', color='green')
    ax.set_title('Warm Earth Flash Freeze Temp Profile')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'q3_flash_freeze.png')


def q4_solar(tstop=10000):
    '''
    Plot out the Average Global Temperature vs. Gamma

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    temp = temp_cold(lats)
    gamma = 0.4
    gamma_list1 = []
    ave_temp_list1 = []
    # first simulation
    while gamma <= 1.4:
        lats, temp = snowball_earth_solar(temp, gamma, lam=30, emiss=0.74)
        ave_temp = np.mean(temp)
        ave_temp_list1.append(ave_temp)
        gamma_list1.append(gamma)
        gamma += 0.05

    gamma = 1.4
    gamma_list2 = []
    ave_temp_list2 = []
    # second simulation
    while gamma >= 0.4:
        lats, temp = snowball_earth_solar(temp, gamma, lam=30, emiss=0.74)
        ave_temp = np.mean(temp)
        # print(ave_temp)
        ave_temp_list2.append(ave_temp)
        gamma_list2.append(gamma)
        gamma -= 0.05

    # Create figure and plot!
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    ax1.plot(gamma_list1, ave_temp_list1, marker='o', label='Ave Global Temp vs. gamma')
    ax2.plot(gamma_list2, ave_temp_list2, marker='o', label='Ave Global Temp vs. gamma', color='green')
    ax1.set_title('Ave Global Temp vs. gamma (increasing)')
    ax2.set_title('Ave Global Temp vs. gamma (decreasing)')

    ax1.set_xlabel('Gamma')
    ax1.set_ylabel('Ave Global Temperature ($^{\circ} C$)')
    ax2.set_xlabel('Gamma')
    ax2.set_ylabel('Ave Global Temperature ($^{\circ} C$)')
    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax2.invert_xaxis()
    fig.tight_layout()
    fig.savefig(f'q4_ave_temp.png')


def q4_explore_gamma(tstop=10000):
    '''
    Plot out the Average Global Temperature vs. Gamma

    Parameters
    ----------
    tstop : int:
        Max years before simulation stops.
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get initial condition
    temp = temp_cold(lats)
    gamma = 0.4
    gamma_list1 = []
    ave_temp_list1 = []
    initial = None
    transit_temp = None
    transit_lats = None
    # first simulation
    while gamma <= 2.05:
        if 0 < gamma - 1.4 < 0.01:
            print(gamma)
            initial = temp
        lats, temp = snowball_earth_solar(temp, gamma, lam=30, emiss=0.74)
        if gamma - 1.4 < 0.1:
            transit_temp = temp
            transit_lats = lats
        ave_temp = np.mean(temp)
        ave_temp_list1.append(ave_temp)
        gamma_list1.append(gamma)
        gamma += 0.05

    # Create figure and plot!
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    ax1.plot(gamma_list1, ave_temp_list1, marker='o', label='Ave Global Temp vs. gamma')

    ax1.set_title('Ave Global Temp vs. gamma (increasing)')

    ax1.set_xlabel('Gamma')
    ax1.set_ylabel('Ave Global Temperature ($^{\circ} C$)')

    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax1.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f'q4_explore_gamma.png')

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(lats, initial, label='Init. Cond.')
    ax.plot(transit_lats, transit_temp, label='Diffusion + Sphere. Corr. + Radiative, gamma = 1.4', color='green')
    ax.set_title('Earth Temp Profile')
    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    fig.text(0.5, 0.01, f"Thermal Diffusivity = {30}m^2/s, Emissivity = {0.74}", ha="center", fontsize=12)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(f'q4_transition.png')


def main():
    q1_validation()
    q2_param()
    q2_comb()
    q3_hot()
    q3_cold()
    q3_freeze()
    q4_solar()
    q4_explore_gamma()


if __name__ == '__main__':
    main()
