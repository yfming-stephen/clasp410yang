import matplotlib.pyplot as plt
import scipy as sp


def dNdt_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra predator-prey coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]
    return dN1dt, dN2dt


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    return dN1dt, dN2dt


def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    <Your good docstring here>
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float
        Stepsize for time in years
    t_final : float
        Integrate until this value is reached, in years

    Returns
    -------
    time : list
        Time elapsed in years
    N1, N2
    '''
    # Important code goes here #
    N1 = []
    N2 = []
    time = list(range(int(t_final/dT)))
    N1.append(N1_init)
    N2.append(N2_init)
    for i in range(1, len(time)):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]], a, b, c, d)
        N1.append(N1[i-1]+dN1*dT)
        N2.append(N2[i-1]+dN2*dT)
        time[i] = i*dT
    return time, N1, N2


def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=[a, b, c, d], method='DOP853', max_step=dT)
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2


def main():
    a = 1.5
    b = 2
    c = 1
    d = 3
    N1_init = 0.5
    N2_init = 0.1

    # Q1
    time_euler, N1_euler, N2_euler = euler_solve(dNdt_comp, N1_init, N2_init, 1, 1000, a, b, c, d)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init, N2_init, 1, 1000, a, b, c, d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(time_euler, N1_euler, label=r'$N_1$ Euler', color='blue', linestyle='-', linewidth=2)
    ax1.plot(time_euler, N2_euler, label=r'$N_2$ Euler', color='red', linestyle='-', linewidth=2)
    ax1.plot(time_rk8, N1_rk8, label=r'$N_1$ RK8', color='blue', linestyle='dotted', linewidth=2)
    ax1.plot(time_rk8, N2_rk8, label=r'$N_2$ RK8', color='red', linestyle='dotted', linewidth=2)
    ax1.set_title('Lotka-Volterra Competition Model')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Population/Carrying Cap.')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    fig.text(0.5, 0.04, f'Coefficients: a={a}, b={b}, c={c}, d={d}', ha='center', fontsize=12)

    time_euler, N1_euler, N2_euler = euler_solve(dNdt_prey, N1_init, N2_init, 0.01, 100, a, b, c, d)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_prey, N1_init, N2_init, 0.01, 100, a, b, c, d)
    ax2.plot(time_euler, N1_euler, label=r'$N_1$ (Prey) Euler ', color='blue', linestyle='-', linewidth=2)
    ax2.plot(time_euler, N2_euler, label=r'$N_2$ (Predator) Euler', color='red', linestyle='-', linewidth=2)
    ax2.plot(time_rk8, N1_rk8, label=r'$N_1$ (Prey) RK8', color='blue', linestyle='dotted', linewidth=2)
    ax2.plot(time_rk8, N2_rk8, label=r'$N_2$ (Predator) RK8', color='red', linestyle='dotted', linewidth=2)
    ax2.set_title('Lotka-Volterra Prey-Predator Model')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Population/Carrying Cap.')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    plt.show()

    # Q2 & 3
    time_euler, N1_euler, N2_euler = euler_solve(dNdt_comp, N1_init, N2_init, 1, 1000, a, b, c, d)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init, N2_init, 1, 1000, a, b, c, d)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.plot(time_euler, N1_euler, label=r'$N_1$ Euler', color='blue', linestyle='-', linewidth=2)
    ax1.plot(time_euler, N2_euler, label=r'$N_2$ Euler', color='red', linestyle='-', linewidth=2)
    ax1.plot(time_rk8, N1_rk8, label=r'$N_1$ RK8', color='blue', linestyle='dotted', linewidth=2)
    ax1.plot(time_rk8, N2_rk8, label=r'$N_2$ RK8', color='red', linestyle='dotted', linewidth=2)
    ax1.set_title('Lotka-Volterra Competition Model')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Population/Carrying Cap.')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    fig.text(0.5, 0.04, f'Coefficients: a={a}, b={b}, c={c}, d={d}', ha='center', fontsize=12)
    plt.show()
    fig2, (ax2, phase) = plt.subplots(1, 2, figsize=(16, 8))
    time_euler, N1_euler, N2_euler = euler_solve(dNdt_prey, N1_init, N2_init, 0.01, 100, a, b, c, d)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_prey, N1_init, N2_init, 0.01, 100, a, b, c, d)
    ax2.plot(time_euler, N1_euler, label=r'$N_1$ (Prey) Euler ', color='blue', linestyle='-', linewidth=2)
    ax2.plot(time_euler, N2_euler, label=r'$N_2$ (Predator) Euler', color='red', linestyle='-', linewidth=2)
    ax2.plot(time_rk8, N1_rk8, label=r'$N_1$ (Prey) RK8', color='blue', linestyle='dotted', linewidth=2)
    ax2.plot(time_rk8, N2_rk8, label=r'$N_2$ (Predator) RK8', color='red', linestyle='dotted', linewidth=2)
    ax2.set_title('Lotka-Volterra Prey-Predator Model')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Population/Carrying Cap.')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    phase.plot(N1_euler, N2_euler, label=r'Euler', color='blue', linestyle='-', linewidth=2)
    phase.plot(N1_rk8, N2_rk8, label=r'RK8', color='red', linestyle='-', linewidth=2)
    phase.grid(True)
    phase.set_title('Phase Diagram')
    phase.set_xlabel('Prey Population')
    phase.set_ylabel('Predator Population.')
    phase.legend(loc='upper left')
    fig2.text(0.5, 0.04, f'Coefficients: a={a}, b={b}, c={c}, d={d}', ha='center', fontsize=12)
    plt.show()


if __name__ == '__main__':
    main()
