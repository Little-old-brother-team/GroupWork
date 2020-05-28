# Computational Physics

## Assignment Week 13

last update: 26-05-2020

### Group Assignment

1. Please read the following code, write comments and discuss its purpose. 
    ```python
    import numpy as np
    from scipy import fftpack
    N = 201
    dx = (9 - (-9)) / (N - 1)
    x = np.arange(-9, 9+dx, dx)
    psi = np.exp(-(x**2)/3)*np.cos(2*x)
    dpsi = fftpack.diff(psi, period=N*dx)
    ```
2. The Hamiltonian for the quantum harmonic oscillator is given by 
$$
H =\frac{p^2}{2m}+\frac{1}{2}m\omega^2x^2
$$
Substituting this into the time-independent Schrödinger equation $H\psi(x) = E\psi(x)$ and solving for the first excited state, we find that this corresponds to the wavefunction
$$
\psi_1(x) =\left(\frac{4\alpha^3}{\pi}\right)^{1/4}e^{-\alpha x^2/2}
$$
where $\alpha = m\omega/\hbar$.
(1) Using the Riemann sum technique to approximate the Fourier transform, let $\alpha = 1$ and discretise $\psi(x)$ over a reasonable grid size and modify the sample script to numerically calculate the momentum-space wavefunction $\phi(k)$. Remember to choose your domain $−x_\text{max} < x < x_\text{max}$ such that $\psi(\pm x_\text{max}) \approx 0$.
(2) How does your result compare against the exact result? Produce an error plot comparing the exact and numerical results over the domain.
(3) Repeat part (1) for varying grid discretisations $\Delta x$, and plot the resulting maximum absolute error vs $\Delta x$. What do you find? How does the error in this approximation scale with $\Delta x$?

2. (1) Use Fourier differentiation and FFT implementation to calculate the first derivative of $\psi_1(x)$ from problem 1, letting $\alpha = 1$ and using $N = 256$ discretised grid points.
    (2) Modify the formula to instead calculate the second derivative. Use this expression to calculate the second derivative of $\psi_1(x)$.
    (3) Compare your results to the exact solution to the first derivative $\psi_1(x)$ and the second derivative $\psi_1(x)$. What do you find? Repeat your results for various values of $N$, and plot the maximum error vs $N$.
    (4) Extend your analysis to include the method of finite differences. What can you say about the accuracy of Fourier differentiation vs the finite-difference method?
    Hint: the vectorised Cooley–Tukey algorithm can be implemented as follows:
    ```python
    def FFT2_numpy(f):
        # Compute all 2-point DFTs
        N = f.shape[0]
        W = np.array([[1, 1], [1, -1]])
        F = np.dot(W, f.reshape([2, -1]))
        
        # number of remaining Cooley-Tukey stages
        stages = int(np.log2(N))-1
        for i in range(stages):
            k = F.shape[0] # size of the DFTs to combine
            n = F.shape[1] # number of DFTs to combine
            Am = F[:, :n//2] # 'even' terms
            Bm = F[:, n//2:] # 'odd' terms
            twiddle = np.exp(-1.j*np.pi*np.arange(k)/k)[:, None]
            F = np.vstack([Am + twiddle*Bm, Am - twiddle*Bm])
        return F.flatten()
    ```

4. Consider the following code, which uses the Euler method to compute the trajectory of a bouncing ball, assuming perfect reflection at the surface $x = 0$:
   (1) Compile and run the program, and then plot and interpret the output. Are your numerical results physically correct? If not, can you identify a **systematic error** in the algorithm, and then fix the problem?
   (2) Change the time step $dt$ in the code, but keep the same total evolution time. Explain the changes in the results.
    (3) Change the initial velocity and position of the falling ball in the code. Plot and interpret your results.
    (4) Consider inelastic collisions with the table (e.g. the ball loses 10% of its speed after every collision). Plot and interpret your results.
    ```python
    x = 1.0 # initial height of the ball
    v = 0 # initial velocity of the ball
    g = 9.8 # gravitational acceleration
    t = 0 # initial time
    dt = 0.01 # size of time step
    # loop for 300 timesteps
    for steps in range(300):
        t = t + dt
        x = x + v*dt
        v = v - g*dt
        # reflect the motion of the ball
        # when it hits the surface x=0
        if x < 0:
            x = -x
            v = -v
        # write out data at each time step
        with open("bounce.dat", "a+") as f:
            f.write("{} {} {}\n".format(t, x, v))
    ```
    

   
5. Consider the differential equation
    $$
    y^\prime (x) = 1+2xy(x), y(0) = 0
    $$
    The exact solution to this differential equation is given by
    $$
    y(x) =\frac{1}{2}\sqrt{\pi}e^{x^2}\text{erf}(x)
    $$
    where $\text{erf}(x)$ is the error function.
    (1) Find the numerical solution for $0 \leq x \leq 1$, using Rk2 algorithm.   
    (2) Calculate the numeric error in your solution for various values of $\Delta x$, and plot how the error scales with $\Delta x$.
        (i) How does this compare to the Euler method?
        (ii) How does this compare to the <font color=green>leap-frog</font> method?
    (3) Now solve the differential equation using RK4. Analyse your results. How does the error scaling compare to RK2 method?
    Hint： the leap frog method is given as
    $$
    y_{n+1}=y_{n-1}+2\Delta xf(x_n,y_n)
    $$
    We require two initial conditions, $y_0$ and $y_1$. The latter can be estimated from the forward Euler scheme.

6. Use `scipy.integrate.odeint` to solve the following set of nonlinear ODEs.
    $$\begin{aligned}
    \frac{dx}{dt}= a(y - x),\;\frac{dy}{dt}= (c - a)x - xz + cy, \; \frac{dz}{dt}= xy - bz
    \end{aligned}$$
    For the initial conditions, use $x_0 = -10$, $y_0 = 0$, $z_0 = 35$. Setting the initial parameters to $a = 40$, $b = 5$, $c = 35$ gives chaotic solutions like those shown below. Setting $b = 10$ while keeping $a = 40$ and $c = 35$ yields periodic solutions. Take care to choose a small enough     time step (but not too small!).

    <img width=600 src=david_chaos.png>