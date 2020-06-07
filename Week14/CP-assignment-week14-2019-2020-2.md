# Computational Physics

## Assignment Week 14

last update: 02-06-2020

### Group Assignments

1. A classical model in mathematical ecology is the Lotka-Volterra predator-prey model. Consider a simple ecosystem consisting of rabbits that have an infinite supply of food and foxes that prey on the rabbits for their food. This is modeled by a pair of nonlinear, first-order differential equations: 
    $$
    \begin{aligned}
    \frac{dr}{dt}&=2r -\alpha rf, \;r(0) = r_0,\\
    \frac{df}{dt}&=-f + \alpha rf, \;f (0) = f_0\end{aligned} 
    $$
    where $t$ is time, $r(t)$ is the number of rabbits, $f (t)$ is the number of foxes, and $\alpha$ is a positive constant.

    * If $\alpha = 0$, the two populations do not interact, the rabbits do what rabbits do best, and the foxes die off from starvation.

    * If $\alpha > 0$, the foxes encounter the rabbits with a probability that is proportional to the product of their numbers.

    Such an encounter results in a decrease in the number of rabbits and (for less obvious reasons) an increase in the number of foxes. The solutions to this nonlinear system cannot be expressed in terms of other known functions; the equations must be solved numerically. It turns out that the solutions are always periodic, with a period that depends on the initial conditions. In other words, for any $r(0)$ and $f (0)$, there is a value $t = t_p$ when both populations return to their original values. Consequently, for all $t$,

    $$
    \begin{aligned}
    r(t + t_p) = r(t), \;f (t + t_p) = f (t)
    \end{aligned}
    $$

    (1) Compute the solution with $r_0 = 300$, $f_0 = 150$, and $\alpha = 0.01$. You should find that $t_p$ is close to 5. Make two plots, one of $r$ and $f$ as functions of $t$ and one a phase plane plot with $r$ as one axis and $f$ as the other.

    (2) Compute and plot the solution with $r_0 = 15$, $f_0 = 22$, and $\alpha = 0.01$. You should find that $t_p$ is close to 6.62.

    (3) Compute and plot the solution with $r_0 = 102$, $f_0 = 198$, and $\alpha = 0.01$. Determine the period $t_p$ by trial-and-error method.

2. The planets in the solar system whose orbits deviate the most from circular are Mercury and Pluto. For Mercury it was known by the early part of the 19th century that the orientation of the axes of the ellipse that describes its orbit rotate with time. This is known as the **precession of the perihelion of Mercury** (the perihelion is the point in an orbit when a planet is nearest the Sun).  The magnitude of this precession is approximately 566 arseconds per century (an arcsecond is 1/3600 of a degree). That is, Mercury's perihelion makes one complete rotation every 230,000 years. While the precision of both the experimental measurement and the theoretical calculation of the precession of Mercury's perihelion are very impressive, the fact remained that they did not agree. It was realized that this disagreement might be evidence for some interesting new physics, and various solutions to this puzzle were proposed. This troubling discrepancy was not explained until 1917 when Einstein developed the theory of general relativity. The force law predicted by general relativity is 
    $$
    F_G \simeq \frac{GM_SM_M}{r^2}\left(1 +\frac{\alpha}{r^2}\right)
    $$
    where $M_M$ is the mass of Mercury and $\alpha\simeq 1.1\times 10^{-8}$ AU^2^. 
    <br></br>
    **Your task is to calculate the precession of the perihelion of Mercury.** 

    <br></br>

    Since the precision rate is fairly small, we develop the algorithm as follows:

    - We calculate the rate of precession as a function of $\alpha$, with values of $\alpha$ that are much larger than the actual value for Mercury. It turns out that the rate of precession is given by $C\alpha$, where $C$ is a constant whose value we shall estimate. 
    - After we have obtained the values of $C$, we can then estimate the rate of precession for $\alpha = 1.1\times 10^{-8}$ AU^2^, which is the case that we are really interested in. 
  
    <br></br>
    **The** initial conditions are given based on the following considerations (see Figure 1)

    (a) the length of the semimajor axis for Mercury's orbit is $a = 0.39$ AU.
    (b) The corresponding velocity, $v_1$, can be computed based on the conservation of both energy and angular momentum over the course of an orbit. It shows
    $$
    v_1 =\sqrt{\frac{GM_S (1 - e)}{a (1 + e)}}
    $$
    where $e = 0.206$ for Mercury. The distance from Mercury to the Sun is $r_{1}=\left(1+e\right)  a=0.47$ AU.

    <img width=400 src=mercury-initialc.png>

    **Now** that the initial conditions are known, we can simulate the motion of Mercury. The result obtained with $\alpha = 0.01$ is shown in Figure 2. This value of $\alpha$ is much larger than the true value for Mercury. The lines drawn from the origin to the points on each orbit that are farthest from the Sun are very useful, since the angles $\theta$ the make with the $x$ axis are the amounts the orbit has precessed.  The farthest points are found by monitoring the distance of Mercury from the Sun and noting when its time derivative changes from positive to negative. 

    <img width=400 src=mercury-xy.png>

    **The** next step is to calculate the angle $\theta$ of precession as a function of time for a particular value of $\alpha$. We can show that the precession rate $d\theta/dt$ is a constant, the value of which can be calculated with the least $\chi$-square fit.
    
    **As** the next step we repeat the calculation with different values of $\alpha$ and obtain the best-fit precession rate in each case. We then plot the precession rate as a function of $\alpha$. We can find that the precession rate varies linearly with $\alpha$ and we can determine the coefficient of proportionality by using the least $\chi$-square fit. The precession rate of Mercury can be extrapolated with $\alpha = 1.1 \times 10^{-8}$ AU^2^.

    Hint:  Choose yr and AU as the time and length units. It can be shown that 
    $$
    GM_S=4\pi^2 \text{ AU}^3/\text{yr}^2
    $$

3. Consider the IVP
    $$
    \frac{dy}{dt}  + 100y = (99t + 1) e^{-t} ,\; y(0) = 1 
    $$
    The exact solution is 
    $$
    y = e^{-100t} + te^{-t} 
    $$
    so that the first term quickly becomes negligible relative to the second term, but continues to govern stability.

    Apply forward Euler's method with $h = 0.1$ and plot the solution estimates versus $0 \leq t \leq 1$. Repeat with $h = 0.01$ and plot versus $0\leq t \leq 5$. Discuss on your results.

4. Use shooting method to solve the given boundary problem using step sizes $h=0.05$. Plot your approximate solutions and the actual solutions on the same coordinate system.
   
    (1) $y^{\prime\prime}=2y^{\prime}-y+x^{2}-1$ over $\left[  0,1\right]$ with $y\left(  0\right)  =5$ and $y\left(  1\right)  =10$, exact solution $y\left(  t\right)   =x^{2}+4x+5$

    (2) $y^{\prime\prime}+\left(  1/x\right)  y^{\prime}+(1-1/\left(4x^{2}\right)  )y=0$ over $\left[  1,6\right]$ with $y\left(  1\right)  =1$ and $y\left(  6\right)  =0$, exact solution $y\left(  x\right)  =\dfrac{0.2913843206\cos\left(  x\right)  +1.001299385\sin\left(  x\right)  }{\sqrt{x}}$
