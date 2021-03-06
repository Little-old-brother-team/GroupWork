# Computational Physics

## Assignment Week 16

last update: 16-06-2020

### Group Assignments

1. According to the central limit theorem the sum of $N$ random variables $x_{k}$ with means $E_k$ and variances $\sigma_{k}^{2}$ will have a normal distribution with mean $\sum_{k=1}^{N}E_{k}$; and variance $\sigma^{2}=\sum_{k=1}^{N}\sigma_{k}^{2}$, if $N$ is large (theoretically infinite).    
  A uniform variate between -1 and +1 has a variance of 1/3 so that sums of 27 such deviates should have a normal distribution with mean zero and standard deviation $\sigma=3$.   
  Using a uniform random number generator find 100 000 such sums and find the proportions within $\sigma$, 2$\sigma$ and 3$\sigma$ from the mean. For a normal distribution these should be 0.6826, 0.9545 and 0.9973.
    
2. Apply the inverse transformation method to generate a series of random variables $x$ with a distribution function $p(x) = \dfrac{1}{ 2} \sin\pi x$ in the range $0 \leq x \leq 1$. 
   Plot the historgram and compare your result with the theoretical value.
   
3. Investigate the behavior of a random walk in which the probabilities for different step directions are not equal. For example, consider a one-dimensional walk with $p_{\text{left}}=0.25$ and $p_{\text{right}}=0.75$. In what sense is the motion still diffusive?

4. Write a program to simulate a 2-D random walk.

    (1) To increase the amount of randomness, independently choose random values for $\Delta x^{\prime}$ and $\Delta y^{\prime}$ in the range $\left[
    -1,1\right]$. Then normalize them so that each step is of unit length
    $$
     \Delta x=\frac{1}{L}\Delta x^{\prime},\text{ }\Delta y=\frac{1}{L}\Delta
    y^{\prime},\text{ }L=\sqrt{\Delta x^{\prime2}+\Delta y^{\prime2}}%
    $$
    (2) Using a plotting program to draw maps of several independent random walks, each of 1000 steps. Comment on whether these look like what you would expect of a random walk.

    (3) If you have your walker taking $N$ steps in a single trial, then conduct a total number $K\simeq\sqrt{N}$ trials. Each trial should have $N$ steps and start with a different seed. 
    Calculate the mean square distance $R^{2}$ for each trial and then take the average of $R^{2}$ for all your $K$ trials: 
    $$
    \left\langle R^{2}\left(  N\right)  \right\rangle =\frac{1}{K}\sum_{k=1}%
    ^{K}R_{\left(  k\right)  }^{2}\left(  N\right)
    $$
    (4) Plot the root-mean-square distance $R_{\text{rms}}=\left\langle R^{2}\left(  N\right)\right\rangle$ as a function of $\sqrt{N}$. Values of $N$ should start with a small number, where $R\simeq\sqrt{N}$ is not expected to be accurate, and end at a quite large value, where two or three places of accuracy should be expected on the average.

    (5) Check to see if you obtained the expected statistical "error band" of $\pm\sqrt{R}\simeq N^{1/4}$ for large values of $N$.
    (6) (Optional) Check the validity of the assumptions made in deriving the theoretical result by checking how well 
    $$
    \frac{\left\langle \Delta x_{i}\Delta x_{j\neq i}\right\rangle }{R^{2}}%
    \simeq\frac{\left\langle \Delta x_{i}\Delta y_{j}\right\rangle }{R^{2}}\simeq0 
    $$
    Do your checking for both a single (long) run and for the average over trials.

5. Imagine a closed box that is divided into two parts of equal volume. The left half contains a gas of $N$ identical particles and the right half is initially empty. We then make a small hole in the partition between the two halves. We know that after some time, the average number of particles in each half of the box will become $N$, and we say that the system has reached equilibrium.

    ​	Let $n$ be the number of particles in the left part of the box. Because each particle has the same chance to go through the hole, the probability per unit time that a particle moves from left to right equals the number of particles on the left divided by the total number of particles. Write the program so that the following algorithm is implemented:

    -   Generate a random number $r$ from a uniformly distributed set of random numbers in the interval $0\leq r <1$.

    -   If $r\leq p=n/N$, move a particle from left to right, that is $n\rightarrow n-1$; otherwise, $n\rightarrow n+1$.

    (1) Does the time dependence of $n$ appear to be deterministic for sufficient large $N$? 

    Hint: $n(t)=\dfrac{N}{2}\left(1+e^{-2t/N}\right)$

    (2) How long does it take for the system to reach equilibrium? How does this time depend on the number of particles? Choose the total number of particles $N$ to be $N=8,16,64,400,800$, and $3600$.

    (3) A measure of the equilibrium fluctuations is the mean square fluctuation $\Delta n^2$, which is defined as 
    $$
    \begin{aligned}
          \Delta n^2=\langle (n-\langle n\rangle)^2\rangle=\langle n^2\rangle-\langle n\rangle^2
          \end{aligned} 
    $$
    The brackets $\langle \cdots\rangle$ denotes an average taken after the system has reached equilibrium. Modify your program so that averages are taken after equilibrium has been reached. Run for a time that is long enough to obtain meaningful results. How does the relative fluctuations, $\Delta n/\langle n\rangle$, depend on $N$?
    
6. One way to calculate $\pi$ is by randomly throwing a dart into the unit square defined by $x \in [0, 1]$ and $y\in [0, 1]$ in the $xy$ plane. The chance of the dart landing inside the unit circle centered at the origin of the coordinates is $\pi/4$, from the comparison of the areas of one quarter of the circle and the unit square. Write a program to calculate $\pi$ in such a manner.
   In your program, repeat this experiment many times (meaning many 5000-sample trials, say 1000 such trials) and use your results to estimate the statistical uncertainty (standard deviation of the mean result) in your calculation.