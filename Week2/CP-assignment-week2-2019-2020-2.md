# Computational Physics

## Assignment Week 2

last update: 10-03-2020

### Individual Assignment

1. Consider the matrix list `x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`. 
   (a) Write a list comprehension to extract the last column of the matrix `[3, 6, 9]`. 
   (b) Write another list comprehension to create a vector of twice the square of the middle column `[8, 50, 128]`.
 
2. Perform the following tasks with `NumPy` arrays. All of them can be done (elegantly) in 1 to 3 lines.
    (a) Create an 8$\times$8 array with ones on all the edges and zeros everywhere else.
    (b) Create an 8$\times$8 array of integers with a checkerboard pattern of ones and zeros.
    (c) Given the array `c = np.arange(2, 50, 5)`, make all the numbers not divisible by 3 negative.
    (d) Find the size, shape, mean, and standard deviation of the arrays you created in parts (a)–(c).
    

3. Write a program to calculate the perimeter $p$ of an $n$-gon inscribed inside a sphere of diameter 1. Find $p$ for $n = 3, 4, 5, 100, 10,000$, and $1,000,000$. Your answers should be

    | $n$ | $p$ | $n$ | $p$ |
    | --- | --- | ---: |---|
    | 3 | 2.59807621135 | 100 | 3.14107590781|
    | 4 | 2.82842712475| 10,000| 3.14159260191|
    | 5 |2.93892626146 | 1,000,000 | 3.14159265358|
    (a) Please print out your results of $p$ as formatted strings on screen, keeping 12 significant figures for each $p$. 
    (b) Save the values of $n$ and $p$ in a *csv* file.
    (c) Save the value of $n$ and $p$ as a `NumPy` array in a *npz* file. 
    (d) Use the module `matplotlib.pyplot` to make a semi-log plot of $p$ versus $n$. What conclusion can you make from the plot?

4. The position of a ball at time $t$ dropped with zero initial velocity from a height $h_0$ is given by 
$$
y = h_0-\cfrac{1}{2}gt^2
$$
where $g = 9.8$ m/s$^2$. Suppose $h_0 = 10$ m.
    (a) Find the sequence of times when the ball passes each half meter assuming the ball is dropped at $t = 0$. It should yield the following results for the $y$ and $t$ arrays:
    ```
    In [1]: y
    Out[1]: array([10., 9.5, 9., 8.5, 8.0, 7.5, 7., 6.5, 6,. 5.5, 5., 4.5, 4., 3.5, 3., 2.5, 2., 1.5, 1.0, 0.5])
    In [2]: t
    OUt[2]: array([0. , 0.31943828, 0.45175395, 0.55328334, 0.63887656, 0.71428571, 0.7824608 , 0.84515425, 0.9035079, 0.95831485, 1.01015254, 1.05945693, 1.10656667, 1.15175111, 1.19522861, 1.23717915, 1.27775313, 1.31707778, 1.35526185, 1.39239919])
    ```
    Once you have created the arrays `y` and `t`, try with `list(zip(t, y))`. Can you explain the result? 
    (b) Recalling that the average velocity over an interval $\Delta t$ is defined as $\bar{v} = \Delta y/\Delta t$, find the average velocity for each time interval in the part (a) using `NumPy` arrays.
    (c) Calculate the acceleration as a function of time using the formula $\bar{a} = \Delta v/\Delta t$. Take care, as you will need to define a new time array `t_mid` that corresponds to the times where the velocities are calculated, which is midway between the times in the original time array. 
    (d) Please use the module `matplotlib.pyplot` to make a plot of `v` versus `t` and a plot of `a` versus `t_mid`. Can you justify your solutions?

### Group Assignment

1. Write a program that determines the day of the week for any given calendar date after January 1, 1900, which was a Monday. Test that your program gives the answers tabulated below.
    | Date | Weekday|
    | --- | --- |
    |January 1, 1900 | Monday|
    |June 28, 1919 | Saturday|
    |January 30, 1928 |Tuesday|
    |December 5, 1933 |Tuesday|
    |February 29, 1948| Sunday|
    |March 1, 1948 |Monday|
    |January 15, 1953 |Thursday|
    |November 22, 1963| Friday|
    |June 23, 1993 |Wednesday|
    |August 28, 2005| Sunday|
    |May 16, 2111 |Saturday|
    You are required to consider three different methods:
    - Method 1: You design your own algorithm to determine the passing days between two arbitrary dates. Your program will need to take into account leap years, which occur in every year that is divisible by 4, except for years that are divisible by 100 but are not divisible by 400. For example, 1900 was not a leap year, but 2000 was a leap year. 
    - Method 2: You can import the module  `datetime`. An example: `today = datetime.date(2020,3,10)`
    - Method 3: You can use the data structure `datetime64` from the module `Numpy`, for example `today = np.datetime64('2020-03-10')`.    
  
    Hint: `str(defined_object)` is a powerful tool to convert a defined object such as integer, float, boolean, list, tuple, and `NumPy` array to a string. You can `split` the string, transform it to a list of strings, and then capture useful information from the list.
