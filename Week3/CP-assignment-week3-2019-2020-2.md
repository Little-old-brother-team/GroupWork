# Computational Physics

## Assignment Week 3

last update: 17-03-2020

### Individual Assignment

1. Consider the 32-bit single-precision floating-point number $A$

    |  | $s$  | $e$  |$f$ |
    | --- | ---| --- | --- |
    | Bit position | 31   | 30 $\dots$ 23 | 22 $\dots$ 0                   |
    | value  | 0    | 0000 1110   | 1010 0000 0000 0000 0000 000 |

    Determine the full value of $A$.

2. Write a program to determine the under- and overflow limits.

3. Write a program to determine your machine precision for double-precision floats.

### Group Assignment

1. The quadratic equation $ax^{2}+bx+c=0$ has an analytic solution that can be written as either 
    $$
    x_{1,2}=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}\text{ or }x_{1,2}=\frac{-2c}{-b\pm\sqrt{b^{2}-4ac}}
    $$
    When $b^{2}\gg4ac$, the square root and its preceding term nearly cancel for one of the roots. Consequently, subtractive cancellation (and consequently an increase in error) arises. Consider the following equations:  
    (1) $x^2-1000.001x+1=0$;  
    (2) $x^2-10000.0001x+1=0$;  
    (3) $x^2-100000.00001x+1=0$;  
    (4) $x^2-1000000.000001x+1=0$.  

    (a) Using the appropriate method to find the roots of the equations.  
    (b) Determine the absolute and relative errors for your results.   

2. Several mathematical constants are used very frequently in science, such as $\pi$,  $e$, and the Euler constant $\gamma= \displaystyle\lim_{n\rightarrow\infty}\left(\displaystyle\sum_{k=1}^n k^{-1}-\ln n\right)$.   
  Find **three** ways of creating each of $\pi$, $e$, and $\gamma$ in a code. After considering language specifications, numerical accuracy, and efficiency, which way of creating each of them is most appropriate? If we need to use such a constant many times in a program, should the constant be created once and stored under a variable to be used over and over again, or should it be created/accessed every time it is needed?

