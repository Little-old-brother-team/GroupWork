# Mid Term Review

## 1 Errors and Uncertainties

### Storage
1. Fixed point notation:  
   Fixed decimal point;  
   Same absolute error.  
2. Floating point notation:  
   Single: Sign(1) + Exponent(8) + Fraction(23) + Bias(127)  
   Double: Sign(1) + Exponent(11) + Fraction(52) + Bias(1023)  
   **First mantissa bit is always 1**  

### Error
1. Round off error  
    **From storage limit**.  
    Appear significantly in:
    1. subtractive cancellation (big - big = small) 
    2. large computation (many rounds) 
    3. adding a large and a small number  
    4. smearing (summation from large to small)
    5. inner products  

2. Truncation errors  
   **From mathematic approximation**


## 2 Roots

### Bisection Method
**Basic Method**
1. Method:   
    1. search root in $[x_1, x_2]$, and $f(x_1)f(x_2)<0$
    2. divide $[x_1, x_2]$ **equally**, determine where is the root
    3. repeat 1. and 2., until reach the target

2. Pseudocode and Example: 
```
Define F(x)
Define BisectionMethod(F, x1, x2, target):
    Repeat N times or until error < target :
        mid = (x1 + x2) / 2
        if (F(x1)F(mid) < 0): x2 = mid
        else: x1 = mid
    return mid
 
Find x1, x2 which make F(x1) * F(x2) < 0
Call BisectionMethod(F, x1, x2, eps)
```

### Inverse Linear Interpolation
**Improved Bisection Method**
Method: Change the middle point $\frac{x_1+x_2}{2}$ of Bisection Method to learn prediction $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$.

### Secant Method
**Can't find a initial bracket**
Method: Don't need to consider the signs of boundaries. And mid point is the same as Inverse Linear Interpolation, which is $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$. 

### Brute Force Method
1. Method:  
    1. Divide the interval into small segments
    2. Find the segment $[x_1, x_2]$ containing a root
    3. Final result: $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$
2. Pseudocode and Example: 
```
Define F(x)
Define BruteForceMethod(F, x1, x2, n):
    divide [x1, x2] into segments [x1, x2, x3 ... xn]
    while F(xi) * F(xj) > 0:
        ++ i
        ++ j

    return xj - F(xj) * (xj - xi) / (F(xj) - F(xi))

Find x1, x2 
Call BisectionMethod(F, x1, x2, n)
```

### Fixed Point Method
1. Method:  
    **Iteration:** $x_n = g(x_{n-1})$ for equation $g(x)-x=0$.

2. Pseudocode and Example:  
    find solution of $F(x) = x$:
```
Define F(x)
Define FixedPointMethod(F, x1, N, target):
    Repeat N times or until error < target:
        x = F(x)
    return x

Find x1
Call BisectionMethod(F, x1, N, eps)
```

### Newton Raphson Method
1. Method: 
    **Iteration:** $\Delta x = -\frac{f(x)}{f^\prime (x)}$
2. Pseudocode and Example: 
```
Define F(x), F_prime(x)
Define NewtonMethod(F, F_prime, x1, target):
    Repeat N times or until error < target:
        x -= F(x)/F_prime(x)
    return x

Find x1
Call BisectionMethod(F, F_prime, x1, eps)
```

## Matrix

### (non-linear) Newton-Raphson Algorithm
1. Method: **Just a high dimensional Newton method**  
    1. Calculate Jacobi matrix $F^\prime$ of equations $f(x)$.
    2. Solve $\Delta x$ by $F^\prime \Delta x = -f(x)$
2. Pseudocode and Example: 
```
Define F(x), F_prime(x)
Define NewtonMethod(F, F_prime, x1, target):
    Repeat N times or until error < target:
        solve F_prime(x) * delta_x = F(x)
        x -= delta_x
    return x

Find x1
Call BisectionMethod(F, F_prime, x1, eps)
```

### (linear) Gaussian Elimination Method with pivoting
1. Method: for $Ax=b$  
    1. Reduce matrix $A$ to upper triangular matrix with $b$. Before each elimination try to put a row with largest pivot on the eliminating row. 
    2. Back substitution.
2. Pseudocode and Example:
```
Define Elimination(M, b):
    # reduce to an upper triangular matrix
    for row i from 0 to n:
        exchange the row with largest pivot to row i
        eliminate column i
    
    # get solution
    for row i from n to 0:
        eliminate column i and get solution

    return solution
```

### 