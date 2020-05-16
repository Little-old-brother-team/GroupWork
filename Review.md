
# 1. Review of Computational Physics

<!-- TOC -->

- [1. Review of Computational Physics](#1-review-of-computational-physics)
  - [1.1. Errors and Uncertainties](#11-errors-and-uncertainties)
    - [1.1.1. Storage](#111-storage)
    - [1.1.2. Error](#112-error)
  - [1.2. Roots](#12-roots)
    - [1.2.1. Bisection Method](#121-bisection-method)
    - [1.2.2. Inverse Linear Interpolation](#122-inverse-linear-interpolation)
    - [1.2.3. Secant Method](#123-secant-method)
    - [1.2.4. Brute Force Method](#124-brute-force-method)
    - [1.2.5. Fixed Point Method](#125-fixed-point-method)
    - [1.2.6. Newton Raphson Method](#126-newton-raphson-method)
  - [1.3. Matrix](#13-matrix)
    - [1.3.1. (non-linear) Newton-Raphson Algorithm](#131-non-linear-newton-raphson-algorithm)
    - [1.3.2. (linear) Gaussian Elimination Method with pivoting](#132-linear-gaussian-elimination-method-with-pivoting)
    - [1.3.3. (linear tridiagonal) Thomas Method](#133-linear-tridiagonal-thomas-method)
    - [1.3.4. (linear) Jacobi method](#134-linear-jacobi-method)
    - [1.3.5. Gauss-Seidel method](#135-gauss-seidel-method)
    - [1.3.6. Conjugate Gradient Descent Method](#136-conjugate-gradient-descent-method)
    - [1.3.7. Power method](#137-power-method)
    - [1.3.8. SOR](#138-sor)
  - [1.4. Fitting](#14-fitting)
    - [1.4.1. Least-Square Method](#141-least-square-method)
    - [1.4.2. Fitting With Uncertainty](#142-fitting-with-uncertainty)
  - [1.5. Interpolation](#15-interpolation)
    - [1.5.1. Linear Interpolation](#151-linear-interpolation)
    - [1.5.2. Polynomial Interpolation](#152-polynomial-interpolation)
    - [1.5.3. Lagrange Interpolation](#153-lagrange-interpolation)
    - [1.5.4. Newton Interpolation](#154-newton-interpolation)
    - [1.5.5. Spline Interpolation](#155-spline-interpolation)
  - [1.6. Differentiation](#16-differentiation)
  - [1.7. Integration](#17-integration)
    - [1.7.1. Trapezoid Rule](#171-trapezoid-rule)
    - [1.7.2. Midpoint Rule](#172-midpoint-rule)
    - [1.7.3. Simpson's Rule](#173-simpsons-rule)
    - [1.7.4. Romberg Integration](#174-romberg-integration)
    - [1.7.5. Gaussian Quadrature](#175-gaussian-quadrature)
    - [1.7.6. Improper Integration](#176-improper-integration)

<!-- /TOC -->

## 1.1. Errors and Uncertainties

### 1.1.1. Storage
1. Fixed point notation:  
   Fixed decimal point;  
   Same absolute error.  
2. Floating point notation:  
   Single: Sign(1) + Exponent(8) + Fraction(23) + Bias(127)  
   Double: Sign(1) + Exponent(11) + Fraction(52) + Bias(1023)  
   **First mantissa bit is always 1**  

### 1.1.2. Error
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


## 1.2. Roots

### 1.2.1. Bisection Method
**Basic Method**
1. Method:   
    1. search root in $[x_1, x_2]$, and $f(x_1)f(x_2)<0$
    2. divide $[x_1, x_2]$ **equally**, determine where is the root
    3. repeat 1. and 2., until reach the target

2. Pseudocode and Example: 
    ```python
    def bisection(f,left,right,tolerance=1e-8,maxRuns=1000):
        its = 0
        while abs(right-left) > tolerance and its < maxRuns:
            mid = (left + right) / 2
            if f(mid) == 0:
                break
            else:
                if f(mid)*f(right) < 0:
                    left = mid
                else:
                    right = mid
            its = its + 1
            if its == maxRuns:
                print('Reach the max run.') 
        return mid
    ```

### 1.2.2. Inverse Linear Interpolation
**Improved Bisection Method**
Method: Change the middle point $\frac{x_1+x_2}{2}$ of Bisection Method to learn prediction $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$.
```python
def InverseLinear(f,left,right,tolerance=1e-8,maxRuns=1000):
    its = 0
    while abs(right-left) > tolerance and its < maxRuns:
        mid = right - f(right)*(right-left)/(f(right)-f(left))
        if f(mid) == 0:
            break
        else:
            if f(mid)*f(right) < 0:
                left = mid
            else:
                right = mid
        its += 1
        
        if its == maxRuns:
            print('Reach the max run.') 
    return mid
```

### 1.2.3. Secant Method
**Can't find a initial bracket**
Method: Don't need to consider the signs of boundaries. And mid point is the same as Inverse Linear Interpolation, which is $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$. 
```python
def secant(f, x0, x1, eps=1e-5, its=100):
    f_x0 = f(x0)
    f_x1 = f(x1)
    iteration_counter = 0
    while abs(f_x1) > eps and iteration_counter < its:
        try:
            denominator = (f_x1 - f_x0)/(x1 - x0)
            x = x1 - f_x1/denominator
        except ZeroDivisionError:
            print('Error! - denominator zero for x = ', x)
            sys.exit(1) # Abort with error
        x0 = x1
        x1 = x
        f_x0 = f_x1
        f_x1 = f(x1)
        iteration_counter = iteration_counter + 1
    # Here, either a solution is found, or too many iterations
    if abs(f_x1) > eps:
        iteration_counter = -1
    return x, iteration_counter
```
### 1.2.4. Brute Force Method
1. Method:  
    1. Divide the interval into small segments
    2. Find the segment $[x_1, x_2]$ containing a root
    3. Final result: $x_2-f(x_2)\frac{x_2-x_1}{f(x_2)-f(x_1)}$
2. Pseudocode and Example: 
    ```python
    def brute_force_root_finder(f, a, b, n):
        from numpy import linspace
        x = linspace(a, b, n)
        y = f(x)
        
        #plt.plot(x,y,'r-', label = 'zero of f(x)',lw=2, marker='s', ms=2)     
            # square size 10
        #plt.axhline(y=0)
        
        roots = []
        for i in range(n-1):
            if y[i]*y[i+1] < 0:
                root = x[i] - (x[i+1] - x[i])/(y[i+1] - y[i])*y[i]
                roots.append(root)
            elif y[i] == 0:
                root = x[i]
                roots.append(root)
        return roots
    ```
### 1.2.5. Fixed Point Method
1. Method:  
    **Iteration:** $x_n = g(x_{n-1})$ for equation $g(x)-x=0$.

2. Pseudocode and Example:  
    find solution of $F(x) = x$:
    ```python
    def FixedPointMethod(F, x1, N=100, eps=1e-5):
        its = 0
        last_x = x1-1
        x = x1
        while its < N and abs(last_x-x) > eps:
            x = F(x)
            its = its + 1
            if its == N:
                print('Reach the max run.') 
        return x
    ```

### 1.2.6. Newton Raphson Method
1. Method: 
    **Iteration:** $\Delta x = -\frac{f(x)}{f^\prime (x)}$
2. Pseudocode and Example: 
    ```python
    import sys

    def Newton(f, dfdx, x, eps):
        f_value = f(x)
        iteration_counter = 0
        while abs(f_value) > eps and iteration_counter < 100:
            try:
                x = x - f_value/dfdx(x)
            except ZeroDivisionError:
                print('Error! - derivative zero for x = ', x)
                sys.exit(1)     # Abort with error

            f_value = f(x)
            iteration_counter = iteration_counter + 1

        # Here, either a solution is found, or too many iterations
        if abs(f_value) > eps:
            iteration_counter = -1
        return x, iteration_counter
    ```

## 1.3. Matrix

### 1.3.1. (non-linear) Newton-Raphson Algorithm
1. Method: **Just a high dimensional Newton method**  
    1. Calculate Jacobi matrix $F^\prime$ of equations $f(x)$.
    2. Solve $\Delta x$ by $F^\prime \Delta x = -f(x)$
2. Pseudocode and Example: 
    ```python
    def Newton_system(F, J, x, eps):
        """
        Solve nonlinear system F=0 by Newton’s method.
        J is the Jacobian of F. Both F and J must be functions of x.
        At input, x holds the start value. The iteration continues
        until ||F|| < eps.
        """
        
        F_value = F(x)
        
        F_norm = np.linalg.norm(F_value, ord=2) # l2 norm of vector
        
        iteration_counter = 0
        
        while abs(F_norm) > eps and iteration_counter < 100:
            delta = np.linalg.solve(J(x), -F_value)    # J(x) delta = -F
            x = x + delta
            F_value = F(x)
            F_norm = np.linalg.norm(F_value, ord=2)
            iteration_counter = iteration_counter + 1
            
        # Here, either a solution is found, or too many iterations
        if abs(F_norm) > eps:
            iteration_counter = -1
        return x, iteration_counter

    from sympy import *
    x0, x1 = symbols('x0 x1')
    F0 = x0**2 - x1 + x0*cos(pi*x0)
    F1 = x0*x1 + exp(-x1) - x0**(-1)
    print(diff(F0, x0))
    print(diff(F0, x1))
    print(diff(F1, x0))
    print(diff(F1, x1))
    ```

### 1.3.2. (linear) Gaussian Elimination Method with pivoting
1. Method: for $Ax=b$  
    1. Reduce matrix $A$ to upper triangular matrix with $b$. Before each elimination try to put a row with largest pivot on the eliminating row. 
    2. Back substitution.
2. Pseudocode and Example:
    ```python
    Define Elimination(M, b):
        # reduce to an upper triangular matrix
        for row i from 0 to n:
            exchange the row with largest pivot to row i
            for row j from i to n:
                eliminate row j with row i
        
        # get solution
        for row i from n to 0:
            eliminate column i and get solution

        return solution
    ```
3. Numpy implementation:
   ```python
   np.linalg.solve(A, b)
   ```

### 1.3.3. (linear tridiagonal) Thomas Method
**a specialized Gaussian elimination method**
1. Method: Elimination becomes very easy when the matrix is tridiagonal. Just apply eliminate for one time when reducing a row. 
2. Pseudocode and Example:
```python
Define Thomas(M, b):
    # reduce to an upper triangular matrix
    for row i from 1 to n:
        eliminate row i with row i-1
    
    # get solution
    for row i from n to 0:
        eliminate column i and get solution

    return solution
```

### 1.3.4. (linear) Jacobi method
Indirect algorithm to solve linear equation by iteration.
* Attention: Only valid when A is diagonal dominant!
* Main idea: First convert A to L + U + D (Lower, Upper and Diag). Then calculate $J = -D^{-1}(L+R)$, find the eigenvalue of J and check whether the max absolute value is greater than 1. If so, the iteration will not convergence, you'd better find another algorithm to solve it! 
* Algorithm: $x_j^{(k+1)} = \frac{1}{a_{jj}}(b_j-\sum_{m\neq j}a_{jm}x_m^{(k)})$
```python
from numpy import diag, allcose, zeros_like
def jacobi(A, b, N=1000, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = zeros_like(b)

    # Create a vector of the diagonal elements of A   
    # and subtract them from A       
    D = diag(A)
    R = A - diagflat(D)
    # Iterate for N times      
    for i in range(N):
        x_new = (b - dot(R,x)) / D
        if allclose(x, x_new, rtol=1e-8): # compare
            break
        
        x = x_new
    return x
```

### 1.3.5. Gauss-Seidel method
* Main idea:  Gauss-Seidel method is a refinement of the Jacobi method, it converges faster than Jacobi method by introduce the updated value in the same iteration.
* Algorithm: $x_j^{(k+1)} = \frac{1}{a_{jj}}(b_j-\sum_{m>j}a_{jm}x_m^{(k)}-\sum_{m<j}a_{jm}x_m^{(k+1)})$
```python
from numpy import zeros_like, allclose, dot
def Gauss_Seidel(A, b, N=25, x=None, omega=1.5):
    """
    Solves the equation Ax=b via the Jacobi iterative method.
    """
    # Create an initial guess if needed  
    if x is None:
        x = zeros_like(b)

    # Iterate for N times  
    for it_count in range(N):
        x_new = zeros_like(x)
        print("Iteration {0}: {1}".format(it_count, x))
        for i in range(A.shape[0]):
            s1 = dot(A[i, :i], x_new[:i])
            s2 = dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if allclose(x, x_new, rtol=1e-8):
            break
        x = x_new
    return x
```

### 1.3.6. Conjugate Gradient Descent Method
**Method:**   
    Compared to Gradient Descent Method, the descent vector is conjugate with each other. Here conjugate means $u^T A v=v^T A u = 0$, where A is a symmetrical matrix.  
    Problem becomes 
    $$ 
    f=\frac{1}{2}x^T A x -b^T x \\
    \Delta f = Ax -b =0
    $$
    Residue: $r_k=b-Ax_k=-\Delta f(x_k)$  
    **Iteration:**  
    $x_{n+1}=x_{n} + \alpha_n p_n, \alpha_n = \frac{r_n^T r_n}{p^T_n A p_n}$  
    $r_{n+1}=r_n - a_n A p_n$  
    $p_{n+1}=r_{n+1}+\beta_n p_n, \beta_n = \frac{r_{n+1}^T r_{n+1}}{r_n^T r_n}$

### 1.3.7. Power method
* Main idea: Find the largest or smallest eigenvalue by mutiply $A$ or $A^{-1}$ for k times (k>>1). Since $\frac{\lambda_{i}}{\lambda_{max}} < 1$, then $(\frac{\lambda_{i}}{\lambda_{max}})^k \approx 0$
* Algorithm: $\lambda_{max} = \frac{x^T_kx_{k+1}}{x^T_kx_{k}}$, the same algorithm for $\lambda_{min}$.

```python
def power(A,x):
    def rayleigh_quotient(A,x):
        return np.dot(x, np.dot(A, x))/np.dot(x,x)
        
    # function to normalise a vector
    def normalize(x,eps=1e-10):
        N = np.sqrt(np.sum(abs(x)**2))
        if N < eps: # in case it is the zero vector!
            return x
        else:
            return x/N

    RQnew = rayleigh_quotient(A,x)
    RQold = 0

    # perform the power iteration
    while np.abs(RQnew-RQold) > 1e-6:
        RQold = RQnew
        x = normalise(np.dot(A, x))
        RQnew = rayleigh_quotient(A, x)
```
* Numpy
  ```python
  eig_value, eig_vector = np.linalg.eig(A)
  ```

### 1.3.8. SOR
* Main idea: SOR is also a refinement of the Gauss Seidel method, by introducing a parameter $\omega$ to balence the x value between the old and new iteration, SOR can achieve the highest speed among all algorithm introduced above.
* Algorithm: $x_i^{(k+1)}=(1-\omega)x_i^{(k)}+\frac{\omega}{a_{ii}}\left(b_i-\displaystyle\sum_{j<i}a_{ij}x_j^{(k+1)}-\displaystyle\sum_{j>i}a_{ij}x_j^{(k)}\right)$
```python
from numpy import zeros_like, allclose, dot
def SOR(A, b, N=25, x=None, omega=1.5):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = zeros_like(b)

    # Iterate for N times  
    for it_count in range(N):
        x_new = zeros_like(x)
        print("Iteration {0}: {1}".format(it_count, x))
        for i in range(A.shape[0]):
            s1 = dot(A[i, :i], x_new[:i])
            s2 = dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1-omega) * x[i] + omega * (b[i] - s1 - s2) / A[i, i]
        if allclose(x, x_new, rtol=1e-8):
            break
        x = x_new
    return x
```

## 1.4. Fitting

### 1.4.1. Least-Square Method

### 1.4.2. Fitting With Uncertainty
Chi-square

## 1.5. Interpolation

### 1.5.1. Linear Interpolation

### 1.5.2. Polynomial Interpolation
Vandermonde matrix

### 1.5.3. Lagrange Interpolation

### 1.5.4. Newton Interpolation

### 1.5.5. Spline Interpolation

## 1.6. Differentiation

## 1.7. Integration

### 1.7.1. Trapezoid Rule

### 1.7.2. Midpoint Rule

### 1.7.3. Simpson's Rule

### 1.7.4. Romberg Integration

### 1.7.5. Gaussian Quadrature

### 1.7.6. Improper Integration