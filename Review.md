
# 1. Review of Computational Physics
## 1.1. Catalog

<!-- TOC -->

- [1. Review of Computational Physics](#1-review-of-computational-physics)
  - [1.1. Catalog](#11-catalog)
- [2. Errors and Uncertainties](#2-errors-and-uncertainties)
  - [2.1. Storage](#21-storage)
  - [2.2. Error](#22-error)
- [3. Roots](#3-roots)
  - [3.1. Bisection Method](#31-bisection-method)
  - [3.2. Inverse Linear Interpolation](#32-inverse-linear-interpolation)
  - [3.3. Secant Method](#33-secant-method)
  - [3.4. Brute Force Method](#34-brute-force-method)
  - [3.5. Fixed Point Method](#35-fixed-point-method)
  - [3.6. Newton Raphson Method](#36-newton-raphson-method)
- [4. Matrix](#4-matrix)
  - [4.1. (non-linear) Newton-Raphson Algorithm](#41-non-linear-newton-raphson-algorithm)
  - [4.2. (linear) Gaussian Elimination Method with pivoting](#42-linear-gaussian-elimination-method-with-pivoting)
  - [4.3. (linear tridiagonal) Thomas Method](#43-linear-tridiagonal-thomas-method)
  - [4.4. (linear) Jacobi method](#44-linear-jacobi-method)
  - [4.5. Gauss-Seidel method](#45-gauss-seidel-method)
  - [4.6. Conjugate Gradient Descent Method](#46-conjugate-gradient-descent-method)
  - [4.7. Power method](#47-power-method)
  - [4.8. Jacobi Method(to find eigenvalue and eigenvector)](#48-jacobi-methodto-find-eigenvalue-and-eigenvector)
  - [4.9. SOR](#49-sor)
  - [4.10. Inverse of L and U](#410-inverse-of-l-and-u)
  - [4.11. Cholesky Decomposition](#411-cholesky-decomposition)
  - [4.12. QR Decomposition](#412-qr-decomposition)
- [5. Fitting](#5-fitting)
  - [5.1. Least-Square Method](#51-least-square-method)
  - [5.2. Fitting With Uncertainty](#52-fitting-with-uncertainty)
- [6. Interpolation](#6-interpolation)
  - [6.1. Linear Interpolation](#61-linear-interpolation)
  - [6.2. Polynomial Interpolation](#62-polynomial-interpolation)
  - [6.3. Lagrange Interpolation](#63-lagrange-interpolation)
  - [6.4. Newton Interpolation](#64-newton-interpolation)
  - [6.5. Spline Interpolation](#65-spline-interpolation)
- [7. Differentiation](#7-differentiation)
- [8. Integration](#8-integration)
  - [8.1. Trapezoid Rule](#81-trapezoid-rule)
  - [8.2. Midpoint Rule](#82-midpoint-rule)
  - [8.3. Simpson's Rule](#83-simpsons-rule)
  - [8.4. Romberg Integration](#84-romberg-integration)
  - [8.5. Gaussian Quadrature](#85-gaussian-quadrature)
  - [8.6. Improper Integration](#86-improper-integration)

<!-- /TOC -->
# 2. Errors and Uncertainties

## 2.1. Storage
1. Fixed point notation:  
   Fixed decimal point;  
   Same absolute error.  
2. Floating point notation:  
   Single: Sign(1) + Exponent(8) + Fraction(23) + Bias(127)  
   Double: Sign(1) + Exponent(11) + Fraction(52) + Bias(1023)  
   **First mantissa bit is always 1**  

## 2.2. Error
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


# 3. Roots

## 3.1. Bisection Method
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

## 3.2. Inverse Linear Interpolation
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

## 3.3. Secant Method
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
## 3.4. Brute Force Method
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
## 3.5. Fixed Point Method
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

## 3.6. Newton Raphson Method
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

# 4. Matrix

## 4.1. (non-linear) Newton-Raphson Algorithm
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

## 4.2. (linear) Gaussian Elimination Method with pivoting
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

## 4.3. (linear tridiagonal) Thomas Method
**a specialized Gaussian elimination method**
1. Method: Elimination becomes very easy when the matrix is tridiagonal. Just apply eliminate for one time when reducing a row. 
2. Pseudocode and Example:
```python
import numpy as np
def Thomas(M_init, b_init):
    # reduce to an upper triangular matrix
    
    M = M_init.copy().astype(np.float64)
    b = b_init.copy().astype(np.float64)
    
    n = M.shape[0]
    
    for i in range(1,n):
        r = M[i,i-1] / M[i-1,i-1]
        M[i,i-1] = 0
        M[i,i] = M[i,i] - r * M[i-1,i]
        if i != n-1:
            M[i,i+1] = M[i,i+1] - r * M[i-1,i+1]
        b[i] = b[i] - r * b[i-1]
    # get solution
    solution = np.zeros(n)
    solution[-1] = b[-1] / M[n-1,n-1]
    for i in range(n-2,-1,-1):
        solution[i] = ( b[i] - M[i,i+1]* solution[i+1] ) / M[i,i]

    return solution
```

## 4.4. (linear) Jacobi method
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

## 4.5. Gauss-Seidel method
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

## 4.6. Conjugate Gradient Descent Method
**Method:**   
    Compared to Gradient Descent Method, the descent vector is conjugate with each other. Here conjugate means $u^T A v=v^T A u = 0$, where A is a symmetrical matrix.  
    Problem becomes 
    $$    f=\frac{1}{2}x^T A x -b^T x $$$$
    \Delta f = Ax -b =0    $$
    Residue: $r_k=b-Ax_k=-\Delta f(x_k)$  
    **Iteration:**  
    $x_{n+1}=x_{n} + \alpha_n p_n, \alpha_n = \frac{r_n^T r_n}{p^T_n A p_n}$  
    $r_{n+1}=r_n - a_n A p_n$  
    $p_{n+1}=r_{n+1}+\beta_n p_n, \beta_n = \frac{r_{n+1}^T r_{n+1}}{r_n^T r_n}$

```python
#非老师给的代码，但加强了一波，适用于厄米矩阵(原先只适用于对称（实）矩阵)
def conjugate_gradient(A, b, x=None):
    n = len(b)
    if not x:
        x = np.ones(n)

    x = x
    r = b - dot(A,x)
    p = r
    
    for i in range(2*n):
        rAr = np.dot(np.conj(r),np.dot(A,r))
        Ap = np.dot(A,p)
        alpha = rAr / np.dot( np.conj(Ap) , Ap )
        
        x = x + alpha * p
        r = r - alpha * Ap

        rAr_plus_one = np.dot( np.conj(r) ,np.dot(A,r))
        beta = rAr_plus_one / rAr

        
        if abs(np.linalg.norm(r)) < 1e-5:
            print(alpha)
            print(p)
            break

        p = r + beta * p

    return x
```

## 4.7. Power method
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

## 4.8. Jacobi Method(to find eigenvalue and eigenvector)
* Main idea: eliminate the largest element with rotation matrix $$R = \left[\begin{matrix}\cos\phi&\sin\phi \\-\sin\phi &\cos\phi \end{matrix}\right]$$over and over again nitil convergence (if possible).
* $\cdots R_2R_1AR_1^{-1}R_{2}^{-1}\cdots = RAR^{-1}=\left[\begin{matrix}\lambda_1\\&\ddots\\&&\lambda_N \end{matrix}\right]$
* eigenvalue: $\lambda_1,\lambda_2,\cdots,\lambda_N$
* eigenvector: $AR^{-1}=R^{-1}\lambda$, the eigenvectors are columns of $R^{-1}$


## 4.9. SOR
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

## 4.10. Inverse of L and U
$L D = I$,$$
L = \left[\begin{matrix}1\\l_{21}&1\\l_{31}&l_{32}&1\\l_{41}&l_{42}&l_{43}&1\end{matrix}\right]
$$$$d_{jk}=-l_{jk}-\sum_{m=1}^{j-1}l_{jm}d_{mk}\qquad(j>k)$$
$U E=I$,
$$U = \left[\begin{matrix}u_{11}&u_{12}&u_{13}&u_{14}\\&u_{22}&u_{23}&u_{24}\\&&u_{33}&u_{34}\\&&&u_{44}\end{matrix}\right]$$$$e_{kk} = \frac{1}{u_{kk}},e_{jk}=-\frac{1}{u_{kk}}\sum_{m=j+1}^{k-1}e_{jm}u_{mk}(j<k)$$

## 4.11. Cholesky Decomposition
$A = LL^T$
$$
\begin{aligned}
&\begin{pmatrix}
a_{11} & a_{12} & a_{13}\\
a_{12} & a_{22} & a_{23}\\
a_{13} & a_{32} & a_{33}%
\end{pmatrix}  = \begin{pmatrix}
l_{11} & 0 & 0\\
l_{21} & l_{22} & 0\\
l_{31} & l_{32} & l_{33}%
\end{pmatrix}\begin{pmatrix}
l_{11} & l_{21} & l_{31}\\
0 & l_{22} & l_{32}\\
0 & 0 & l_{33}%
\end{pmatrix} \\
&  =\begin{pmatrix}
l_{11}^{2} & l_{11}l_{21} & l_{11}l_{31}\\
l_{21}l_{11} & l_{21}^{2}+l_{22}^{2} & l_{21}l_{31}+l_{22}l_{32}\\
l_{31}l_{11} & l_{21}l_{31}+l_{22}l_{32} & l_{31}^{2}+l_{32}^{2}+l_{33}^{2}%
\end{pmatrix}\\
&l_{jj}=\sqrt{a_{jj}-\displaystyle\sum_{k=1}^{k<j} l_{jk}^2},\quad 
l_{jk}=\frac{a_{jk}-\displaystyle\sum_{m=1}^{m<k}l_{jm}l_{km}}{l_{jj}}, \;( j>k)
\end{aligned}
$$

## 4.12. QR Decomposition

for $m>n$ 
$$
A=\underbrace{Q_{m\times m}}_{\text{orthogonal}}\underbrace{R_{m\times n}}_{\text{upper trigular}}%
$$$$
\underbrace{R}_{\text{upper trigular}}=Q_{n-1}\cdots Q_{2}Q_{1}A$$$$
A  =\left(  Q_{n-1}\cdots Q_{2}Q_{1}\right)  ^{-1}R =\underbrace{\left(  Q_{1}^{T}Q_{2}^{T}\cdots Q_{n}^{T}\right)  }_{Q}R
$$
**Gram-Schmidt Process**

**Householder Reflection**

# 5. Fitting

## 5.1. Least-Square Method

## 5.2. Fitting With Uncertainty
Chi-square

# 6. Interpolation

## 6.1. Linear Interpolation

## 6.2. Polynomial Interpolation
Vandermonde matrix

## 6.3. Lagrange Interpolation

## 6.4. Newton Interpolation

## 6.5. Spline Interpolation

# 7. Differentiation

# 8. Integration

## 8.1. Trapezoid Rule

## 8.2. Midpoint Rule

## 8.3. Simpson's Rule

## 8.4. Romberg Integration

## 8.5. Gaussian Quadrature

## 8.6. Improper Integration