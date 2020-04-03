import numpy as np 
import matplotlib.pyplot as plt
from sympy import symbols, diff, exp

# a =================================================
x1 = np.linspace(-2, 1, 201)
y1 = -2 * np.exp(x1)

theta = np.linspace(0, 2*np.pi, 200)
x2 = np.cos(theta) * np.sqrt(8/3)
y2 = np.sin(theta) * np.sqrt(2)

plt.figure()
plt.plot(x1, y1, label='curve 1')
plt.plot(x2, y2, label='curve 2')
plt.title("Two roots are intersections on the figure")
plt.show()

# b ==================================================
def NewtonMethod(F, J, x, precision=1e-4, max_rd=100):
    F_value = F(x)
    F_norm = np.linalg.norm(F_value, ord=2) 
    rd = 0
    while abs(F_norm) > precision:
        delta = np.linalg.solve(J(x), F_value) 
        x -= delta
        F_value = F(x)
        F_norm = np.linalg.norm(F_value, ord=2)
        rd += 1
        if rd > max_rd:
            print(f"Method failed in {rd:d} steps.")
            exit(-1)
        
    return x

sym_x, sym_y = symbols("sym_x sym_y")
F0 = 2 * exp(sym_x) + sym_y
F1 = 3 * sym_x**2 + 4 * sym_y**2 - 8

print(diff(F0, sym_x))
print(diff(F0, sym_y))
print(diff(F1, sym_x))
print(diff(F1, sym_y))
