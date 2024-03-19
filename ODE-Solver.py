# ODE-Solver.py

import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# initialising constants
a = -5.0
b = 5.0
N = 8
h = (b-a)/N
C = 2

def f(x,y):
    f = -C*y-C*x**3-3*x**2
    return f

# exact solution to ODE

def y(x):
    y = 10*(math.e)**((-C*x)) - x**3
    return y

# defining functions for the numerical methods

def y_new(N):
    y_values = [y(a)]
    x_values = np.linspace(a, b, N+1)
    for i in range(1,N+1):
        y_values.append(y(x_values[i]))
    return y_values[-1]

def euler(N):
    y_euler_values = [y(a)]
    x_values = np.linspace(a, b, N+1)
    for i in range(1,N+1):
        y_euler_values.append(y_euler_values[i-1] + f(x_values[i-1],y_euler_values[i-1])*(b-a)/N)
    return y_euler_values[-1]

def RK(N):
    y_RK_values = [y(a)]
    x_values = np.linspace(a, b, N+1)
    for i in range(1,N+1):
        k_1 = f(x_values[i-1], y_RK_values[i-1])
        k_2 = f(x_values[i-1] + h/2, y_RK_values[i-1] + h*k_1/2)
        k_3 = f(x_values[i-1] + h/2, y_RK_values[i-1] + h*k_2/2)
        k_4 = f(x_values[i-1] + h, y_RK_values[i-1] + h*k_3)
        y_RK_values.append(y_RK_values[i-1] + ((b-a)/N)*(k_1 + 2*k_2 + 2*k_3 + k_4)/6)
    return y_RK_values[-1]

# observing error variation as a function of bin count N

data_N = [8, 16, 32, 64, 128, 256, 512, 1024]
ln_data_N = [math.log(_) for _ in data_N]
y_values_N = [y_new(_) for _ in data_N]

y_euler_values_N = [euler(_) for _ in data_N]
error_euler_N = [abs(i - j) for i,j in zip(y_euler_values_N, y_values_N)]
ln_error_euler_N = [math.log(_) for _ in error_euler_N]

y_RK_values_N = [RK(_) for _ in data_N]
error_RK_N = [abs(i - j) for i,j in zip(y_RK_values_N, y_values_N)]
ln_error_RK_N = [math.log(_) for _ in error_RK_N]

table1 = []

for i in range(0,8):
    table1 += [[data_N[i], y_values_N[i], y_euler_values_N[i], error_euler_N[i], y_RK_values_N[i], error_RK_N[i]]]
col_names1 = ['N', 'y(b)', 'y_N Euler', 'error Euler', 'y_N RK', 'error RK']

print(tabulate(table1, headers = col_names1))

# plotting linear regression fits

coeff_euler = np.polyfit(ln_data_N, ln_error_euler_N, 1)
fit_euler = np.poly1d(coeff_euler)

coeff_RK = np.polyfit(ln_data_N, ln_error_RK_N, 1)
fit_RK = np.poly1d(coeff_RK)

plt.scatter(ln_data_N, ln_error_euler_N)
plt.plot(ln_data_N, fit_euler(ln_data_N), label=f'Euler method: {coeff_euler[0]:.2f}x+{coeff_euler[1]:.2f}')

plt.scatter(ln_data_N, ln_error_RK_N)
plt.plot(ln_data_N, fit_RK(ln_data_N), label=f'RK method: {coeff_RK[0]:.2f}x+{coeff_RK[1]:.2f}')

plt.title('Error variation for different step counts (N) for numerical ODE solutions')
plt.xlabel('ln(N)')
plt.ylabel('ln|y_N - y(x_N)|')
leg = plt.legend(loc = 'lower left')
plt.show()