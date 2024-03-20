# PDE-Solver.py

import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate

C = 4.0

# define exact solution to Laplace's equation in 2D

def Phi_exact(x, y):
    f = C*(x**2 - y**2) + x*y
    return f

# generating N+2 x N+2 sized grid of Phi values with boundary values fixed

def Phi(N):
    x_s = y_s = np.linspace(0, 1, N+2)
    array = np.zeros((N+2, N+2))

    array[:,0] = C*(x_s)**2
    array[:,-1] = C*(x_s**2 - 1) + x_s
    array[0,:] = -C*(y_s)**2
    array[-1,:] = C*(1 - y_s**2) + y_s
    return array

# Jacobi Method

def Jacobi_it(sigma, array):
    """performs a single iteration of Jacobi"""
    array[1:-1,1:-1] = (1-sigma)*array[1:-1,1:-1] + (sigma/4)*(array[2:,1:-1] + array[:-2,1:-1] + array[1:-1,2:] + array[1:-1,:-2])
    return array

def Jacobi(sigma, N, k):
    """performs k iterations of Jacobi"""
    array_old = Phi(N)    
    for _ in range(k):
        array_new = np.copy(array_old)
        for i in range(1, N+1):
            for j in range(1, N+1):
                array_new[i,j] = (1-sigma)*array_old[i,j] + (sigma/4)*(array_old[i+1,j] + array_old[i-1,j] + array_old[i,j+1] + array_old[i,j-1])
        array_old = np.copy(array_new)
    return array_new

# _it functions perform error calc by doing singular iteration on inputted array
# ensure indices are made integer and not floats by using "//"

def e_it(x, y, N, sigma, array):
    return abs(Phi_exact(x,y) - Jacobi_it(sigma, array)[(N+1)//2][(N+1)//2])

def e(x, y, N, k, sigma):
    return abs(Phi_exact(x,y) - Jacobi(sigma, N, k)[(N+1)//2][(N+1)//2])

def e_rel_it(x, y, N, sigma, array):
    pe = Phi_exact(x,y)
    return abs((pe - Jacobi_it(sigma, array)[(N+1)//2][(N+1)//2])) / abs(pe)

def e_rel(x, y, N, k, sigma):
    return abs((Phi_exact(x,y) - Jacobi(sigma, N, k)[(N+1)//2][(N+1)//2])) / abs(Phi_exact(x,y))

k_s_0 = np.linspace(1, 200, 200)
y1_s = []
array = Phi(13)
for ele in range(1, 201):
    y1_s.append(e_it(0.5, 0.5, 13, 1, array))

# defining array again after previous mutation
array = Phi(13)
y2_s = []
for ele in range(1, 201):
    y2_s.append(e_rel_it(0.5, 0.5, 13, 1, array))

plt.plot(k_s_0, y1_s, label = "absolute error")
plt.plot(k_s_0, y2_s, label = "relative error")
plt.xlabel("kth iteration")
plt.ylabel("error")
leg = plt.legend(loc = 'upper right')
plt.show()

# estimate number of iterations k to obtain <1% error

def kN(sigma, N, max_iterations=1000, tolerance=0.01):
    array = Phi(N)
    for k in range(1, max_iterations+1):
        if e_rel_it(0.5, 0.5, N, sigma, array) <= tolerance:
            return k
# if iterations > 1000 we denote the fail by returning -1
    return -1

# graphing the minimum iteration count for <1% error against grid size N

data_N = (np.linspace(1, 13, 7)).astype(int)
log_data_N = np.log(data_N)

k_s_1 = []
for _ in data_N:
    k_s_1.append(kN(1, _))

log_kN = np.log(k_s_1)

plt.scatter(log_data_N, log_kN, label = "Jacobi Data")
slope, intercept = np.polyfit(log_data_N, log_kN, 1)
plt.plot(log_data_N, slope*log_data_N + intercept, color='red', label=f'Linear Fit: {np.round(slope,3)}x + {np.round(intercept,3)}')
plt.xlabel("log(N)")
plt.ylabel("log(k_N)")
plt.title("iteration number k to achieve < 1% error varying with grid size N")
leg = plt.legend(loc = 'upper left')
plt.show()

# tabulating k values for each relaxation σ (Jacobi)

sigma_s = np.arange(0.2, 2.2, 0.2)
k_s_2 = []
for ele in sigma_s:
    k_s_2.append(kN(ele, 13))

table1 = []

for i in range(10):
    table1.append([sigma_s[i], k_s_2[i]])

col_names1 = ['σ', 'k_N']

print(tabulate(table1, headers=col_names1))

# demonstrating divergence of Jacobi for σ>1.0
xphi_s = np.linspace(0, 1, 15)
phi_s = []
for i in range(0,15):
    phi_s.append(Jacobi(1.1, 13, 100)[7][i])

plt.scatter(xphi_s, phi_s)
plt.xlabel("x")
plt.ylabel("φ(x, 0.5)")
plt.show()

def GS(sigma, N, k):
    array = Phi(N)
    
    # looping the Gauss-Seidel iteration over k times
    for _ in range(k):
        for i in range(1, N+1):
            for j in range(1, N+1):
                array[i,j] = (1-sigma)*array[i,j] + (sigma/4)*(array[i+1,j] + array[i-1,j] + array[i,j+1] + array[i,j-1])
    return array

def GS_it(sigma, array):
    """performs a single iteration of Gauss-Seidel"""
    N = len(array) - 2
    for i in range(1, N+1):
        for j in range(1, N+1):
            array[i,j] = (1-sigma)*array[i,j] + (sigma/4)*(array[i+1,j] + array[i-1,j] + array[i,j+1] + array[i,j-1])
    return array
    
def e_rel_GS_it(x, y, N, sigma, array):
    pe = Phi_exact(x, y)
    return abs((pe - GS_it(sigma, array)[(N+1)//2][(N+1)//2])) / abs(pe)    
    
def e_rel_GS(x, y, N, k, sigma):
    return abs((Phi_exact(x,y) - GS(sigma, N, k)[(N+1)//2][(N+1)//2])) / abs(Phi_exact(x,y))

def kNGS(sigma, N, max_iterations=1000, tolerance=0.01):
    array = Phi(N)
    for k in range(1, max_iterations+1):
        if e_rel_GS_it(0.5, 0.5, N, sigma, array) <= tolerance:
            return k
# if iterations > 1000 we denote the fail by returning -1
    return -1

k_s_3 = []
for _ in data_N:
    k_s_3.append(kNGS(1, _))

log_kNGS = np.log(k_s_3)

plt.scatter(log_data_N, log_kNGS, label = "Gauss-Seidel Data")
slope, intercept = np.polyfit(log_data_N, log_kNGS, 1)
plt.plot(log_data_N, slope*log_data_N + intercept, color='red', label=f'Linear Fit: {np.round(slope,3)}x + {np.round(intercept,3)}')
plt.xlabel("log(N)")
plt.ylabel("log(k_NGS)")
plt.title("iteration number k to achieve < 1% error varying with grid size N")
leg = plt.legend(loc = 'upper left')
plt.show()

# tabulating k values for each relaxation σ (Gauss-Seidel)

k_s_4 = []
for ele in sigma_s:
    k_s_4.append(kNGS(ele, 13))

table2 = []

for i in range(10):
    table2.append([sigma_s[i], k_s_4[i]])

col_names2 = ['σ', 'k_NGS']

print(tabulate(table2, headers=col_names2))