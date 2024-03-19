# RootFinder.py

import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# defining range a<x<b in which roots are contained, discretising x
a = -3.0
b = 3.0
N = 64
k = 6.5 # some constant
delta_x = (b-a)/N

def f(x):
    f = math.exp(x) - (2*k*x**2)/(2+x**2)
    return f

# BISECTION METHOD

A_b = [-3.0]
B_b = [3.0]
C_b = [(A_b[0] + B_b[0])/2]

for i in range(21):
    if f(C_b[i]) * f(A_b[i]) <= 0:
        A_b.append(A_b[i])
        B_b.append(C_b[i])
    elif f(C_b[i]) * f(B_b[i]) <= 0:
        A_b.append(C_b[i])
        B_b.append(B_b[i])
    else:
        break
    C_b.append((A_b[i+1] + B_b[i+1])/2)

table = []

for i in range(21):
    table += [[i, A_b[i], C_b[i], B_b[i], f(A_b[i]), f(C_b[i]), f(B_b[i])]]
    
col_names = ['i', 'A_i', 'C_i', 'B_i', 'f(A_i)', 'f(C_i)', 'f(B_i)']

print(tabulate(table, headers = col_names))

# observing error variation with iteration count

errorBM = []

for i in range(20):
    errorBM.append(abs(C_b[i]-C_b[20]))
    
ln_erorrBM = [math.log(_) for _ in errorBM]

i_s = np.arange(0, 20)

plt.plot(i_s, ln_erorrBM)
plt.grid(True)
plt.xlabel('i_th iteration')
plt.ylabel('ln|C_i - C_20|')
plt.title('Variation of ln|error| with iteration count')

# NEWTON RAPHSON

def f_der(x):
    f = math.exp(x) - (8*k*x)/(2+x**2)**2
    return f

A_r = [-3.0]
B_r = [3.0]

for i in range(21):
    A_r.append(A_r[i] - f(A_r[i])/f_der(A_r[i]))

for i in range(21):
    B_r.append(B_r[i] - f(B_r[i])/f_der(B_r[i]))
    
errorNR_A = [abs(ele - A_r[20]) for ele in A_r]
errorNR_B = [abs(ele - B_r[20]) for ele in B_r]

table1 = []

for i in range(21):
    table1 += [[i, A_r[i], errorNR_A[i], f(A_r[i]), B_r[i], errorNR_B[i], f(B_r[i])]]
    
col_names1 = ['i', 'A_i', '|A_i - A_20|', 'f(A_i)', 'B_i', '|B_i - B_20|', 'f(B_i)']

print(tabulate(table1, headers = col_names1))

ln_errorNR_A = []
for i in range(21):
    if errorNR_A[i] > 0:
        ln_errorNR_A.append(math.log(errorNR_A[i]))
    else:
        ln_errorNR_A.append(float('nan'))
    
ln_errorNR_B = []
for i in range(21):
    if errorNR_B[i] > 0:
        ln_errorNR_B.append(math.log(errorNR_B[i]))
    else:
        ln_errorNR_B.append(float('nan'))
    
# LINEAR INTERPOLATION

A_lp = [-3.0]
B_lp = [3.0]
C_lp = [A_lp[0] - ((B_lp[0] - A_lp[0])*f(A_lp[0])) / (f(B_lp[0]) - f(A_lp[0]))]

for i in range(21):
    if f(C_lp[i])*f(A_lp[i]) <= 0:
        A_lp.append(A_lp[i])
        B_lp.append(C_lp[i])
    elif f(C_lp[i])*f(B_lp[i]) <= 0:
        A_lp.append(C_lp[i])
        B_lp.append(B_lp[i])
    else:
        break
    C_lp.append(A_lp[i+1] - ((B_lp[i+1] - A_lp[i+1])*f(A_lp[i+1])) / (f(B_lp[i+1]) - f(A_lp[i+1])))
    
errorLI = [abs(ele - C_lp[20]) for ele in C_lp]    

table2 = []

for i in range(21):
    table2 += [[i, A_lp[i], C_lp[i], B_lp[i], f(A_lp[i]), f(C_lp[i]), f(B_lp[i])]]

col_names2 = ['i', 'A_i', 'C_i', 'B_i', 'f(A_i)', 'f(C_i)', 'f(B_i)'] 

print(tabulate(table2, headers = col_names2))  

ln_errorLI = []
for i in range(21):
    if errorLI[i] > 0:
        ln_errorLI.append(math.log(errorLI[i]))
    else:
        ln_errorLI.append(float('nan'))
    
# SECANT METHOD

A_s = [-3.0, 3.0]

for i in range(1, 21):
    if abs(f(A_s[i]) - f(A_s[i-1])) > math.pow(10, -30):
        A_s.append(A_s[i] - ((A_s[i] - A_s[i-1])*f(A_s[i])) / (f(A_s[i]) - f(A_s[i-1])))
    else:
        A_s.append(A_s[i])
    
errorSM = [abs(ele - A_s[20]) for ele in A_s]

table3 = []

for i in range(21):
    table3 += [[i, A_s[i], errorSM[i], f(A_s[i])]]
    
col_names3 = ['i', 'A_i', '|A_i - A_20|', 'f(A_i)']

print(tabulate(table3, headers = col_names3))

ln_errorSM = []
for i in range(21):
    if errorSM[i] > 0:
        ln_errorSM.append(math.log(errorSM[i]))
    else:
        ln_errorSM.append(float('nan'))
    
# Plotting the error variation for each method

def plotter(x, colour, name):
    plt.scatter(range(len(x)), x, c=colour, label=name)
        
plotter(ln_erorrBM, 'red', 'Bisection Method')
plotter(ln_errorNR_A, 'green', 'Newton Raphson A')
plotter(ln_errorNR_B, 'blue', 'Newton Raphson B')
plotter(ln_errorLI, 'orange', 'Linear Interpolation')
plotter(ln_errorSM, 'black', 'Secant Method')

plt.title('Convergence rate for different root finders')
plt.xlabel('i_th iteration')
plt.ylabel('ln|error|')
plt.ylim(-37,5)
leg = plt.legend(loc = 'upper right')
plt.show()