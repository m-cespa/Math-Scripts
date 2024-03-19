# NewtonRaphson.py
#
# A script which uses newton raphson to solve equations analytically

import matplotlib.pyplot as plt
import numpy as np

def f(x):
    y = 65 - x**2
    return y

def deriv(p):
    h = 0.00000001
    y = (f((p + h))-f(p))/h
    return y

results = []

x1 = 2

for i in range(50):
    results.append(x1)
    x1 = x1 - (f(x1)/deriv(x1))
print(f"\nThe root of the given equation is {x1}")

x_s = np.linspace(-20, 20, 5000)
fig = plt.figure()
plt.plot(x_s, f(x_s), "r-", label="f(x)")
plt.plot(results, list(map(f,results)), "b--", label="iterations")
plt.grid(True)
plt.legend()
plt.show()
