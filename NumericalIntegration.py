# NumericalIntegration.py

import math
import matplotlib.pyplot as plt
import numpy as np

# define limits of integration and bin count
a = 2.0
b = 5.0
k = 3.5 # some constant

# defining an example function
def f(x):
    return (x**3+(3+k)*x**2)*math.log(x)

# define the analytic solution to the integration problem
int_f=(b**4/4+(3+k)/3*b**3)*math.log(b)-b**4/16-(3+k)/9*b**3-((a**4/4+(3+k)/3*a**3)*math.log(a)-a**4/16-(3+k)/9*a**3)

# now getting to the numerical methods
# defining numerical methods as functions of bin count N

def constant_rule_lower(N):
    constant_rule_lower = 0
    for i in range(0,N):
        constant_rule_lower += f(a+i*(b-a)/N)*(b-a)/N
    return constant_rule_lower

def constant_rule_upper(N):
    constant_rule_upper = 0
    for i in range(1,N+1):
        constant_rule_upper += f(a+i*(b-a)/N)*(b-a)/N
    return constant_rule_upper

def trapezium_rule(N):
    trapezium_rule = 0.5*((b-a)/N)*(f(a)+f(b))
    for i in range(1,N):
        trapezium_rule += (f(a+i*(b-a)/N))*(b-a)/N
    return trapezium_rule

def midpoint_rule(N):
    midpoint_rule = 0
    for i in range(0,N):
        midpoint_rule += f(a+(i+0.5)*(b-a)/N)*(b-a)/N
    return midpoint_rule

def simpson_rule(N):
    simpson_rule = (f(a)+f(b))*(b-a)/(3*N)
    for i in range(1,N):
        if i % 2 == 0:
            pre_factor = 2
        else:
            pre_factor = 4
        simpson_rule += pre_factor*f(a+i*(b-a)/N)*(b-a)/(3*N)
    return simpson_rule

# observing variation in error for increasing bin count

data_N = [16, 32, 64, 128, 256]
ln_data_N = [math.log(ele) for ele in data_N]

constant_rule_lower_N = [constant_rule_lower(ele) for ele in data_N]
error_constant_rule_lower_N = [abs(ele - int_f) for ele in constant_rule_lower_N]
ln_error_constant_rule_lower_N = [math.log(ele) for ele in error_constant_rule_lower_N]

constant_rule_upper_N = [constant_rule_upper(ele) for ele in data_N]
error_constant_rule_upper_N = [abs(ele - int_f) for ele in constant_rule_upper_N]
ln_error_constant_rule_upper_N = [math.log(ele) for ele in error_constant_rule_upper_N]

trapzium_rule_N = [trapezium_rule(ele) for ele in data_N]
error_trapezium_rule_N = [abs(ele - int_f) for ele in trapzium_rule_N]
ln_error_trapezium_rule_N = [math.log(ele) for ele in error_trapezium_rule_N]

midpoint_rule_N = [midpoint_rule(ele) for ele in data_N]
error_midpoint_rule_N = [abs(ele - int_f) for ele in midpoint_rule_N]
ln_error_midpoint_rule_N = [math.log(ele) for ele in error_midpoint_rule_N]

simpson_rule_N = [simpson_rule(ele) for ele in data_N]
error_simpson_rule_N = [abs(ele - int_f) for ele in simpson_rule_N]
ln_error_simpson_rule_N = [math.log(ele) for ele in error_simpson_rule_N]

# calculating linear regression fits to error plots

coef_constant_rule_lower = np.polyfit(ln_data_N, ln_error_constant_rule_lower_N, 1)
fit_constant_rule_lower = np.poly1d(coef_constant_rule_lower)

coef_constant_rule_upper = np.polyfit(ln_data_N, ln_error_constant_rule_upper_N, 1)
fit_constant_rule_upper = np.poly1d(coef_constant_rule_upper)

coef_midpoint_rule = np.polyfit(ln_data_N, ln_error_midpoint_rule_N, 1)
fit_midpoint_rule = np.poly1d(coef_midpoint_rule)

coef_trapezium_rule = np.polyfit(ln_data_N, ln_error_trapezium_rule_N, 1)
fit_trapezium_rule = np.poly1d(coef_trapezium_rule)

coef_simpson_rule = np.polyfit(ln_data_N, ln_error_simpson_rule_N, 1)
fit_simpson_rule = np.poly1d(coef_simpson_rule)

# graphing the findings

plt.scatter(ln_data_N, ln_error_constant_rule_lower_N)
plt.plot(ln_data_N,fit_constant_rule_lower(ln_data_N),label=f'Constant rule lower: {coef_constant_rule_lower[0]:.2f}x+{coef_constant_rule_lower[1]:.2f}')

plt.scatter(ln_data_N, ln_error_constant_rule_upper_N)
plt.plot(ln_data_N,fit_constant_rule_upper(ln_data_N),label=f'Constant rule upper: {coef_constant_rule_upper[0]:.2f}x+{coef_constant_rule_upper[1]:.2f}')

plt.scatter(ln_data_N, ln_error_midpoint_rule_N)
plt.plot(ln_data_N,fit_midpoint_rule(ln_data_N),label=f'Midpoint rule: {coef_midpoint_rule[0]:.2f}x+{coef_midpoint_rule[1]:.2f}')

plt.scatter(ln_data_N, ln_error_trapezium_rule_N)
plt.plot(ln_data_N,fit_trapezium_rule(ln_data_N),label=f'Trapezium rule: {coef_trapezium_rule[0]:.2f}x+{coef_trapezium_rule[1]:.2f}')

plt.scatter(ln_data_N, ln_error_simpson_rule_N)
plt.plot(ln_data_N,fit_simpson_rule(ln_data_N),label=f'Simpson rule: {coef_simpson_rule[0]:.2f}x+{coef_simpson_rule[1]:.2f}')

# Plotting
plt.title('ln|error| vs ln(N) for different numerical integrators')
plt.xlabel('ln(N)')
plt.ylabel('ln|error|')
leg = plt.legend(loc='lower left')
plt.show()