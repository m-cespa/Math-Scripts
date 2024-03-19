# MonteCarlo to calculate pi

import math
import random
import matplotlib.pyplot as plt

N = 10000
print("Parameter: N =", str(N))
# generate array of iterations
iterations = []
# generate array of results
results = []

count_in = 0
for i in range(1,N+1):
    # generate random (x,y) coordinate
    x = random.random()
    y = random.random()
    # condition checking whether random coordinate lies outside circle r=0.5
    cond = x**2 + y**2
    outcome = 1 if cond <= 1 else 0
    # stores value if point is inside circle 
    count_in += outcome
    # fraction of points in circle = A(circle)/A(square) = pi/4
    fraction_in = count_in/i

    # store results into the array
    results.append(4.0 * fraction_in)
    # stores interations into the array
    iterations.append(i)

    # last printed number converges to pi
    print(f"Location: {outcome}\t{x}\t{y}\t{count_in}\t{i}\t{(4)*fraction_in}")
    
# plotting results
fig = plt.figure()
plt.plot(iterations, results, "k-", label="numerical pi")
plt.plot([0, iterations[-1]], [math.pi, math.pi], "r-", label="analytical pi")
plt.grid(True)
plt.legend()
plt.ylabel("Result [-]")
plt.xlabel("Iteration [-]")
plt.show()