# reservoir_sampling.py
# 
# implementation of reservoir sampling algorithm

import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

k = 20
no_elements = 40
iterations = 100000

# setup axes for live plotting
fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo', markersize=5)

# create average probability marker line
avg_line = ax.axhline(0, color='red', linestyle='--')
avg_text = ax.text(0.2, 0.5, '', color='red')

ax.grid(True)
ax.set_xlim(0, no_elements+1)
ax.set_ylim(0, 1)
ax.set_xlabel('Bin Count')
ax.set_ylabel('Probability of success')
ax.set_title('Real-Time Reservoir Sampling')

successes = Counter()

def get_successes(k, n):
    my_map = {}
    for i in range(1, n + 1):
        # for the first k elements, add to map
        if i <= k:
            my_map[i] = i
        # for elements > k, if random integer <= k -> replace
        else:
            j = random.randint(1, i)
            if j <= k:
                my_map[j] = i

    # all elements have equal probability k/n+1 of being in final map
    return my_map

def update(frame):
    global successes

    sample = get_successes(k, no_elements)
    successes.update(sample.values())

    rft = {key: val / (frame+1) for key, val in successes.items()}

    avg_prob = sum(rft.values()) / no_elements
    avg_line.set_ydata([avg_prob, avg_prob])
    avg_text.set_position((0.2, avg_prob + 0.02))
    avg_text.set_text(f'Avg: {round(avg_prob, 3)}')

    line.set_data(list(rft.keys()), list(rft.values()))
    return [line, avg_line, avg_text]

ani = FuncAnimation(fig, update, frames=iterations, blit=False, repeat=False, interval=50)
plt.show()

