import time
from Interaction import Interaction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = ['serif']
rcParams['font.serif'] = ["Times New Roman"]
plt.rc('font', size=14)  # default text size
plt.rc('axes', titlesize=16)  # title
plt.rc('axes', labelsize=14)  # x and y labels
plt.rc('xtick', labelsize=14)  # x tick labels
plt.rc('ytick', labelsize=14)  # y tick labels
plt.rc('legend', fontsize=14)  # legend

# Initial set up
G = 1
mass = [1, 0, 0]
particles_init = np.array([[[6, 0], [0, np.sqrt(G * mass[0] / 6)]]])

# Getting the figures ready for plotting
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Duration')
ax1.set_ylabel('Relative energy deviation per unit mass')
ax1.set_title('Energy deviation of the particle vs time.')
ax1.grid(linestyle='dashed')

fig2, ax2 = plt.subplots()
ax2.set_xlabel('Duration')
ax2.set_ylabel('Execution time / s')
ax2.set_title('Execution time vs time')
ax2.grid(linestyle='dashed')

# Lines and labels lists for the legend
lns1, lns2, labs = [], [], []

routines = ['LSODA', 'RK45', 'DOP853']

# Double for loop to run each routine for multiple durations and store the execution time and energy deviation
times = np.linspace(1, 95000, 10)
for routine in routines:
    routine_time = []
    routine_deviation = []
    for time_stamp in times:
        # Do for multiple different durations
        start_time = time.time()
        interact = Interaction(mass=mass,
                               particles=particles_init,
                               steps=95000,
                               duration=time_stamp,
                               num_particles=1,
                               rtol=1e-10,
                               atol=1e-10,
                               method=routine)
        interact.produce_orbits()
        deviation = interact.energy_deviation()

        # Adding execution time and energy deviation to array
        routine_time.append(time.time() - start_time)
        routine_deviation.append(deviation)

    line1, = ax1.plot(times, routine_deviation, "o-")
    line2, = ax2.plot(times, routine_time, "o-")
    lns1.append(line1)
    labs.append(str(routine))
    lns2.append(line2)

ax1.legend(labels=labs, handles=lns1, loc='center left', title='Routine')
ax2.legend(labels=labs, handles=lns2, loc='upper center', title='Routine')

plt.show()