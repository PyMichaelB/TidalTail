from Interaction import Interaction
import time
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

# Values of rtol and atol we want to experiment with
rtols = [1e-10, 1e-12]
atols = [5e-8, 2.5e-8, 1e-8, 5e-9, 2.5e-9, 1e-9, 5e-10, 2.5e-10, 1e-10, 5e-11, 2.5e-11, 1e-11, 5e-12, 2.5e-12, 1e-12, 5e-13, 2.5e-13, 1e-13]

# Getting figures ready for plotting
fig1, ax1 = plt.subplots()
ax1.set_xlabel('atol')
ax1.set_ylabel('Relative energy change')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Variation of the error in the energy with atol')
ax1.grid(linestyle='dashed')

fig2, ax2 = plt.subplots()
ax2.set_xlabel('atol')
ax2.set_ylabel('Execution time / s')
ax2.set_xscale('log')
ax2.set_title('Variation of execution time with atol')
ax2.grid(linestyle='dashed')

# Lines and labels lists for the legend
lns1, lns2, labs = [], [], []

for rtol in rtols:
    # Errors for this particular value of rtol
    rtol_errors = []
    rtol_times = []

    for atol in atols:
        start_time = time.time()
        interact = Interaction(mass=mass,
                               particles=particles_init,
                               steps=95000,
                               duration=95000,
                               num_particles=1,
                               rtol=rtol,
                               atol=atol,
                               method='DOP853')
        interact.produce_orbits()

        # Compute the error and execution time associated with this pair of rtol and atol
        rtol_errors.append(abs(interact.energy_deviation()))
        rtol_times.append(time.time() - start_time)

    # Plotting a line for each value of rtol on each figure.
    line1, = ax1.plot(atols, rtol_errors, "o-")
    line2, = ax2.plot(atols, rtol_times, "o-")
    lns1.append(line1)
    labs.append(str(rtol))
    lns2.append(line2)

ax1.legend(labels=labs, handles=lns1, loc='upper center', title='rtol values')
ax2.legend(labels=labs, handles=lns2, loc='upper center', title='rtol values')
plt.show()