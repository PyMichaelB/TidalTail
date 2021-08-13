from MpsFast import MpsFast
import numpy as np
import random

"""
Notes about use of if __name__ == "__main__". In Windows, there is no fork and so when the multiprocessing module
tries to create a new process on each thread, it tries to import the calling module (HeavyMass.py) each time. This results 
in issues because without if __name__ == "__main__", every time it is imported, all the code is run. So you get an infinite 
number of processes trying to form. We put all the code inside if __name__ == "__main__" so that none of it is run UNLESS
it is the main script - its intended use. On Linux, we have fork and so it doesn't need to import the module for every new process.
"""

if __name__ == "__main__":
    # Reference quantities
    G = 1
    mass_ref = 1
    r_ref = 1
    v_ref = 1
    period_ref = 1

    initial_heavy_1 = np.array([[0, 0], [0, 0]])
    m1 = 1  # First heavy mass

    initial_heavy_2 = np.array([[20, 30], [0, -np.sqrt(2 / np.sqrt((20**2)+(30**2)))]])
    m2 = 1.1  # Second heavy mass

    num_test = 10000

    # Initialise the test particle positions and velocities array
    particle_pos = np.empty((num_test, 2))
    particle_vels = np.empty((num_test, 2))

    for k in range(num_test):
        radius = 2 + (6 - 2) * (k / num_test)
        theta = random.random() * 2 * np.pi
        speed = np.sqrt(G * m1 / radius)
        particle_pos[k] = np.array([radius*np.cos(theta), radius*np.sin(theta)])
        particle_vels[k] = np.array([-speed*np.sin(theta), speed*np.cos(theta)])

    duration, steps = 150, 10000
    config = f'../data/config/{num_test}_{duration}_{steps}_{m1}_{m2}'
    system = MpsFast(scaling_constants=[G, mass_ref, r_ref, v_ref, period_ref],
                     init_heavy_1=initial_heavy_1,
                     init_heavy_2=initial_heavy_2,
                     particle_pos=particle_pos,
                     particle_vels=particle_vels,
                     heavy_masses=[m1, m2],
                     number_test_particles=num_test,
                     duration=duration,
                     steps=steps,
                     parallel=True,
                     config_name=config)

    system.produce_solution()
    system.static_plot()
    system.produce_animation()

