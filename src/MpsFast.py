import os
import time
import numpy as np
import scipy.integrate as integrate
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

rcParams['font.family'] = ['serif']
rcParams['font.serif'] = ["Times New Roman"]
plt.rc('font', size=14)  # default text size
plt.rc('axes', titlesize=16)  # title
plt.rc('axes', labelsize=14)  # x and y labels
plt.rc('xtick', labelsize=14)  # x tick labels
plt.rc('ytick', labelsize=14)  # y tick labels
plt.rc('legend', fontsize=14)  # legend


class MpsFast:
    def __init__(self, scaling_constants, init_heavy_1, init_heavy_2, particle_pos, particle_vels, heavy_masses,
                 number_test_particles, duration, steps, parallel, config_name):

        # Factors to convert equations of motion.
        g, mass, r, v, period = scaling_constants[0],\
                                scaling_constants[1],\
                                scaling_constants[2],\
                                scaling_constants[3],\
                                scaling_constants[4]

        alpha_1 = g * period * mass / ((r ** 2) * v)
        alpha_2 = v * period / r

        # Coefficients to 'fix' equations of motion.
        self.normalisation = [alpha_1, alpha_2]

        # File names to store data and plots
        pre_dir = os.path.dirname(os.path.realpath('__file__'))
        self.dir_name = os.path.join(pre_dir, f'{config_name}')

        if parallel and number_test_particles >= mp.cpu_count():
            self.workers = mp.cpu_count()
        else:
            self.workers = 1
        self.number_of_sub_simulations = self.workers  # SIMPLE CASE OF ONE GROUP PER WORKER

        # Initial parameters.
        self.mass1, self.mass2 = heavy_masses[0], heavy_masses[1]
        self.number_test_particles = number_test_particles

        # The value of our initial parameters for two heavy particles.
        self.initial_r_1, self.initial_v_1 = init_heavy_1[0], init_heavy_1[1]
        self.initial_r_2, self.initial_v_2 = init_heavy_2[0], init_heavy_2[1]
        self.particle_pos_inits = particle_pos
        self.particle_vel_inits = particle_vels

        # Other parameters that might be needed.
        self.softening_radius = 0.001
        self.duration = duration
        self.steps = steps

    def equations_of_motion(self, t, pos_and_vels):
        # Equations of motion which will be used by the odeint solver. pos_and_vels is our total state vector which
        # describes every particle in the system.
        positions, velocities = np.array(np.array_split(pos_and_vels, 2))

        # First we consider HEAVY MASSES.
        pos_heavy_1 = positions[:2]
        vel_heavy_1 = velocities[:2]
        pos_heavy_2 = positions[2:4]
        vel_heavy_2 = velocities[2:4]

        r = np.linalg.norm(pos_heavy_2 - pos_heavy_1) + self.softening_radius

        # Computing the rate of change of velocity and position for the heavy masses in vector form.
        vel_heavy_1_dot = self.normalisation[0] * self.mass2 * (pos_heavy_2 - pos_heavy_1) / (r ** 3)
        vel_heavy_2_dot = self.normalisation[0] * self.mass1 * (pos_heavy_1 - pos_heavy_2) / (r ** 3)
        pos_heavy_1_dot = self.normalisation[1] * vel_heavy_1
        pos_heavy_2_dot = self.normalisation[1] * vel_heavy_2

        # Now onto the TEST PARTICLES. Splitting remainder of arrays into groups of two for EACH particle.
        # particle_positions/velocities takes the form [ [x, y], [x, y], ... ]
        particle_positions = np.array(np.array_split(positions[4:], positions[4:].size//2))
        particle_velocities = np.array(np.array_split(velocities[4:], velocities[4:].size//2))

        # to_heavy_i is in the form [ [delta x, delta y], [delta x, delta y], ... ]
        to_heavy_1 = pos_heavy_1 - particle_positions
        to_heavy_2 = pos_heavy_2 - particle_positions

        # Distance to heavy masses including softening radius. keepdims makes sure that we end up with a 2D array.
        # r_to_i is in the form [ [distance], [distance], ... ]
        r_to_1 = np.linalg.norm(to_heavy_1, axis=1, keepdims=True) + self.softening_radius
        r_to_2 = np.linalg.norm(to_heavy_2, axis=1, keepdims=True) + self.softening_radius

        # Derivatives of velocity.
        particle_vel_dot = \
            np.divide(self.normalisation[0] * self.mass2 * to_heavy_2, r_to_2 ** 3) + \
            np.divide(self.normalisation[0] * self.mass1 * to_heavy_1, r_to_1 ** 3)

        particle_pos_dot = self.normalisation[1] * particle_velocities

        position_derivs = np.empty(np.shape(velocities))
        velocity_derivs = np.empty(np.shape(positions))

        # Assigning calculated derivatives to arrays to pass onto integrator.
        position_derivs[:2] = pos_heavy_1_dot
        position_derivs[2:4] = pos_heavy_2_dot
        position_derivs[4:] = particle_pos_dot.flatten()

        velocity_derivs[:2] = vel_heavy_1_dot
        velocity_derivs[2:4] = vel_heavy_2_dot
        velocity_derivs[4:] = particle_vel_dot.flatten()
        return np.concatenate((position_derivs, velocity_derivs))

    def produce_solution(self):
        # Splitting the test particle data into groups where each group can be run on one thread
        """
        We have one big simulation and we want to break it up and split it into smaller simulations to run separately.
        Suppose we have 4 processors. The simplest way to parallel this code is to split the simulation into 4
        sub-simulations and create 4 workers (one for each processor). Then we just start 4 processes. If the processes
        individually take long enough, then each worker will be used - one for each
        """
        start_time = time.time()

        # Break particle simulation data into
        sub_sim_positions = np.array_split(self.particle_pos_inits, self.number_of_sub_simulations)
        sub_sim_velocities = np.array_split(self.particle_vel_inits, self.number_of_sub_simulations)

        # Opens a pool with either 1 (for small particle numbers and non-parallel option) worker or one worker for every
        # cpu core.
        pool = mp.Pool(self.workers)

        for i in range(self.number_of_sub_simulations):
            pool.apply_async(self.solve_ode, args=(sub_sim_positions[i], sub_sim_velocities[i], i))

        pool.close()
        pool.join()

        duration = time.time() - start_time
        print(f"--- {duration:.2f} seconds ---")

    def solve_ode(self, positions_in_thread, vels_in_thread, i):
        print(f"Job {i} started.")

        t = np.arange(0, self.duration, self.duration / self.steps)

        # Have had to adjust mxstep so that the integrator doesn't give up too soon
        initials = np.concatenate((self.initial_r_1, self.initial_r_2, positions_in_thread.flatten(),
                                   self.initial_v_1, self.initial_v_2, vels_in_thread.flatten()))

        # The integration magic...
        solution = integrate.solve_ivp(fun=self.equations_of_motion,
                                       method='DOP853',
                                       y0=initials,
                                       t_span=(0, self.duration),
                                       t_eval=t,
                                       atol=1e-10,
                                       rtol=1e-10)
        solution = solution.y.T

        # Saving data as .npy file
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)

        np.save(f'{self.dir_name}/thread_{i}', solution)
        print(f"Job {i} finished.")

    def static_plot(self):
        # Plotting code which will draw the data files produced on each thread and then plot them all on the same plot
        pos_heavy1, pos_heavy2, xs, ys, distances_to_first_mass = [], [], [], [], []

        for sub_sim in range(self.number_of_sub_simulations):
            # Extracting data for plotting.
            try:
                solution = np.load(f'{self.dir_name}/thread_{sub_sim}.npy')
                positions, velocities = np.array(np.array_split(solution, 2, axis=1))
            except FileNotFoundError:
                print("Please enter correct configuration name and make sure solution has been produced.")
                raise

            if sub_sim == 0:
                pos_heavy1 = solution[-1][:2]
                pos_heavy2 = solution[-1][2:4]

            # Calculating how many particles where solved in this thread
            # Every particle has 4 numbers in the solution - 2 position and 2 velocity
            number_in_sub_sim = np.shape(solution)[1] // 4 - 2

            for i in range(number_in_sub_sim):
                last_position = positions[-1][2 * i + 4:2 * i + 6]
                xs.append(last_position[0])
                ys.append(last_position[1])
                distances_to_first_mass.append(np.linalg.norm(last_position - pos_heavy1))

        fig1, ax1 = plt.subplots(figsize=(10, 10))
        fig1.gca().set_aspect('equal', adjustable='box')

        # Plotting final positions for all particles
        ax1.scatter(xs, ys, color='grey', s=10, edgecolor='black')  # Last position
        ax1.scatter(pos_heavy2[0], pos_heavy2[1], color='red', s=100)
        ax1.scatter(pos_heavy1[0], pos_heavy1[1], color='blue', s=100)

        ax1.grid(linestyle='dashed')

        # Styling for plot
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_title(
            f'Trajectories for {self.number_test_particles} test particles. Duration {self.duration}, steps {self.steps}.\n'
            f'Heavy masses of mass {self.mass1} and {self.mass2}')
        fig1.savefig(f'{self.dir_name}/plot.png', dpi=600, bbox_inches='tight')

        circ = plt.Circle(pos_heavy1, 7, fill=False)
        ax1.add_artist(circ)
        fig1.savefig(f'{self.dir_name}/distribution.png', dpi=600, bbox_inches='tight')
        # "sum(map(lambda x: (x <= 7), distances_to_first_mass))" is code to count number un-disturbed

    def animate(self, i, scatters, data):
        # Function to animate the solution
        index_to_plot = 100 * i + 1
        if index_to_plot >= len(data[0]):
            index_to_plot = len(data[0]) - 1
        for (j, scat) in enumerate(scatters):
            scat.set_data(data[j][index_to_plot][0], data[j][index_to_plot][1])
        return scatters

    def produce_animation(self):
        # Extracting solution and setting up plots.
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(xlim=(-40, 60), ylim=(-100, 20))

        scatters, data = [], []  # List to hold scatter plots and the data

        for sub_sim in range(self.number_of_sub_simulations):
            # Extracting data for plotting.
            try:
                solution = np.load(f'{self.dir_name}/thread_{sub_sim}.npy')
            except FileNotFoundError:
                print("Please enter correct configuration name and make sure solution has been produced.")
                raise

            if sub_sim == 0:
                # Plotting the data for the heavy particles
                pos_heavy1 = solution[:, :2]
                pos_heavy2 = solution[:, 2:4]

                data.append(pos_heavy1)
                data.append(pos_heavy2)

                scatters.append(ax.plot([], [], color='blue', marker="o", markersize=10)[0])
                scatters.append(ax.plot([], [], color='red', marker="o", markersize=10)[0])

            # Calculating how many particles where solved in this thread - every particle (heavy or not) has 4 numbers
            # in the solution - 2 position and 2 velocity
            number_in_sub_sim = np.shape(solution)[1] // 4 - 2
            for i in range(number_in_sub_sim):
                position = solution[:, 2 * i + 4:2 * i + 6]
                data.append(position)
                # Scatter for all the test particles
                scatters.append(ax.plot([], [], color='grey', marker="o", markersize=2, markeredgecolor='black')[0])

        # Styling
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Animating the trajectories of all particles')

        # Producing the animation and saving to directory.
        anim = FuncAnimation(fig, self.animate, fargs=(scatters, data), interval=40, blit=True)
        anim.save(f'{self.dir_name}/animation.gif', writer='pillow', fps=15)