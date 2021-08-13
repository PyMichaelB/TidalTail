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

masses = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
          1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

particles_config1 = [1200, 1200, 1200, 1200, 1200, 1164, 1109, 1073, 1015, 922, 871,
                777, 753, 697, 684, 657, 608, 569, 530, 454, 472]


particles_config2 = [1200, 1200, 1200, 1200, 1200, 1198, 1178, 1151, 1122, 1062, 1007,
                958, 945, 890, 812, 765, 715, 668, 630, 577, 549]

plt.plot(masses, particles_config2, lw=2, label='Configuration 2')
plt.plot(masses, particles_config1, lw=2, label='Configuration 1')

plt.grid(linestyle='dashed')
plt.xlabel('Perturbing particle mass')
plt.ylabel('Number of particles within 7 units radius')
plt.title('Number of particles un-disturbed against perturbing mass')
plt.legend(fontsize=14)
plt.show()
