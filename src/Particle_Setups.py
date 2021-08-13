# Initialisation of test particle locations and velocities
# Set up for config1
i = int(k / 10)
radius = 2 / r_ref

if 0 <= i < 12:
    radius = 2 / r_ref
elif 12 <= i < 30:
    radius = 3 / r_ref
elif 30 <= i < 54:
    radius = 4 / r_ref
elif 54 <= i < 84:
    radius = 5 / r_ref
elif 84 <= i < 120:
    radius = 6 / r_ref