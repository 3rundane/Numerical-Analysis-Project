# from convertProgram import *
from math import cos, sin
# from parameters import *
import numpy as np

# READ ME
# -------
# Vehicle receives 10 pieces of data from dat files
# 1. beginning time
# 2. ending time
# 3. number of steps of trip
# 4. degrees latitude
# 5. minutes latitude
# 6. seconds latitude
# 7. hemisphere (1 if Northern, -1 if Southern)
# 8. degrees longitude
# 9. minutes longitude
# 10. seconds longitude
# 11. hemisphere (1 if Eastern, -1 if Western)
# 12. altitude of destination, in meters

# Satellite then reads 10 pieces of data
# 1. t_v
# 2. degrees latitude
# 3. minutes latitude
# 4. seconds latitude
# 5. hemisphere (1 if Northern, -1 if Southern)
# 6. degrees longitude
# 7. minutes longitude
# 8. seconds longitude
# 9. hemisphere (1 if Eastern, -1 if Western)
# 10. altitude of destination, in meters

# Satellite has three tasks:
# --------------------------
# A. Compute position of vehicle, x_v, at time t_v.
#    - Essentially convert geo --> cartesian.
#    - Return cart value of position of vehicle
#
# B. Given x_v and t_v from A, determine which satellites are above the horizon at time t_v. Compute time t_s and x_s
#    for those satellites. Determine which satellites were above horizon at time t_s.
#
#     i) Determine if which satellites are above x_v at time t_v
#           - use equation (20) and exercise 8
#           - store i_s, satellite number [0,23] to some list/dictionary
#    ii) For Satellites from i) find t_s via fixed point iteration. Then x_s by equation (20)
#           - Use equation (20) and problem 9 to find t_s, stop program when c|t_v - t_s| < 10^-2 (1 cm of accuracy)
#           - Plug t_s into equation (20) to find x_s
#           - store t_s and x_s in list/dictionary?
#   iii) Final check. See if x_s positions from ii) satisfy exercise 8. Were they above horizon at necessary t_s time?
#           - store i_s, t_s, x_s for the satellites that remain.
#
# C. Write copy of standard input (data from Vehicle) and standard output(i_s, t_s, x_s) to log file.



# PROGRAM BEGINS
# --------------

# Read data.dat file. ~Dane

data = {0: [], 1: [], 23: []} # for testing purposes

# radius of earth (meters)
R = 6367444.50

# seconds for one sidereal day
s = 86164.09

# orbital perdiod of satelite
p = s/2

c = 2.997924580000000000E+08

pi =  3.141592653589793116E+00

# Grab all geo-coordinates from Vehicle program
# below for testing purposes
t_v = 1
lat_d = 1
lat_m = 1
lat_s = 1
NS = 1
log_d = 1
log_m = 1
log_s = 1
EW = 1
h = 1

# Functions

# To convert geo-degrees to radians
def deg_to_rad(deg, min, sec):
    rad = (deg + min / 60 + sec / 3600) / 180 * pi
    return rad
# convert geo to car of vehicle
def geo_to_car(t, lat_d, lat_m, lat_s, NS, log_d, log_m, log_s, EW, h):
    phi = deg_to_rad(lat_d, lat_m, lat_s)  # convert lattitude into radians
    theta = deg_to_rad(log_d, log_m, log_s)  # convert longitutde into radians
    alpha = abs((R + h) * cos(NS * phi))  # Projected radii onto the xy-plane

    # Grab car coordinates at t=0
    x0 = alpha * cos(EW * theta)
    y0 = alpha * sin(EW * theta)
    z0 = (R + h) * sin(NS * phi)

    # rotate <x0, y0, z0> about z-axis for t seconds. Gives correct cartesian values.
    Rot_matrix = np.array([[cos(2 * pi * t / s), -sin(2 * pi * t / s), 0],
                           [sin(2 * pi * t / s), cos(2 * pi * t / s), 0],
                           [0, 0, 1]])
    X0 = np.array([[x0], [y0], [z0]])
    xt, yt, zt = np.dot(Rot_matrix, X0)
    return xt[0], yt[0], zt[0]
# Equation (20) position of satellite at time t
def eqn20(t, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt):
    s_1 = (R + alt) * (u_1 * cos(2 * pi * t / per + phase) + v_1 * sin(2 * pi * t / per + phase))
    s_2 = (R + alt) * (u_2 * cos(2 * pi * t / per + phase) + v_2 * sin(2 * pi * t / per + phase))
    s_3 = (R + alt) * (u_3 * cos(2 * pi * t / per + phase) + v_3 * sin(2 * pi * t / per + phase))
    return s_1, s_2, s_3


# Part A.
x_1, x_2, x_3 = geo_to_car(t_v, lat_d, lat_m, lat_s, NS, log_d, log_m, log_s, EW, h)

# Part B
# i) Which satellites are above horizon at time t_v. Return in the good_satellite dict
good_satellites = {}
for key in data.keys():
    satellite_info = data[key]
    u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt = satellite_info
    s_1, s_2, s_3 = eqn20(t_v, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt)
    if x_1 * s_1 + x_2 * s_2 + x_3 * s_3 >= x_1 ** 2 + x_2 ** 2 + x_3 ** 2:
        good_satellites[key] = data[key]

# ii) Which of the previous satellites were above the horizon at t_s
best_boi_satellite = {}
for key in good_satellites.keys():
    satellite_info = best_boi_satellite[key]
    u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt = satellite_info

    # Begin fixed point iteration
    t_s = t_v
    while True:
        s_1, s_2, s_3 = eqn20(t_s, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt)
        t_s = t_v - np.sqrt((s_1 - x_1)**2 + (s_2 - x_2)**2 + (s_3 - x_3)**2) / c
        if c*abs(t_v - t_s) < 0.01:
            break
    if x_1 * s_1 + x_2 * s_2 + x_3 * s_3 >= x_1 ** 2 + x_2 ** 2 + x_3 ** 2:
        best_boi_satellite[key] = [t_s, s_1, s_2, s_3]

# At point where returns dict of key: satellite number, and value: time t_s and coordinates at t_s

# Part C. Not sure how to do this yet



