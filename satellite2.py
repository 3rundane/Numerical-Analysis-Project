from math import cos, sin, sqrt
# from parameters import *
# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, SIG_DFL)
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
import sys
from typing import Optional, Any

import numpy as np
from decimal import *


def grab_datafile():
    File_object = open(r"data.dat", "r")
    data_dat = File_object.readlines()
    # print(data_dat)
    # length of data set
    length = len(data_dat)

    # Holds the data to the left of the delimiter '/='
    data_dat_raw = np.zeros(length, dtype=Decimal)
    # #OPTIONAL holds the original names of the values in data_dat_raw
    # data_dat_names = np.zeros(length)

    satellite_data = {"0": np.zeros(9, dtype=Decimal), "1": np.zeros(9, dtype=Decimal), "2": np.zeros(9, dtype=Decimal),
                      "3": np.zeros(9, dtype=Decimal), "4": np.zeros(9, dtype=Decimal), "5": np.zeros(9, dtype=Decimal),
                      "6": np.zeros(9, dtype=Decimal), "7": np.zeros(9, dtype=Decimal),
                      "8": np.zeros(9, dtype=Decimal), "9": np.zeros(9, dtype=Decimal),
                      "10": np.zeros(9, dtype=Decimal), "11": np.zeros(9, dtype=Decimal),
                      "12": np.zeros(9, dtype=Decimal), "13": np.zeros(9, dtype=Decimal),
                      "14": np.zeros(9, dtype=Decimal), "15": np.zeros(9, dtype=Decimal),
                      "16": np.zeros(9, dtype=Decimal), "17": np.zeros(9, dtype=Decimal),
                      "18": np.zeros(9, dtype=Decimal), "19": np.zeros(9, dtype=Decimal),
                      "20": np.zeros(9, dtype=Decimal), "21": np.zeros(9, dtype=Decimal),
                      "22": np.zeros(9, dtype=Decimal), "23": np.zeros(9, dtype=Decimal)}

    constants_names = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0}
    # This variable is a temporary container meant to hold a given satellite's 9 pieces of information.
    for i in range(length):
        # data_dat_raw[i] = float(data_dat[i].split('/=')[0])
        data_dat_raw[i] = Decimal(data_dat[i].split('/=')[0])

    def fill_data(i, data_dat_raw):
        data_holder2 = np.zeros(9, dtype=Decimal)
        data_holder2[0] = data_dat_raw[i]
        data_holder2[1] = data_dat_raw[i + 1]
        data_holder2[2] = data_dat_raw[i + 2]
        data_holder2[3] = data_dat_raw[i + 3]
        data_holder2[4] = data_dat_raw[i + 4]
        data_holder2[5] = data_dat_raw[i + 5]
        data_holder2[6] = data_dat_raw[i + 6]
        data_holder2[7] = data_dat_raw[i + 7]
        data_holder2[8] = data_dat_raw[i + 8]
        return data_holder2

    ## big problem with readlines, the full number from the data file is not going in.

    for i in range(length):
        if i <= 3:
            constants_names[str(i)] = data_dat_raw[i]
        if i == 4:
            satellite_data["0"] = fill_data(i, data_dat_raw)
        if i == 13:
            satellite_data["1"] = fill_data(i, data_dat_raw)
        if i == 22:
            satellite_data["2"] = fill_data(i, data_dat_raw)
        if i == 31:
            satellite_data["3"] = fill_data(i, data_dat_raw)
        if i == 40:
            satellite_data["4"] = fill_data(i, data_dat_raw)
        if i == 49:
            satellite_data["5"] = fill_data(i, data_dat_raw)
        if i == 58:
            satellite_data["6"] = fill_data(i, data_dat_raw)
        if i == 67:
            satellite_data["7"] = fill_data(i, data_dat_raw)
        if i == 76:
            satellite_data["8"] = fill_data(i, data_dat_raw)
        if i == 85:
            satellite_data["9"] = fill_data(i, data_dat_raw)
        if i == 94:
            satellite_data["10"] = fill_data(i, data_dat_raw)
        if i == 103:
            satellite_data["11"] = fill_data(i, data_dat_raw)
        if i == 112:
            satellite_data["12"] = fill_data(i, data_dat_raw)
        if i == 121:
            satellite_data["13"] = fill_data(i, data_dat_raw)
        if i == 130:
            satellite_data["14"] = fill_data(i, data_dat_raw)
        if i == 139:
            satellite_data["15"] = fill_data(i, data_dat_raw)
        if i == 148:
            satellite_data["16"] = fill_data(i, data_dat_raw)
        if i == 157:
            satellite_data["17"] = fill_data(i, data_dat_raw)
        if i == 166:
            satellite_data["18"] = fill_data(i, data_dat_raw)
        if i == 175:
            satellite_data["19"] = fill_data(i, data_dat_raw)
        if i == 184:
            satellite_data["20"] = fill_data(i, data_dat_raw)
        if i == 193:
            satellite_data["21"] = fill_data(i, data_dat_raw)
        if i == 202:
            satellite_data["22"] = fill_data(i, data_dat_raw)
        if i == 211:
            satellite_data["23"] = fill_data(i, data_dat_raw)
            return satellite_data, constants_names
        File_object.close()

satellite_data, constants_names = grab_datafile()

R = Decimal(constants_names["2"])
s = Decimal(constants_names["3"])
p = s / Decimal(2)
c = Decimal(constants_names["1"])
pi = Decimal(constants_names["0"])
# Read in data from standard input. This command is to be used when piping data from a file into this program via the
# command line. e.g. cat data.dat | readingfiles_trial.py
log = open("satellite.log","w")

log.write("satellite log, Dane Gollero, Ike Griss Salas, Magon Bowling \n \n data.dat \n")
log.write(f"pi = {pi} \nc = {c} \nR = {R} \ns = {s} \n")
##iterate through dictionary satellite_data
u1 = 0
u2 = 0
u3 = 0
v1 = 0
v2 = 0
v3 = 0
data_period = 0
data_phase = 0
data_alt = 0
for keys in satellite_data.keys():
    u1 = satellite_data[keys][0]
    u2 = satellite_data[keys][1]
    u3 = satellite_data[keys][2]

    v1 = satellite_data[keys][3]
    v2 = satellite_data[keys][4]
    v3 = satellite_data[keys][5]

    data_period = satellite_data[keys][6]
    data_alt = satellite_data[keys][7]
    data_phase = satellite_data[keys][8]

    log.write(f"u[{int(keys)}][0] = {u1} \nu[{int(keys)}][1] = {u2} \nu[{int(keys)}][2] = {u3} \n")
    log.write(f"v[{int(keys)}][0] = {v1} \nv[{int(keys)}[1] = {v2} \nv[{int(keys)}][2] = {v3} \n")
    log.write(f"period[{int(keys)}] = {data_period} \naltitude[{int(keys)}] = {data_alt} \nphase[{int(keys)}] = {data_phase} \n")
log.write("\n end data.dat\n\n")
vehicle_dat_raw = sys.stdin.readlines()

# File_object = open(r"b12.dat", "r")
# vehicle_dat_raw = File_object.readlines()
length_raw = len(vehicle_dat_raw)

for i in range(length_raw):

    log.write(f" --- epoch = {i}\nread {vehicle_dat_raw[i]} \nwrote:\n")
    vehicle_dat = vehicle_dat_raw[i].split()
    truth = (vehicle_dat == [])
    # if statement here protects against an index out of bounds error.
    if truth:
        # print('hit')
        break
    # print(type(vehicle_dat[0]))
    t_v = Decimal(vehicle_dat[0])

    lat_d = Decimal(vehicle_dat[1])
    lat_m = Decimal(vehicle_dat[2])
    lat_s = Decimal(vehicle_dat[3])
    NS = Decimal(vehicle_dat[4])
    log_d = Decimal(vehicle_dat[5])
    log_m = Decimal(vehicle_dat[6])
    log_s = Decimal(vehicle_dat[7])
    EW = Decimal(vehicle_dat[8])
    h = Decimal(vehicle_dat[9])



    # Functions

    # # To convert geo-degrees to radians
    def deg_to_rad(deg, min, sec):
        rad = (deg + min / Decimal(60) + sec / Decimal(3600)) / Decimal(180) * pi
        return rad


    # Convert geo to cart at time t
    def geo_to_car(t, lat_d, lat_m, lat_s, NS, log_d, log_m, log_s, EW, h):
        phi = 2 * pi * NS * (lat_d / Decimal(360) + lat_m / Decimal((360 * 60)) + lat_s / Decimal((360 * 60 * 60)))
        lamb = 2 * pi * EW * (log_d / Decimal(360) + log_m / Decimal((360 * 60)) + log_s / Decimal((360 * 60 * 60)))

        x0 = (R + h) * Decimal(cos(phi)) * Decimal(cos(lamb))
        y0 = (R + h) * Decimal(cos(phi)) * Decimal(sin(lamb))
        z0 = (R + h) * Decimal(sin(phi))

        # Rotate for t seconds
        alpha = 2 * pi * t / s

        x = x0 * Decimal(cos(alpha)) - y0 * Decimal(sin(alpha))
        y = x0 * Decimal(sin(alpha)) + y0 * Decimal(cos(alpha))
        z = z0

        return x, y, z


    # def eqn20(t, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt):

    # Equation (20) position of satellite at time t
    ## am I losing precision when computing cos and sin from the fact that they are returning floats??
    def eqn20(t, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt):
        s_1 = (R + alt) * (u_1 * Decimal(cos(2 * pi * t / per + phase)) + v_1 * Decimal(sin(2 * pi * t / per + phase)))
        s_2 = (R + alt) * (u_2 * Decimal(cos(2 * pi * t / per + phase)) + v_2 * Decimal(sin(2 * pi * t / per + phase)))
        s_3 = (R + alt) * (u_3 * Decimal(cos(2 * pi * t / per + phase)) + v_3 * Decimal(sin(2 * pi * t / per + phase)))
        return s_1, s_2, s_3


    # Part A.
    x_1, x_2, x_3 = geo_to_car(t_v, lat_d, lat_m, lat_s, NS, log_d, log_m, log_s, EW, h)

    # Part B
    # i) Which satellites are above horizon at time t_v. Return in the good_satellite dict
    good_satellites = {}
    for key in satellite_data.keys():

        satellite_info = satellite_data[key]

        u_1, u_2, u_3, v_1, v_2, v_3, per, alt, phase = satellite_info
        s_1, s_2, s_3 = eqn20(t_v, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt)
        # peter's condition of above the horizon uses > not >=...
        ##Still need to check below...
        if (x_1 * s_1) + (x_2 * s_2) + (x_3 * s_3) >= (x_1 ** 2) + (x_2 ** 2) + (x_3 ** 2):
            # print('happy')
            good_satellites[key] = satellite_data[key]

    # ii) Which of the previous satellites were above the horizon at t_s
    best_boi_satellite = {}
    # tracker = 0
    for key in good_satellites.keys():
        # satellite_info = best_boi_satellite[key]
        satellite_info = good_satellites[key]

        u_1, u_2, u_3, v_1, v_2, v_3, per, alt, phase = satellite_info
        # Begin fixed point iteration
        t_2 = t_v
        while True:
            # print(tracker)
            #
            t_1 = t_2
            s_1, s_2, s_3 = eqn20(t_1, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt)
            t_2 = t_v - Decimal(sqrt((s_1 - x_1) ** 2 + (s_2 - x_2) ** 2 + (s_3 - x_3) ** 2)) / c
            # print(t_s)
            # print(abs(t_2 - t_1))
            #     t_s = t_s - numerator/derivative
            if c*abs(t_2 - t_1) < Decimal(0.01):
                s_1, s_2, s_3 = eqn20(t_1, u_1, u_2, u_3, v_1, v_2, v_3, per, phase, alt) #might be a fix... default is nothing here
                break
            # Algorithm has a problem because once it finds the value of t_s it doesn't update s_1,s_2,s_3 to reflect
            # that. maybe assume the opposite condition and delete those satellites who satisfy it?
        if x_1 * s_1 + x_2 * s_2 + x_3 * s_3 >= x_1 ** 2 + x_2 ** 2 + x_3 ** 2:
            # below might be a problem idk.
            best_boi_satellite[key] = [t_2, s_1, s_2, s_3]
            # print(key, t_2, s_1, s_2, s_3)
            string_output = str(key) + ' ' + str(t_2) + ' ' + str(s_1) + ' ' + str(s_2) + ' ' + str(s_3)
            # print('hello')
            log.write(string_output + '\n')
            print(string_output)
    log.write('\n')
log.write("Dirty D, Griss the Grit, Mighty Tiny waz here")
log.close()
