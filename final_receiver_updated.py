import numpy as np
import sys
from decimal import *
from math import sqrt, atan, cos, sin, asin
import numpy.linalg as la

#########################
##### RECEIVER CODE #####
#########################

######################
# DANE GOLLERO #######
# IKE GRISS SALAS ####
# MAGON BOWLING ######
######################

######################
# UNIVERSITY OF UTAH #
# MATH 5600 ##########
######################

# function to gather data from data.dat
def grab_datafile():
    File_object = open(r"data.dat", "r")
    data_dat = File_object.readlines()
    length = len(data_dat)

    # Holds the data to the left of the delimiter '/='
    data_dat_raw = np.zeros(length, dtype=Decimal)

    # #OPTIONAL holds the original names of the values in data_dat_raw
    # data_dat_names = np.zeros(length)

    # Initialize dictionary to hold satellite data from data.dat
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
        data_dat_raw[i] = Decimal(data_dat[i].split('/=')[0])

    # function to collect data relative to a given satellite.
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

    # Iterate through data.dat and place constants and satellite info in appropriate dictionary.
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

# Assign dictionary containing satellite information and relevant constants.
satellite_data, constants_names = grab_datafile()

# Define parameters from data.dat
R = float(constants_names["2"])
ss = float(constants_names["3"]) #do not name s!!!
p = float(ss) / 2
c = float(constants_names["1"])
pi = float(constants_names["0"])

# Initialize log file/write over existing log file (in same directory as receiver.py)
log = open("receiver.log","w")

log.write(f"receiver log, Dane Gollero, Magon Bowling, Ike Griss Salas \n\n data.dat: \n\npi = {pi} \nc = {c} \nR = {R} \ns = {ss} \n\n end data.dat\n\n")

####################
###### NORMS #######
####################
### equations (68)
# i_s is the key of a given satellite whose norm of the difference between it and the current-iteration vehicle position is given.
# x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
# s_1,s_2,s_3 are the x,y,z components of the satellite at time t_s.
# norm will be used in each iteration through being called in A below, as well as other places in the Newton's method iteration.
def norm(i_s, x_1, x_2, x_3, input_satellites):
    s_1 = input_satellites[i_s][1]
    s_2 = input_satellites[i_s][2]
    s_3 = input_satellites[i_s][3]
    return sqrt((s_1 - x_1) ** 2 + (s_2 - x_2) ** 2 + (s_3 - x_3) ** 2)

##this is the norm to be used in Newton' method iteration, NOT in the function definitions below.
def normal_norm(s_1, s_2, s_3):
    return sqrt(s_1 ** 2 + s_2 ** 2 + s_3 ** 2)

#########################################
## DIFFERENCE EQUATION AND DERIVATIVES ##
#########################################

# difference equations are coded below. This function returns the ith output of the A_i in Peter's equation 68. This will be called at each
# step in the Newton's method iteration.
# i_current and i_subsequent are the ith and (i + 1)th satellite keys to be used for computing the ith and (i+1)th norm.
# t_current and t_subsequent are the t_s's of the ith and (i+1)th satellites.
##x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
def A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    t_current = input_satellites[i_current][0]
    t_subsequent = input_satellites[i_subsequent][0]
    return subsequent_norm - current_norm - c * (t_current - t_subsequent)

# partial derivative wrt first coordinate of A or the difference equation.
def X(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]
    return -(s_1_subsequent - x_1) / subsequent_norm + (s_1_current - x_1) / current_norm

# partial derivative wrt second coordinate of A or the difference equation.
def Y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]
    return -(s_2_subsequent - x_2) / subsequent_norm + (s_2_current - x_2) / current_norm

# partial derivative wrt third coordinate of A or the difference equation.
def Z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]
    return -(s_3_subsequent - x_3) / subsequent_norm + (s_3_current - x_3) / current_norm

#################
## EQUATION 69 ##
#################

## Everything below until HESSSSSIIIaAAANN TIMMMEEE is for the right hand side of our Newton's method iteration.
def f(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    f_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        f_value = (A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)) ** 2 + f_value
        index = index - 1
    return f_value

##Partial derivatives of little f wrt x. Recursive definition.
def f_partial_x(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    f_partial_xvalue = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        f_partial_xvalue = f_partial_xvalue + A_i * X_i
        index = index - 1
    return 2 * f_partial_xvalue

##Partial derivatives of little f wrt y. Recursive definition.
def f_partial_y(index, x_1, x_2, x_3, input_satellites):
    f_partial_xvalue = 0
    satellite_keys = [i for i in input_satellites.keys()]
    f_partial_yvalue = 0

    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        f_partial_yvalue = f_partial_yvalue + A_i * Y_i
        index = index - 1
    return 2 * f_partial_yvalue

##Partial derivatives of little f wrt z. Recursive definition.
def f_partial_z(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    f_partial_zvalue = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        f_partial_zvalue = f_partial_zvalue + A_i * Z_i
        index = index - 1
    return 2 * f_partial_zvalue

#################
## EQUATION 71 ##
#################

######################################
## HESSSSSSIAAAAANNNNNNN TIMMMEEEEE ##
######################################

##the order is the same as that in Peter's homework 1
##partial X partial x
def X_partial_x(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]
    current_fraction = (current_norm ** 2 - (s_1_current - x_1) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_1_subsequent - x_1) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction

##partial Y partial x
def Y_partial_x(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    # x_s_i, x_s_i+1 respectively
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]

    # y_s_i, y_s_i+1, respectively
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]

    current_fraction = ((s_2_current - x_2) * (s_1_current - x_1)) / (current_norm ** 3)
    subsequent_fraction = -((s_2_subsequent - x_2) * (s_1_subsequent - x_1)) / (subsequent_norm ** 3)
    return subsequent_fraction + current_fraction

##partial X partial y
def X_partial_y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    return Y_partial_x(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)

##partial X partial z
def X_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    # x_s_i, x_s_i+1 respectively
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]

    # z_s_i, z_s_i+1, respectively
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]

    current_fraction = ((s_1_current - x_1) * (s_3_current - x_3)) / (current_norm ** 3)
    subsequent_fraction = -((s_1_subsequent - x_1) * (s_3_subsequent - x_3)) / (subsequent_norm ** 3)
    return subsequent_fraction + current_fraction

##partial Z partial x
def Z_partial_x(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    return X_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)

##partial Y partial y
def Y_partial_y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]
    current_fraction = (current_norm ** 2 - (s_2_current - x_2) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_2_subsequent - x_2) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction

##partial Y partial z
def Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    # y_s_i, y_s_i+1 respectively
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]

    # z_s_i, z_s_i+1, respectively
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]

    current_fraction = ((s_2_current - x_2) * (s_3_current - x_3)) / (current_norm ** 3)
    subsequent_fraction = -((s_2_subsequent - x_2) * (s_3_subsequent - x_3)) / (subsequent_norm ** 3)
    return subsequent_fraction + current_fraction

##partial Z partial y
def Z_partial_y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    return Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)

##partial Z partial z
def Z_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites):
    current_norm = norm(i_current, x_1, x_2, x_3, input_satellites)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3, input_satellites)
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]
    current_fraction = (current_norm ** 2 - (s_3_current - x_3) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_3_subsequent - x_3) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction

#################
## EQUATION 70 ##
#################

##partialf^2 partial x,x
def f_double_partial_xx(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        X_partial_x_i = X_partial_x(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value = X_i ** 2 + A_i * X_partial_x_i + almost_final_value
        index = index - 1
    return 2 * almost_final_value

##partialf^2 partial x,y
def f_double_partial_xy(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        X_partial_y_i = X_partial_y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value += X_i * Y_i + A_i * X_partial_y_i
        index = index - 1
    return 2 * almost_final_value

##partialf^2 partial y,x
def f_double_partial_yx(index, x_1, x_2, x_3, input_satellites):
    return f_double_partial_xy(index, x_1, x_2, x_3, input_satellites)

##partialf^2 partial xz
def f_double_partial_xz(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        X_partial_z_i = X_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value += X_i * Z_i + A_i * X_partial_z_i
        index = index - 1
    return 2 * almost_final_value

##partial^2 partial zx
def f_double_partial_zx(index, x_1, x_2, x_3, input_satellites):
    return f_double_partial_xz(index, x_1, x_2, x_3, input_satellites)

##partialf^2 partial yy
def f_double_partial_yy(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Y_partial_y_i = Y_partial_y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value = Y_i ** 2 + A_i * Y_partial_y_i + almost_final_value
        index = index - 1
    return 2 * almost_final_value

##partial^2 partial yz
def f_double_partial_yz(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Y_partial_z_i = Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value = Y_i * Z_i + A_i * Y_partial_z_i + almost_final_value
        index = index - 1
    return 2 * almost_final_value

##partial f^2 partial zy
def f_double_partial_zy(index, x_1, x_2, x_3, input_satellites):
    return f_double_partial_yz(index, x_1, x_2, x_3, input_satellites)

###partial^2 partial zz
def f_double_partial_zz(index, x_1, x_2, x_3, input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    almost_final_value = 0
    while index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        Z_partial_z_i = Z_partial_z(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3, input_satellites)
        almost_final_value = Z_i ** 2 + A_i * Z_partial_z_i + almost_final_value
        index = index - 1
    return 2 * almost_final_value

####################
## HESSIAN VALUES ##
####################

##begin processing with Newton's method to find x_s
# the negative of the gradient of f
def B(index, x_1, x_2, x_3, input_satellites):
    f_x = f_partial_x(index, x_1, x_2, x_3, input_satellites)
    f_y = f_partial_y(index, x_1, x_2, x_3, input_satellites)
    f_z = f_partial_z(index, x_1, x_2, x_3, input_satellites)
    b_ret = [-1 * f_x, -1 * f_y, -1 * f_z]
    return b_ret

# compute and return the hessian rows for a given input.
def Hessian_rows(index, x_1, x_2, x_3, input_satellites):
    f_xx = f_double_partial_xx(index, x_1, x_2, x_3, input_satellites)
    f_xy = f_double_partial_xy(index, x_1, x_2, x_3, input_satellites)
    f_yx = f_xy
    f_xz = f_double_partial_xz(index, x_1, x_2, x_3, input_satellites)
    f_zx = f_xz
    f_yz = f_double_partial_yz(index, x_1, x_2, x_3, input_satellites)
    f_zy = f_yz
    f_yy = f_double_partial_yy(index, x_1, x_2, x_3, input_satellites)
    f_zz = f_double_partial_zz(index, x_1, x_2, x_3, input_satellites)
    return [f_xx, f_xy, f_xz], [f_yx, f_yy, f_yz], [f_zx, f_zy, f_zz]

#########################
## CONVERSION FORMULAS ##
#########################

# # To convert geo-degrees to radians
def deg_to_rad(deg, min, sec):
    rad = pi * (deg + min / 60 + sec / 3600) / 180
    return rad

# Convert geo to cart at time t
def geo_to_car(t, lat_d, lat_m, lat_s, NS, log_d, log_m, log_s, EW, h):
    phi = 2 * pi * NS * (lat_d / 360 + lat_m / (360 * 60) + lat_s / (360 * 60 * 60))
    lamb = 2 * pi * EW * (log_d / 360 + log_m / (360 * 60) + log_s / (360 * 60 * 60))

    x0 = (R + h) * cos(phi) * (cos(lamb))
    y0 = (R + h) * cos(phi) * sin(lamb)
    z0 = (R + h) * sin(phi)

    # Rotate for t seconds
    alpha = 2 * pi * t / ss

    x = x0 * cos(alpha) - y0 * sin(alpha)
    y = x0 * sin(alpha) + y0 * cos(alpha)
    z = z0

    return x, y, z

# Convert radians to degrees.
def rad_to_deg(rad):
    total_seconds = int(round(rad * 180 / pi * 3600))
    sec = total_seconds % 60
    min = (total_seconds // 60) % 60
    deg = total_seconds // 3600
    return deg, min, sec

# Convert cartesian coordinates to geographic coordinates.
def car_to_geo(xt, yt, zt, t):
    # Un-rotate for t seconds.
    x = xt * cos(2 * pi / ss * t) + yt * sin(2 * pi / ss * t)
    y = -xt * sin(2 * pi / ss * t) + yt * cos(2 * pi / ss * t)
    z = zt

    # Define h
    h = sqrt(x ** 2 + y ** 2 + z ** 2) - R
    # Define North South
    if z < 0:
        NS = -1
    else:
        NS = 1

    # Define psi (Latitude)
    psi = abs(asin(z / sqrt(x ** 2 + y ** 2 + z ** 2)))

    psi_degree, psi_minute, psi_second = rad_to_deg(psi)

    # Define West East
    if y < 0:
        EW = -1
    else:
        EW = 1
    # Define lambda (Longitude)
    if x == 0:
        lambda_whole = pi / 2
        # print('1')
    elif (y == 0) and (x > 0):
        lambda_whole = 0
        # print('2')
    elif (y == 0) and (x < 0):
        lambda_whole = pi
        # print('3')
    elif (y != 0) and (x > 0):
        lambda_whole = abs(atan(y / x))
        # print('4')
    elif (y > 0) and (x < 0):
        lambda_whole = pi + atan(y / x)
        # print('5')
    else:
        lambda_whole = pi - atan(y / x)
        # print('6')
    # Peters but didn't work for whatever reason
    # if (x > 0) and (y > 0):
    #     lambda_whole = atan(y/x)
    #     # print(f"1: {lambda_whole}")
    # elif x < 0:
    #     lambda_whole = pi + atan(y / x)
    #     # print(f"2: {lambda_whole}, {y/x}")
    # else:
    #     lambda_whole = 2*pi + atan(y/x)
    # print(f"3: {lambda_whole}")
    # print(lambda_whole)
    # print(pi)

    lambda_degree, lambda_minute, lambda_second = rad_to_deg(lambda_whole)

    return t, psi_degree, psi_minute, psi_second, NS, lambda_degree, lambda_minute, lambda_second, EW, h

# Function for explicit purpose of logging.
def make_life_easier(input_satellites):
    counter = 0
    for keys in input_satellites.keys():
        sat_num = keys
        time = input_satellites[keys][0]
        x = input_satellites[keys][1]
        y = input_satellites[keys][2]
        z = input_satellites[keys][3]

        log.write(f"{counter} {sat_num} {time} {x} {y} {z} \n")
        counter = counter + 1

#######################
## HESSIAN ITERATION ##
#######################

# Function to compute iteration using Hessian.
def iteration_step(input_satellites,initial_guess):
    # For logging
    make_life_easier(input_satellites)

    # Create list of current satellite keys for iterating over.
    satellite_keys = [i for i in input_satellites.keys()]

    # Number of satellites in epoch
    satty_length = len(satellite_keys)

    #Initalize count
    num_iterations = 0

    # Initial guess for iteration (as string)
    x_0, y_0, z_0 = initial_guess[0],initial_guess[1],initial_guess[2]
    log.write(f" starting at: {x_0} {y_0} {z_0}\n")

    # Initial guess for iteration (as float)
    x, y, z = float(x_0), float(y_0), float(z_0)


    condition = True
    while condition:
        # Compute rows of Hessian matrix.
        hessian_row1, hessian_row2, hessian_row3 = Hessian_rows(satty_length - 1, x, y, z, input_satellites)

        # Compute B vector for solving Hessian equation.
        B_vector = B(satty_length - 1, x, y, z, input_satellites)

        # Place Hessian as array
        H = np.array([hessian_row1, hessian_row2, hessian_row3])

        # Solve Hessian equation. (Newton's step)
        s = np.linalg.solve(H, B_vector)

        # X_n+1 = X_n + s
        x = x + s[0]
        y = y + s[1]
        z = z + s[2]

        # Update count. For logging purposes.
        num_iterations = num_iterations + 1

        # Check if within 1cm. Break loop if true.
        if normal_norm(s[0], s[1], s[2]) < 0.01:
            key = satellite_keys[0]
            normy_norm = norm(key, x, y, z, input_satellites)
            t_s = input_satellites[key][0]
            t_v = t_s + normy_norm / c
            log.write(f" vehicle at: {x} {y} {z} after {num_iterations} iterations\n\n")
            return t_v, x, y, z

# Read string piped in from satellite program.
list_of_satellites = sys.stdin.readlines()

# Number of satellites read in
length_of_satellites = len(list_of_satellites)

# One list to rule them all.
# - Create list to contain all satellite data to be organized.
list_of_lists = []
for i in range(length_of_satellites):
    list_of_lists.append(list_of_satellites[i].split())#splits each line of list_satellites into a list of space delimited values eg. [satellite #, time, x, y, z]

# Organize list by time of satellites in increasing order.
sorting_hat = sorted(list_of_lists, key=lambda x: float(x[1]))

# Initialize dictionary that will contain satellites from each epoch.
input_satellites = {}

#build input_satellites
#first epoch time value used for comparison on each epoch
epoch_time = float(sorting_hat[0][1]) # first time value.

# epoch guess.
# - Used as first guess for each epoch. Namely position of b12 at similar time.
x,y,z= geo_to_car(epoch_time,40, 45, 55.0, 1, 111, 50, 58.0, -1, 1372.0)#b12 coordinates + the custom time to account for Earth's rotation.
initial_guess = [x,y,z]

# Initialize for counting number of epochs.
number_epochs = 0

#################
## COMPUTATION ##
#################

# Iterate through list of satellites that have been organized by time.
for sat_info in sorting_hat:

    # Check if last satellite in sorting hat, and if not then check if in same epoch as previous satellite.
    # i.e if within 1/2 second of previous time.
    if  (sat_info != sorting_hat[-1]) and (abs(epoch_time - float(sat_info[1])) < 0.5):
        i = sat_info[0]
        t = float(sat_info[1])
        x = float(sat_info[2])
        y = float(sat_info[3])
        z = float(sat_info[4])
        input_satellites[i] = [t, x, y, z]

    # If NOT within 1/2 second of previous satellite begin iteration function and start new epoch.
    else:
        # Try normal iteration.
        try:
            log.write(f" Epoch {number_epochs} -- Satellite Data:\n")
            number_epochs = number_epochs + 1

            t_v,x_v,y_v,z_v =iteration_step(input_satellites,initial_guess)#error should occur here if iteration_step fails then our except block catches
            t_forprinting, psi_degree, psi_minute, psi_second, NS, lambda_degree, lambda_minute, lambda_second, EW, h=car_to_geo(x_v,y_v,z_v,t_v)
            print(round(t_forprinting,2), round(psi_degree,2), round(psi_minute,2), round(psi_second,2), round(NS,2), round(lambda_degree,2), round(lambda_minute,2), round(lambda_second,2), round(EW,2), round(h,2))
            log.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(round(t_forprinting,2), round(psi_degree,2), round(psi_minute,2), round(psi_second,2), round(NS,2), round(lambda_degree,2), round(lambda_minute,2), round(lambda_second,2), round(EW,2), round(h,2)))

            input_satellites = {}
            i = sat_info[0]
            t = float(sat_info[1])
            x = float(sat_info[2])
            y = float(sat_info[3])
            z = float(sat_info[4])
            input_satellites[i] = [t, x, y, z]

            initial_guess = [x_v,y_v,z_v] #update initial guess. This assumes x_v, y_v, z_v are at the right time.
            epoch_time = t

        # If error, give appropriate warning.
        except:
            print('no convergence in receiver')
            print(0,0,0,0,-1,0,0,0, -1, 0)

            log.write('no convergence in receiver\n')
            log.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(round(0, 2), round(0, 2),
                                                                 round(0, 2), round(0, 2),
                                                                 round(-1, 2), round(0, 2),
                                                                 round(0, 2), round(0, 2),
                                                                 round(-1, 2), round(0, 2)))
            break
log.close()