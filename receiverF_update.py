import numpy as np
import sys
from decimal import *
from math import sqrt, atan, cos, sin, asin
import numpy.linalg as la


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


# Grab useful data
satellite_data, constants_names = grab_datafile()

# Define parameters from data.dat
R = float(constants_names["2"])
ss = float(constants_names["3"])
p = float(ss) / 2
c = float(constants_names["1"])
pi = float(constants_names["0"])



# input_satellites = {}
# input_satellites is a dictionary containing the time, x,y,z information of the satellites sent in by satellite.py

##input satellites needs to be changed later, below is the testing version!
# input_satellites = {"0":[0,2,1,1],"1":[0,1,1,1],"2":[0,2,1,1],"2b":[0,1,1,1],"3": [12122.917273538935, 2.605234313778725E7, 2986153.9652697924, 4264669.833325115],
#                     "4": [12122.918115974104, -1.718355633086311E7, -1.8640834276186436E7, 7941901.319733662]
#                     ,"test0":[0,0,0,0],"test1":[0,1,1,1],"testX_current":[0,2,1,1],"testX_sub":[0,1,1,1],"testY_current":[0,1,2,1],"testY_sub":[0,1,1,1],"testZ_current":[0,1,1,2],"testZ_sub":[0,1,1,1]}
# This is testing version without unittests.
# input_satellites = {"3": [12122.917273538935, 2.605234313778725E7, 2986153.9652697924, 4264669.833325115],
#                     "4": [12122.918115974104, -1.718355633086311E7, -1.8640834276186436E7, 7941901.319733662],
#                     '8': [12122.91517247339, 1.8498279256616846E7, -1.4172390064384513E7, -1.2758766855293432E7],
#                     '11': [12122.929474004011, -2903225.4285143306, -1.9661358537802488E7, 1.7630410370147068E7],
#                     '14': [12122.93081680465, 1477645.012869009, -1.5214872308462147E7, 2.172908956016601E7],
#                     '15': [12122.91559232703, 2.6526323652830362E7, 847508.5779779141, -1210367.686336006],
#                     '17': [12122.932126735379, 4939777.113795485, -1.796566328317718E7, 1.893839095916287E7],
#                     '20': [12122.9302901758, 1.790346111594521E7, -1.680512822049418E7, 1.0143118495964047E7]
#                     }
# sat_length is the length of the input_satellites list, used for a for loop iteration below.
# sat_length = len(input_satellites)


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


###########################
# DANE I DID SOMETHING HERE#
###########################

# changed t_current - t_subsequent --> Decimal(t_current - t_subsequent)


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


# satellite_keys = input_satellites.keys()
# satellite_keys = [i for i in input_satellites.keys()]

##largest index in satellite_keys, meant to be used in f below as well as some other stuff below.
# satty_length = len(satellite_keys)


## equations (69)
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


###################
## X should be Y ##
###################
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


###################
## X should be Z ##
###################

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


##Hesssssiaaaaannnnnnn timmmeeeeee
##equations (71)
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


##partial


##equations(70)

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


#############################
## This is recursive still ##
#############################

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


##################
### Seems okay ###
##################

##begin processing with Newton's method to find x_s
# the negative of the gradient of f
def B(index, x_1, x_2, x_3, input_satellites):
    f_x = f_partial_x(index, x_1, x_2, x_3, input_satellites)
    f_y = f_partial_y(index, x_1, x_2, x_3, input_satellites)
    f_z = f_partial_z(index, x_1, x_2, x_3, input_satellites)
    b_ret = [-1 * f_x, -1 * f_y, -1 * f_z]
    return b_ret


##################
### Seems okay ###
##################

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


##################
## Didn't check ##
##################

# # To convert geo-degrees to radians
def deg_to_rad(deg, min, sec):
    rad = pi * (deg + min / 60 + sec / 3600) / 180
    return rad


##################
## Didn't check ##
##################

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


# Maybe use peters function somehow
def rad_to_deg(rad):
    total_seconds = int(round(rad * 180 / pi * 3600))
    sec = total_seconds % 60
    min = (total_seconds // 60) % 60
    deg = total_seconds // 3600
    return deg, min, sec


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


# D = R + 1000

#######################
## Page 2 lecture 10 ##
#######################

# starting guess in cartesian coordinates x,y,z #Peter said starting guess for the first iteration of our for loop
# should be the coordinates of the lampost b12, the subsequent initial guesses as the program goes along should be...
def iteration_step(input_satellites):
    satellite_keys = [i for i in input_satellites.keys()]
    satty_length = len(satellite_keys)

    x_0, y_0, z_0 = geo_to_car(12122.917273538935, 40, 45, 55.0, 1, 111, 50, 58.0, -1, 1372.0)
    print(geo_to_car(12123.0, 40, 45, 55.0, 1, 111, 50, 58.0, -1, 1372.0))
    x, y, z = float(x_0), float(y_0), float(z_0)
    condition = True

    while condition:
        hessian_row1, hessian_row2, hessian_row3 = Hessian_rows(satty_length - 1, x, y, z, input_satellites)

        B_vector = B(satty_length - 1, x, y, z, input_satellites)
        H = np.array([hessian_row1, hessian_row2, hessian_row3])
        s = np.linalg.solve(H, B_vector)
        x = x + s[0]
        y = y + s[1]
        z = z + s[2]
        print(x, y, z)
        print('S:', normal_norm(s[0], s[1], s[2]), '\n')
        if normal_norm(s[0], s[1], s[2]) < 0.01:
            key = satellite_keys[0]
            normy_norm = norm(key, x, y, z, input_satellites)
            t_s = input_satellites[key][0]
            t_v = t_s + normy_norm / c
            # print(t_v)
            t, psi_degree, psi_minute, psi_second, NS, lambda_degree, lambda_minute, lambda_second, EW, h = car_to_geo(
                x, y, z, t_v)
            # print(car_to_geo(x,y,z,t_v))
            break

# iteration_step(input_satellites)
# t_v = t_s + ||X_s - X_v|| / c

# process piped in data
list_of_satellites = sys.stdin.readlines()
length_of_satellites = len(list_of_satellites)
sorting_hat = {}
# one list to rule them all
list_of_lists = []
for i in range(length_of_satellites):
    list_of_lists.append(list_of_satellites[i].split())
    sorting_hat[i] = float(list_of_lists[i][1])
# dictionary to hold shit
sorting_hat = dict(sorted(sorting_hat.items(), key=lambda item: item[1]))

print(sorting_hat)