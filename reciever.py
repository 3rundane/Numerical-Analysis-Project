import numpy as np
import sys
from decimal import *
from math import cos, sin, sqrt
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


satellite_data, constants_names = grab_datafile()

R = Decimal(constants_names["2"])
s = Decimal(constants_names["3"])
p = s / Decimal(2)
c = Decimal(constants_names["1"])
pi = Decimal(constants_names["0"])

# File_object = open(r"b12.dat", "r")
# vehicle_dat_raw = File_object.readlines()
# input_satellites is a dictionary containing the time, x,y,z information of the satellites sent in by satellite.py

##input satellites needs to be changed later, below is the testing version!
input_satellites = {"3": [12122.917273538935, 2.605234313778725E7, 2986153.9652697924, 4264669.833325115],
                    "4": [12122.918115974104, -1.718355633086311E7, -1.8640834276186436E7, 7941901.319733662]
                    }

# sat_length is the length of the input_satellites list, used for a for loop iteration below.
sat_length = len(input_satellites)


### equations (68)
# i_s is the key of a given satellite whose norm of the difference between it and the current-iteration vehicle position is given.
# x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
# s_1,s_2,s_3 are the x,y,z components of the satellite at time t_s.
# norm will be used in each iteration through being called in A below, as well as other places in the Newton's method iteration.
def norm(i_s, x_1, x_2, x_3):
    s_1 = input_satellites[i_s][1]
    s_2 = input_satellites[i_s][2]
    s_3 = input_satellites[i_s][3]
    return sqrt((s_1 - x_1) ** 2 + (s_2 - x_2) ** 2 + (s_3 - x_3) ** 2)


##this is the norm to be used in Newton' method iteration, NOT in the function definitions below.
def normal_norm(s_1, s_2, s_3):
    return sqrt(s_1 ** 2 + s_2 ** 2 + s_3 ** 2)

###########################
#DANE I DID SOMETHING HERE#
###########################

# changed t_current - t_subsequent --> Decimal(t_current - t_subsequent)


# difference equations are coded below. This function returns the ith output of the A_i in Peter's equation 68. This will be called at each
# step in the Newton's method iteration.
# i_current and i_subsequent are the ith and (i + 1)th satellite keys to be used for computing the ith and (i+1)th norm.
# t_current and t_subsequent are the t_s's of the ith and (i+1)th satellites.
##x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
def A(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
    t_current = input_satellites[i_current][0]
    t_subsequent = input_satellites[i_subsequent][0]
    return subsequent_norm - current_norm - c * (t_current - t_subsequent)


# partial derivative wrt first coordinate of A or the difference equation.
def X(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]
    return -(s_1_subsequent - x_1) / subsequent_norm + (s_1_current - x_1) / current_norm


# partial derivative wrt second coordinate of A or the difference equation.
def Y(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]
    return -(s_2_subsequent - x_2) / subsequent_norm + (s_2_current - x_2) / current_norm


# partial derivative wrt third coordinate of A or the difference equation.
def Z(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]
    return -(s_3_subsequent - x_3) / subsequent_norm + (s_3_current - x_3) / current_norm


# satellite_keys = input_satellites.keys()
satellite_keys = [i for i in input_satellites.keys()]

print(satellite_keys)
##largest index in satellite_keys, meant to be used in f below as well as some other stuff below.
satty_length = len(satellite_keys)


## equations (69)
## Everything below until HESSSSSIIIaAAANN TIMMMEEE is for the right hand side of our Newton's method iteration.
def f(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        f_value = (A(i_current, i_subsequent, x_1, x_2, x_3)) ** 2 + f(index - 1, x_1, x_2, x_3)
    else:
        f_value = 0
    return f_value


##Partial derivatives of little f wrt x. Recursive definition.
def f_partial_x(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        f_partial_xvalue = A(i_current, i_subsequent, x_1, x_2, x_3) * X(i_current, i_subsequent, x_1, x_2,
                                                                         x_3) + f_partial_x(index - 1, x_1, x_2, x_3)
    else:
        f_partial_xvalue = 0
    return 2 * f_partial_xvalue


##Partial derivatives of little f wrt y. Recursive definition.
def f_partial_y(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        f_partial_yvalue = A(i_current, i_subsequent, x_1, x_2, x_3) * Y(i_current, i_subsequent, x_1, x_2,
                                                                         x_3) + f_partial_y(index - 1, x_1, x_2, x_3)
    else:
        f_partial_yvalue = 0
    return 2 * f_partial_yvalue


##Partial derivatives of little f wrt z. Recursive definition.
def f_partial_z(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        f_partial_zvalue = A(i_current, i_subsequent, x_1, x_2, x_3) * Z(i_current, i_subsequent, x_1, x_2,
                                                                         x_3) + f_partial_z(index - 1, x_1, x_2, x_3)
    else:
        f_partial_zvalue = 0
    return 2 * f_partial_zvalue


##Hesssssiaaaaannnnnnn timmmeeeeee
##equations (71)
##the order is the same as that in Peter's homework 1
##partial X partial x
def X_partial_x(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
    s_1_current = input_satellites[i_current][1]
    s_1_subsequent = input_satellites[i_subsequent][1]
    current_fraction = (current_norm ** 2 - (s_1_current - x_1) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_1_subsequent - x_1) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction


##partial Y partial x
def Y_partial_x(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(i_current, x_1, x_2, x_3)
    subsequent_norm = norm(i_subsequent, x_1, x_2, x_3)
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
def X_partial_y(i_current, i_subsequent, x_1, x_2, x_3):
    return Y_partial_x(i_current, i_subsequent, x_1, x_2, x_3)


##partial X partial z
def X_partial_z(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(input_satellites[i_current], x_1, x_2, x_3)
    subsequent_norm = norm(input_satellites[i_subsequent, x_1, x_2, x_3])
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
def Z_partial_x(i_current, i_subsequent, x_1, x_2, x_3):
    return X_partial_z(i_current, i_subsequent, x_1, x_2, x_3)


##partial Y partial y
def Y_partial_y(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(input_satellites[i_current], x_1, x_2, x_3)
    subsequent_norm = norm(input_satellites[i_subsequent, x_1, x_2, x_3])
    s_2_current = input_satellites[i_current][2]
    s_2_subsequent = input_satellites[i_subsequent][2]
    current_fraction = (current_norm ** 2 - (s_2_current - x_2) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_2_subsequent - x_2) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction


##partial Y partial z
def Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(input_satellites[i_current], x_1, x_2, x_3)
    subsequent_norm = norm(input_satellites[i_subsequent, x_1, x_2, x_3])
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
def Z_partial_y(i_current, i_subsequent, x_1, x_2, x_3):
    return Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3)


##partial Z partial z
def Z_partial_z(i_current, i_subsequent, x_1, x_2, x_3):
    current_norm = norm(input_satellites[i_current], x_1, x_2, x_3)
    subsequent_norm = norm(input_satellites[i_subsequent, x_1, x_2, x_3])
    s_3_current = input_satellites[i_current][3]
    s_3_subsequent = input_satellites[i_subsequent][3]
    current_fraction = (current_norm ** 2 - (s_3_current - x_3) ** 2) / (current_norm) ** 3
    subsequent_fraction = (subsequent_norm ** 2 - (s_3_subsequent - x_3) ** 2) / (subsequent_norm) ** 3
    return subsequent_fraction - current_fraction


##partial


##equations(70)

##partialf^2 partial x,x
def f_double_partial_xx(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3)
        X_partial_x_i = X_partial_x(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = X_i ** 2 + A_i * X_partial_x_i + f_double_partial_xx(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##partialf^2 partial x,y
def f_double_partial_xy(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3)
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3)
        X_partial_y_i = X_partial_y(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = X_i * Y_i + A_i * X_partial_y_i + f_double_partial_xy(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##partialf^2 partial y,x
def f_double_partial_yx(index, x_1, x_2, x_3):
    return f_double_partial_xy(index, x_1, x_2, x_3)


##partialf^2 partial xz
def f_double_partial_xz(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        X_i = X(i_current, i_subsequent, x_1, x_2, x_3)
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3)
        X_partial_z_i = X_partial_z(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = X_i * Z_i + A_i * X_partial_z_i + f_double_partial_xz(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##partial^2 partial zx
def f_double_partial_zx(index, x_1, x_2, x_3):
    return f_double_partial_xz(index, x_1, x_2, x_3)


##partialf^2 partial yy
def f_double_partial_yy(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3)
        Y_partial_y_i = Y_partial_y(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = Y_i ** 2 + A_i * Y_partial_y_i + f_double_partial_yy(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##partial^2 partial yz
def f_double_partial_yz(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Y_i = Y(i_current, i_subsequent, x_1, x_2, x_3)
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3)
        Y_partial_z_i = Y_partial_z(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = Y_i * Z_i + A_i * Y_partial_z_i + f_double_partial_yz(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##partial f^2 partial zy
def f_double_partial_zy(index, x_1, x_2, x_3):
    return f_double_partial_yz(index, x_1, x_2, x_3)


###partial^2 partial zz
def f_double_partial_zz(index, x_1, x_2, x_3):
    if index >= 1:
        i_subsequent = satellite_keys[index]
        i_current = satellite_keys[index - 1]
        Z_i = Z(i_current, i_subsequent, x_1, x_2, x_3)
        Z_partial_z_i = Z_partial_z(i_current, i_subsequent, x_1, x_2, x_3)
        A_i = A(i_current, i_subsequent, x_1, x_2, x_3)
        almost_final_value = Z_i ** 2 + A_i * Z_partial_z_i + f_double_partial_zz(index - 1, x_1, x_2, x_3)
    else:
        almost_final_value = 0
    return 2 * almost_final_value


##begin processing with Newton's method to find x_s
# the negative of the gradient of f
def B(index, x_1, x_2, x_3):
    f_x = f_partial_x(index, x_1, x_2, x_3)
    f_y = f_partial_y(index, x_1, x_2, x_3)
    f_z = f_partial_z(index, x_1, x_2, x_3)
    return -1 * [f_x, f_y, f_z]


# compute and return the hessian rows for a given input.
def Hessian_rows(index, x_1, x_2, x_3):
    f_xx = f_double_partial_xx(index, x_1, x_2, x_3)
    f_xy = f_double_partial_xy(index, x_1, x_2, x_3)
    f_yx = f_xy
    f_xz = f_double_partial_xz(index, x_1, x_2, x_3)
    f_zx = f_xz
    f_yz = f_double_partial_yz(index, x_1, x_2, x_3)
    f_zy = f_yz
    f_yy = f_double_partial_yy(index, x_1, x_2, x_3)
    f_zz = f_double_partial_zz(index, x_1, x_2, x_3)
    return [f_xx, f_xy, f_xz], [f_yx, f_yy, f_yz], [f_zx, f_zy, f_zz]


# starting guess in cartesian coordinates x,y,z
x_0, y_0, z_0 = 10000, 10000, 10000
x, y, z = x_0, y_0, z_0
condition = True

while condition:
    hessian_row1, hessian_row2, hessian_row3 = Hessian_rows(satty_length - 1, x, y, z)
    B_vector = np.array(B(satty_length - 1, x, y, z))
    H = np.array([hessian_row1, hessian_row2, hessian_row3])
    s = np.linalg.solve(H, B_vector)
    x = x + s[0]
    y = y + s[1]
    z = z + s[2]
    if normal_norm(s[0], s[1], s[2]) < 0.01:
        break

print(x, y, z)

##Progress comments
## error is likely in i_s being a list and not a valid key....see error below.
# most recent error message
##['3', '4']
# Traceback (most recent call last):
#   File "C:/Users/Dane/PycharmProjects/UCDavisCode/receiver.py", line 437, in <module>
#     hessian_row1,hessian_row2,hessian_row3 = Hessian_rows(satty_length-1,x,y,z)
#   File "C:/Users/Dane/PycharmProjects/UCDavisCode/receiver.py", line 420, in Hessian_rows
#     f_xx = f_double_partial_xx(index,x_1,x_2,x_3)
#   File "C:/Users/Dane/PycharmProjects/UCDavisCode/receiver.py", line 324, in f_double_partial_xx
#     X_i = X(i_current,i_subsequent,x_1,x_2,x_3)
#   File "C:/Users/Dane/PycharmProjects/UCDavisCode/receiver.py", line 154, in X
#     current_norm = norm(input_satellites[i_current], x_1, x_2, x_3)
#   File "C:/Users/Dane/PycharmProjects/UCDavisCode/receiver.py", line 130, in norm
#     s_1 = input_satellites[i_s][1]
# TypeError: unhashable type: 'list'
