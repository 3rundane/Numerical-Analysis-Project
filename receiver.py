import numpy as np
import sys
from decimal import *
from math import cos,sin,sqrt

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

#input_satellites is a dictionary containing the time, x,y,z information of the satellites sent in by satellite.py
input_satellites = {}
#sat_length is the length of the input_satellites list, used for a for loop iteration below.
sat_length = len(input_satellites)

#i_s is the key of a given satellite whose norm of the difference between it and the current-iteration vehicle position is given.
#x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
#s_1,s_2,s_3 are the x,y,z components of the satellite at time t_s.
# norm will be used in each iteration through being called in A below, as well as other places in the Newton's method iteration.
def norm(i_s,x_1,x_2,x_3):
    s_1 = input_satellites[i_s][1]
    s_2 = input_satellites[i_s][2]
    s_3 = input_satellites[i_s][3]
    return sqrt((s_1-x_1)**2 + (s_2-x_2)**2 + (s_3-x_3)**2)



#difference equations are coded below. This function returns the ith output of the A_i in Peter's equation 68. This will be called at each
#step in the Newton's method iteration.
#i_current and i_subsequent are the ith and (i + 1)th satellite keys to be used for computing the ith and (i+1)th norm.
#t_current and t_subsequent are the t_s's of the ith and (i+1)th satellites.
##x_1,x_2,x_3 are the x,y,z components of the vehicle vector on the kth iteration.
def A(i_current,i_subsequent,x_1,x_2,x_3):
      current_norm = norm(input_satellites[i_current],x_1,x_2,x_3)
      subsequent_norm = norm(input_satellites[i_subsequent,x_1,x_2,x_3])
      t_current = input_satellites[i_current][0]
      t_subsequent = input_satellites[i_subsequent][0]
      return subsequent_norm - current_norm - c*(t_current-t_subsequent)