from math import sqrt, atan, cos, sin, asin, pi

# from math import *
# We should be careful importing all, because that imports pi value which may differ from the data.dat pi
R = 6.367444500000000000E+06
# pi = 1
s = 8.616408999999999651E+04


# Maybe use peters function somehow
def rad_to_deg(rad):
    total_seconds = int(round(rad * 180 / pi * 3600))
    sec = total_seconds % 60
    min = (total_seconds // 60) % 60
    deg = total_seconds // 3600
    return deg, min, sec


def cart_to_geo(xt, yt, zt, t):
    # Un-rotate for t seconds.
    alpha = 2 * pi / s
    x = xt * cos(alpha * t) + yt * sin(alpha * t)
    y = -xt * sin(alpha * t) + yt * cos(alpha * t)
    z = zt
    # print(x,y)

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


D = R + 1000

print(cart_to_geo(0, D, D, 1800))
print(cart_to_geo(0, -D, D, 1800))
# print(cart_to_geo(D, D, -D, 1800))
# print(cart_to_geo(-D, -D, -D, 1800))
# print(cart_to_geo(D, -D, D, 1800))
# print(cart_to_geo(-D, D, -D, 1800))
# print(cart_to_geo(D, D, D, 1800))
