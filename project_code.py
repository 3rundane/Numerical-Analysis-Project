import math
pi = math.pi

def deg_to_rad(deg, min, sec):
    rad = (deg + min / 60 + sec / 3600) / 180 * pi
    return rad

# Need to find a way to separate the whole number.
def rad_to_deg(rad):
    dummy = rad * 180 / pi
    deg = floor(dummy) # Just use the whole number
    min = floor((dummy - floor(dummy)) * 60) # Multiply the decimals by 60
    sec = floor(((dummy - floor(dummy)) * 60 - min) * 60)
    return deg, min, sec


    
