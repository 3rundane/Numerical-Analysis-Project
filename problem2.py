from math import sqrt

def norm(vec):
    square = [i**2 for i in vec]
    square_sum = sum(square)
    size = sqrt(square_sum)
    return size