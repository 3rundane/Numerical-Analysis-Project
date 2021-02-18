from math import sin

#My first program to test how many steps it takes to get epsilon distance from 0
i = 0
x = 1
while True:
    i += 1
    x = sin(x)
    if x < 0.001:
        break
print(x, i)

#Code to look at how at effects of rounding errors.
i = 0
x = 1
n = 10000
rounding = 5

for step in range(n):
    i += 1
    x = round(sin(x), rounding)
    print(f'{step+1}: {x}')

