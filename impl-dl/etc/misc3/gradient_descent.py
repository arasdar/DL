from sympy import *

x, y = symbols('x y')

# Objective Function
f = x*y + 2*(62.5/x) + 2*(62.5/y)

# Differentiating - computing the gradient.
fpx = f.diff(x)
fpy = f.diff(y)
grad = [fpx,fpy]

# Starting point
theta0 = 20
theta1 = 20

# Algorithm parameters
alpha = 0.01
epsilon = 0.00000001

iterations = 0
maxIterations = 1000
printData = True
check = 0

while True:
    # Simultaneously update unknown variables
    tempTheta0 = theta0 - alpha * N(fpx.subs(x, theta0).subs(y, theta1))
    tempTheta1 = theta1 - alpha * N(fpy.subs(y, theta1).subs(x, theta0))

    iterations += 1
    if iterations > maxIterations:
        print("Too many iterations. Adjust alpha.")
        printData = False
        break

    if abs(tempTheta0 - theta0) < epsilon and abs(tempTheta1 - theta1) < epsilon:
        break

    theta0 = tempTheta0
    theta1 = tempTheta1

z = 62.5/(theta0*theta1)

if printData:
    print("x = ", theta0, sep = " ")
    print("y = ", theta1, sep = " ")
    print("z = ", z, sep = " ")