import math
import numpy as np
from sympy.interactive import printing
printing.init_printing(use_latex=True)
from sympy import Eq, solve_linear_system, Matrix
from numpy import linalg
import sympy as sp
from scipy.optimize import fsolve
#import matplotlib as plt
import matplotlib.pyplot as plt
class bond():
    def __init__(self, maturity, swap):
        self.maturity = maturity
        self.price = 100
        self.face = 100
        self.coupon = swap * self.face/2 #coupon amount 
        

bond_1 = bond(1, 0.042)
bond_2 = bond(2, 0.043)
bond_3 = bond(3, 0.047)
bond_5 = bond(5, 0.054)
bond_7 = bond(7, 0.057)
bond_10 = bond(10, 0.06)
bond_12 = bond(12, 0.061)
bond_15 = bond(15, 0.059)
bond_20 = bond(20, 0.056)
bond_25 = bond(25, 0.0555)
maturity = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25]
bonds = [bond_1, bond_2, bond_3, bond_5, bond_7, bond_10, bond_12, bond_15, bond_20, bond_25]

bond_dict = dict() #dict {maturity (int): bond (obj)}
for i in range(len(maturity)):
    bond_dict[maturity[i]] = bonds[i]


def find_exp(t: int)->list:
    """
    Brief: find the exponent for each term
    Input: t - a float
    Return: return a list of length 8, each element of type float
    """
    if t <= 1.0:
        return [t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 2.0:
        return [1.0, t-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 3.0:
        return [1.0, 1.0, t-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 5.0:
        return [1.0, 1.0, 1.0, t-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 7.0:
        return [1.0, 1.0, 1.0, 2.0, t-5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 10.0:
        return [1.0, 1.0, 1.0, 2.0, 2.0, t-7.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 12.0:
        return [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, t-10.0, 0.0, 0.0, 0.0]
    elif t <= 15.0:
        return [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, t-12.0, 0.0, 0.0]
    elif t <= 20.0:
        return [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0, t-15.0, 0.0]
    elif t <= 25.0:
        return[1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 5.0, t-20.0]
    else:
        print("NOOOOOOO!!!")

forward_rates = np.array([0.041565132, 0.043566033, 0.054853221, 0.065057774, 0.065263571, 0.068543393, 0.067401102, 0.045493446, 0.039695413, 0.050885901])

x_axis = [1.0,2.0,3.0,5.0,7.0,10.0,12.0,15.0,20.0,25.0]
y_axis = []

for x in x_axis:
    result = float(1/x)*float(np.dot(np.array(find_exp(x)), forward_rates))
    y_axis.append(result)

y_axis = np.array(y_axis) * 100

print(y_axis)

plt.plot(x_axis, y_axis, label = "Yield Curve")
plt.xlabel("Time")
plt.ylabel("Yield(%)")
plt.title("Yield Curve")
plt.show()

#x_axis_crazy = np.arange(0,25,0.01)
#y_axis_crazy = 