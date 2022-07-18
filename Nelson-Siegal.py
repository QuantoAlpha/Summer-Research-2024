import numpy as np
from scipy.optimize import least_squares

maturity = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25]
# swap_rate_payment with face 100, each correspond to swap rate of each maturity
swap_rate_payment = [x/2 for x in [4.20, 4.30, 4.70, 5.40, 5.70, 6.00, 6.10, 5.90, 5.60, 5.55]]

"""
scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=(- inf, inf), 
                             method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, 
                             x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, 
                             tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, 
                             verbose=0, args=(), kwargs={})
"""

def price_swap(a, b, c, lamda, t, swap_rate_payment_list):
    index = maturity.index(t)
    swap_rate_payment = swap_rate_payment_list[index]
    price = 0
    for i in range(1, t*2+1):
        t_t = float(i)/2.0
        discount_factor = find_discount_factor(a, b, c, lamda, t_t)
        price += swap_rate_payment * discount_factor
    price += find_discount_factor(a, b, c, lamda, t) * 100
    return price

def find_discount_factor(a, b, c, lamda, t):
    interest_rate = r_hat(a, b, c, lamda, t)
    return np.exp(-interest_rate*t)

def r_hat(a, b, c, lamda, t):
    return a + b*(1-np.exp(-lamda*t))/(lamda*t) + c*((1-np.exp(-lamda*t))/(lamda*t)-np.exp(-lamda*t))

def sum_of_square(x):
    a, b, c, lamda = x[0], x[1], x[2], x[3]
    total = 0
    for index in range(len(maturity)):
        t = maturity[index]
        #print(f'Price at maturity {maturity[index]}: {price_swap(a, b, c, lamda, t, swap_rate_payment)}')
        total += np.square(price_swap(a, b, c, lamda, t, swap_rate_payment) - 100)
    return total

val = np.array([0.15, 0.15, 0.15, 0.15])

#res = least_squares(sum_of_square, val)

# Nelson-Siegal.py
cost_dict = dict()
for i in np.arange(0.14, 0.16, 0.001):
    val = np.array([i]*4)
    res = least_squares(sum_of_square, val)
    cost_dict[i] = res

cost = []
for key in cost_dict.keys():
    cost.append(cost_dict[key].cost)
answers = np.array(cost)
index = np.argmin(answers)
min_key = list(cost_dict.keys())[index]
min_res = cost_dict[min_key]

print("a:", min_res.x[0])
print("b:", min_res.x[1])
print("c:", min_res.x[2])
print("lamda:", min_res.x[3])
print(min_res.cost)
#print(res.optimality)


    