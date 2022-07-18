import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math


def calc_lambda_in_ho_lee(a,b,c,l,t,sigma):
    return -b*l*np.exp(l*(-t))-c*l**2*t*np.exp(l*(-t))+c*l*np.exp(l*(-t))+t*sigma**2+lamda0

a = 0.01815544
b = 0.019172417
c = 0.106554005
lam = 0.122863157
sigma = 0.007518843
lambda0 = 0

p = 1/2
q = 1/2

t = 2
T = 6
n = 800 # 1/(T/N)
r0 = 0
# path = np.random.randint(2, size=t*n)
path = np.array([0,1] * (t*n//2))
# print(f'path = {path}')

a = 0.01815544
b = 0.019172417
c = 0.106554005
l = 0.122863157
sigma = 0.007518843
lambda0 = 0

def calc_lambda_in_ho_lee(a,b,c,l,t,sigma):
    return -b*l*math.exp(l*(-t))-c*l**2*t*math.exp(l*(-t))+c*l*math.exp(l*(-t))+t*sigma**2+lambda0

# Find r_t interest rate
calc_lambda = lambda t: calc_lambda_in_ho_lee(a,b,c,l,t,sigma)
vfunc = np.vectorize(calc_lambda)

def calc_rt(path):
    M = 2 * np.sum(path) - len(path)
    result = np.arange(0,len(path),1, dtype = int)
    # result = result/n
    result = vfunc(result/n)
    result = result/n
    r = r0 + np.sum(result) + M * sigma/np.sqrt(n)
    return r

print("Short rate:", calc_rt(path))


# Find lambda between time t to T
def find_lambda(t_scaled, T_scaled):
    time_interval = np.arange(t_scaled, T_scaled, 1)
    find_lambda = np.vectorize(lambda i: calc_lambda_in_ho_lee(a,b,c,lam,i/n, sigma))
    return find_lambda(time_interval)

def calc_dT(rt, num_of_monte_carlo):
    result = np.array([1/(1+rt/n)]*num_of_monte_carlo)
    for num in range(num_of_monte_carlo):
        r = rt
        # Random walk t=1 T=2 n=2
        flip = np.random.randint(2, size=T*n-t*n)

        """
        # all time from t*n to T*n, minus t*n
        time_interval = np.arange(t*n, T*n, 1)
        # lambda values from t*n to T*n, every period
        lambda_list = find_lambda(t*n, T*n)
        # find discount factor based on lambda_list and flip
        M = lambda i : 2 * np.sum(flip[:i-t*n+1]) - (i-t*n+1)
        find_discount = np.vectorize(lambda i : 1/(1+(rt + 1/n*np.sum(lambda_list[:i-t*n]) + sigma/np.sqrt(n)*M(i))/n))
        result[num] = result[num]*np.prod(find_discount(time_interval))
        """
        
        for i in range(t*n+1, T*n):
            if flip[i-t*n-1] == 1:
                r += calc_lambda_in_ho_lee(a,b,c,lam,i/n-1, sigma)/n + sigma/np.sqrt(n)
            else:
                r += calc_lambda_in_ho_lee(a,b,c,lam,i/n-1, sigma)/n - sigma/np.sqrt(n)
            result[num] *= 1/(1+r/n)
    return np.mean(result), np.std(result)

mean, std = calc_dT(calc_rt(path), 100)
print(f'price: {mean}, std: {std}')



'''
x_test = np.arange(1, 25, 0.1)
y_test = []
for t in x_test:
    y_test.append(calc_lambda_in_ho_lee(a,b,c,lam,t,sigma))
plt.plot(x_test, y_test, label = "r_test")
plt.show()
'''
