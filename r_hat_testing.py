import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

a = 0.01815544
b = 0.019172417
c = 0.106554005
lam = 0.122863157
sigma = 0.007518843

def r_hat(a, b, c, lamda, t):
    return a + b*(1-np.exp(-lamda*t))/(lamda*t) + c*((1-np.exp(-lamda*t))/(lamda*t)-np.exp(-lamda*t))

r0 = r_hat(a, b, c, lam, 0.01)
lamda0 = 0
print("r0", r0)

def calc_lambda_in_ho_lee(a,b,c,l,t,sigma):
    return -b*l*np.exp(l*(-t))-c*l**2*t*np.exp(l*(-t))+c*l*np.exp(l*(-t))+t*sigma**2+lamda0

def double(lam, t):
    return integrate.dblquad(lambda x, y: calc_lambda_in_ho_lee(a, b, c, lam, x, sigma), 
                             0, t, lambda x: 0, lambda x: x)[0]

def r_hat_testing(t, lam, sigma):
    lamda = double(lam, t)
    sigma = sigma
    return (r0*t + lamda - (sigma**2)*(t**3)/6)/t

x_test = np.arange(1, 25)
y_test = []
for t in x_test:
    y_test.append(calc_lambda_in_ho_lee(a,b,c,lam,t,sigma))
plt.plot(x_test, y_test, label = "r_test")
x_nelson = np.arange(1, 25)
y_nelson = []
for i in x_nelson:
    y_nelson.append(r_hat(a, b, c, lam, i))
plt.plot(x_nelson, y_nelson, label = "r_nelson")
plt.legend()
plt.xlabel("time")
plt.ylabel("rate")
plt.show()
