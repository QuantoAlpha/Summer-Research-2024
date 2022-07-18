import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class dts_ZCB():
    def __init__(self):
        self.a = 0.01815544
        self.b = 0.019172417
        self.c = 0.106554005
        self.lam = 0.122863157
        self.sigma = 0.007518843 
        self.lambda0 = 0
        self.r0 = 0.03

    def calc_lambda_in_ho_lee(self,t):
        return -self.b*self.lam*np.exp(self.lam*(-t))-self.c*self.lam**2*t*np.exp(self.lam*(-t))+self.c*self.lam*np.exp(self.lam*(-t))+t*self.sigma**2+self.lambda0

    # Find r_t interest rate
    def calc_lambda(self, t):
        return self.calc_lambda_in_ho_lee(t)

    def calc_rt(self,path,n):
        M = 2 * np.sum(path) - len(path)
        result = np.arange(0,len(path),1, dtype = int)
        # result = result/n
        result = (np.vectorize(self.calc_lambda, otypes=[float]))(result/n)
        result = result/n
        r = self.r0 + np.sum(result) + M * self.sigma/np.sqrt(n)
        return r

    # Find lambda between time t to T
    def find_lambda(self, t_scaled, T_scaled, n):
        time_interval = np.arange(t_scaled, T_scaled)
        def find_lambda(self, i, n):
            return self.calc_lambda_in_ho_lee(self.a,self.b,self.c,self.lam,i/n, self.sigma)
        #calc_lambda_in_ho_lee(a,b,c,lam, time_interval,sigma)
        return (np.vectorize(find_lambda, otypes=[float]))(time_interval)
    
    # Generate a random binomial path before time t
    def generate_path(self, t, n):
        return np.random.randint(2,size = t*n)
        # return np.array([0,1] * (t*n//2))

    def Pricing(self, t, T, N, num_of_monte_carlo):
        n = int(1/(T/N))
        path = self.generate_path(t,n)
        rt = self.calc_rt(path, n)
        print(f"short rate (dts): {rt}")
        result = np.array([1/(1+rt/n)]*num_of_monte_carlo)
        for num in range(num_of_monte_carlo):
            r = rt
            # Random walk
            flip = np.random.randint(2, size=T*n-t*n)
            """
            # all time from t*n to T*n, minus t*n when necessary
            time_interval = np.arange(t*n, T*n)
            # lambda values from t*n to T*n, every period
            lambda_list = self.find_lambda(t*n, T*n)
            # find discount factor based on lambda_list and flip
            find_discount = lambda i : 1/(1+(rt + 1/n*np.sum(lambda_list[:i-t*n] + self.sigma/np.sqrt(n)*np.sum(flip[:i-t*n+1])))/n)
            result = find_discount(time_interval)
            """
            for i in range(t*n+1, T*n):
                if flip[i-t*n-1] == 1:
                    r += self.calc_lambda_in_ho_lee(i/n-1)/n + self.sigma/np.sqrt(n)
                else:
                    r += self.calc_lambda_in_ho_lee(i/n-1)/n - self.sigma/np.sqrt(n)
                result[num] *= 1/(1+r/n)
        return np.mean(result), np.std(result), result


class cts_ZCB():
    def __init__(self, a, b, c, lam, sigma):
        self.a = a
        self.b = b
        self.c = c
        self.lam = lam
        self.sigma = sigma
        self.lamda0 = 0

    def r_hat(self, t):
        '''
        return r_hat given t, where r_hat is the spot rate
        '''
        self.r0 = self.a + self.b*(1-np.exp(-self.lam*t))/(self.lam*t) + self.c*((1-np.exp(-self.lam*t))/(self.lam*t)-np.exp(-self.lam*t))

    def calc_lambda_in_ho_lee(self, t):
        '''
        calcluate lambda in ho lee model
        '''
        return -self.b*self.lam*np.exp(self.lam*(-t))-self.c*self.lam**2*t*np.exp(self.lam*(-t))+self.c*self.lam*np.exp(self.lam*(-t))+t*self.sigma**2+self.lamda0

    def ho_lee_short_rate(self, t, N, M, T):
        '''
        calculate short rate under ho lee
        '''
        if t == 0:
            return self.r0
        sims = np.zeros(M)
        dt = T/N
        for i in range(M):
            W = [0]+np.random.standard_normal(size=N)*np.sqrt(dt)
            sims[i] = np.sum(W)
        integral = integrate.quad(lambda x: self.calc_lambda_in_ho_lee(x), 0, t)
        #return self.r0 + integral[0] + self.sigma * np.mean(sims)
        r = self.r0 + integral[0] + self.sigma * sims
        return r

    def ZCB_price_holee_helper(self, u, T):
        '''
        helper
        '''
        #print("T-u", T-u)
        #print("lamda", calc_lambda_in_ho_lee(a,b,c,lam,u,sigma))
        return (T-u)*self.calc_lambda_in_ho_lee(u)

    def Pricing(self, t, T, N, M):
        '''
        return the zcb price bought at time t, maturing at T
        N number of periods
        M number of paths
        '''
        #initialize r0
        #self.r_hat(0.0001)
        self.r0 = 0.03
        #calculate rt using monte carlo simulation
        #rt = self.ho_lee_short_rate(t, 1000, 10)
        #tail1 = -(T-t)*rt
        
        
        rt_lis = self.ho_lee_short_rate(t, N, M, T)
        print(f"short rate (cts): {rt_lis[t]}")
        tail1_lis = -(T-t)*rt_lis
        
        #integrate lambda function
        tail2 = -integrate.quad(lambda x: self.ZCB_price_holee_helper(x,T=T), t, T)[0]

        #last constant turn regarding sigma
        tail3 = self.sigma**2/6*(T-t)**3
        
        price_lis = np.exp(tail1_lis+tail2+tail3)
        return np.mean(price_lis), np.std(price_lis), price_lis

def debugging():
    a = 0.01815544
    b = 0.019172417
    c = 0.106554005
    lam = 0.122863157
    sigma = 0.007518843 

    cts_agent = cts_ZCB(a, b, c, lam, sigma)
    dts_agent = dts_ZCB()

    #Run N number of times, select t, T, plot the distribution of prices, 
    #distribution of standard deviation 

    N = 1
    t = 1
    T = 10
    num_periods = 100
    num_paths = 100
    cts_price, cts_std, dts_price, dts_std = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    x = np.arange(0,num_paths,1,dtype = int)
    y_cts = cts_agent.Pricing(t, T, num_periods, num_paths)[2]
    y_dts = dts_agent.Pricing(t, T, num_periods, num_paths)[2]
    plt.plot(x, y_cts, label = "continuous")
    plt.plot(x, y_dts, label = "discrete")
    plt.xlabel("Path Index")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    # for i in range(N):
    #     print(i)
    #     cts_p, cts_s = cts_agent.Pricing(t, T, num_periods, num_paths)
    #     dts_p, dts_s = dts_agent.Pricing(t, T, num_periods, num_paths)

    #     cts_price[i] = (cts_p)
    #     cts_std[i] = (cts_s)
    #     dts_price[i] = (dts_p)
    #     dts_std[i] = (dts_s)
    
    # plt.hist(cts_price, "cts_price")
    # plt.hist(cts_std, "cts_std")
    # plt.hist(dts_price, "dts_price")
    # plt.hist(dts_std, "dts_std")
    # plt.show()
    
debugging()