import math
import numpy as np
import datetime

#-----request data
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

def T_bills_price(maturity, price):
    #Input maturity as number of years, 1/4 for 3 month
    if maturity <= 1:
        return 100-(price*100*maturity)
    else: 
        return price

class bond():
    def __init__(self, maturity, price, maturity_date, coupon, face):
        self.maturity = maturity
        self.price = T_bills_price(self.maturity, price)
        # today from issue date, in years
        self.days = self.days_between(self.issue_date(maturity_date, maturity), maturity_date)
        self.coupon = coupon
        self.face = face
        #update
        self.link_dict = {0.25: '3M', 0.5: '6M', 1: '1Y', 2: '2Y', 5: '5Y', 7: '7Y', 10: '10Y', 20: '20Y', 30: '30Y'}
    
    def get_issue_day(self, maturity_date, maturity):
        """
        Get the issue date from maturity_date
        """
        return maturity_date.replace(year=maturity_date.year-maturity)

    def get_days_between(self, issue_date, maturity_date):
        """
        Get the difference between issue date and current date

        Return number of years in between
        """
        today_date = datetime.date.today()
        diff = (today_date - issue_date)
        return diff.month*30 + diff.day
    
    def update(self):

        link = 'https://www.cnbc.com/quotes/US' + self.link_dict[self.maturity]

        html = urlopen(link)
        bs = BeautifulSoup(html.read(), 'html.parser')

        lis = bs.findAll(name = 'li',  attrs={"class" :"Summary-stat"})

        for link in lis:
            txt = link.get_text()
            price = re.match('Price\d+', txt)
            
            if price:#update price
                self.price = float(txt[len('Price'):])
                if self.maturity <= 1:
                    self.price = T_bills_price(self.maturity, self.price)
                
            if txt.startswith('Maturity'): #update maturity&days
                self.maturity_date= txt[len('Maturity'):]
                issue = self.get_issue_day(self.maturity_date, self.maturity)
                self.days = self.get_days_between(self, issue, self.maturity_date)
                
            if txt.startswith("Coupon"): #update coupon
                self.coupon = float(txt[len('Coupon'):-1])/100 #convert to decimals

bond_03 = bond(1/4, 1.1075, "2029-06-30", 0.0, 100)
bond_05 = bond(1/2, )
bond_1 = bond(1, )
bond_2 = bond(2)
bond_5 = bond(5, 1, "")
bond_7 = bond(7, 101.0938, "2029-06-30", 0.0325, 100)
bond_10 = bond(10, 98.9062, "2032-05-15", 0.02875, 100)
bond_20 = bond(20, 97.9219, '2042-05-15', 0.0325, 100)
bond_30 = bond(30, 95.0312, '2052-05-15', 0.02875 ,100)

#----we can always update our bond info by calling bond.update


Bond_maturity = [0.5, 1, 2, 5, 7, 10, 20, 30]
Bond_price = {0.5: 0, 1: 0, 2: 0, 5: 0, 7: 0, 10: 0, 20: 0, 30: 0}
parameters = {0.5: 0, 1: 0, 2: 0, 5: 0, 7: 0, 10: 0, 20: 0, 30: 0}
Bonds = [bond_03, bond_05, bond_1, bond_2, bond_5, bond_7, bond_10, bond_20, bond_30]

def find_exp(t: int)->list:
    """
    Brief: find the exponent for each term
    Input: t - a float
    Return: return a list of length 8, each element of type float
    """
    if t <= 0.5:
        return [t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 1:
        return [0.5, t-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 2:
        return [0.5, 0.5, t-1, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif t <= 5:
        return [0.5, 0.5, 1.0 , t-2, 0.0, 0.0, 0.0, 0.0]
    elif t <= 7:
        return [0.5, 0.5, 1.0 , 3.0, t-5, 0.0, 0.0, 0.0]
    elif t <= 10:
        return [0.5, 0.5, 1.0 , 3.0, 2.0, t-7, 0.0, 0.0]
    elif t <= 20:
        return [0.5, 0.5, 1.0 , 3.0, 2.0, 3.0, t-10, 0.0]
    elif t <= 30:
        return [0.5, 0.5, 1.0 , 3.0, 2.0, 3.0, 10.0, t-20]
    else:
        print("NOOOOOOO!!!")

def get_discount_value():
    

def get_equations():
    for bond in Bonds:
        pricing_fomula = 0
        for time in range(0.5, bond.maturity+1, 0.5):
            discount_time = time-bond.days/365
            pricing_fomula += find_exp(discount_time)

        
    

def get_bond_price(maturity:float)->float:
    '''
    return bond price given maturity    
    '''
    return 


def flat_to_full(flat, face, coupon, days):
    return flat + face * (coupon/2)*(days/180)




def solver(equations, results):
    return np.linalg.solve(equations, results)
    

"""
def discounted_value(T: float, yields: float):
    return 1/((1+yields/2)**T)

def calculate_price(yields:float, T:float, coupon:float, face)->float:
    price = 0
    # Coupon Payment
    for i in range(2*T):
        coupon_payment = coupon/2*
        price += discounted_value()
    # Face value payment
"""


"""
from sympy.interactive import printing
printing.init_printing(use_latex=True)
from sympy import Eq, solve_linear_system, Matrix
from numpy import linalg
import numpy as np
import sympy as sp

eq1 = sp.Function('eq1')
eq2 = sp.Function('eq2')

x, y = sp.symbols('x y')

eq1 = Eq(x-y+x, -4)
eq2 = Eq(3*x-1, -2)
display(eq1)
display(eq2)
row1 = [2, -1, -4]
row2 = [3, -1, -2]
system = Matrix((row1, row2))
display(system)
display(solve_linear_system(system, x, y))
"""