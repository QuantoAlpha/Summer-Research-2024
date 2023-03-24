# Summer Research 2024
Yield Curve &amp; Option Pricing

This repo contains a report on fixed income derivative pricing under continuous-time term structure model and a monte carlo approach to price zero coupon bond price with interest rate under Ho-Lee model. 

We further discretized the short rate evolution for Ho-Lee, so you can get the price of both continuous time (simulation of Brownian Motion) and discrete time (simulation of Bernoulli Variables).

You can access the pricing function under ZCB_Pricing_oop.py in the following way:

0. edit the Nelson Sigel Estimates parameters in stats.txt

1. choose either discrete or continuous pricing by creating a class instance with your input 

2. call the pricing function under input passing in following parameter
   t: time to purchase the ZCB
   T: maturity of ZCB (T >= t)
   N: number of periods to generate the Monte Carlo path
   M: number of paths generate for Monte Carlo
   
3. The output should be a tuple of (x, y), where
   x: average of paths for monte carlo simulation - we take this as the price
   y: standard deviation of paths.
