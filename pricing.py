
# coding: utf-8

# In[1]:


import scipy


# In[2]:


import numpy as np
from scipy.stats import norm as norm
import scipy.optimize as optimize
import sys
import pandas as pd

n = 10000  # number of sample paths for each maturity
period = 0.25 # length of each step
no_of_per = 8  # number of periods from time 0 to yr 2
TTMs = period * np.arange(0, no_of_per)  # 8 TTMs, the first TTM is 0, the last TTM is 1.75
current_fwd = np.array(
    [0.0520, 0.0575, 0.0591, 0.0601, 0.0609, 0.0643, 0.0667, 0.0642])  # 8 entries of current 1 period forward rates
BlackISD = np.array(
    [0.2416, 0.2228, 0.2016, 0.1897, 0.1757, 0.1609, 0.1686])  # 7 entries corresponding to the last 7 TTMs

current_bond_price = 1 / (np.cumprod(1 + current_fwd / 4))  # the discount factors, with discrete compounding 4 times per year.


def calculate_caplet_price():
    d1 = (BlackISD * BlackISD * period * np.arange(1, no_of_per) / 2) / (
                BlackISD * np.sqrt(period * np.arange(1, no_of_per)))
    d2 = (- BlackISD * BlackISD * period * np.arange(1, no_of_per) / 2) / (
                BlackISD * np.sqrt(period * np.arange(1, no_of_per)))
    return period * (current_bond_price[1:] * current_fwd[1:] * norm.cdf(d1) - current_fwd[1:] * current_bond_price[
        1:] * norm.cdf(d2)) * 1000


caplet_price = calculate_caplet_price()   # calculate caplet market price from Black Implied Standard Deviations

z = np.random.normal(size=(n, no_of_per - 1, 2))  # generate random numbers


def a_t(index1, param):  # index1 is the index of times (TTMs)
    return param[index1] # param is an array of length 7

def s1(index1, index2, param, h3=0):  # index1 is the index of t in TTMs, index2 is index of T in TTMs
    sigma1 = a_t(index1, param) * param[-1] # phi is the last element in param
    # if sigma1 >= 0:
    return h3 + sigma1
    # else:
    # return 10


def s2(index1, index2, param, h4=0):
    sigma2 = a_t(index1, param) * (np.exp(-2 * (TTMs[index2] - TTMs[index1] - 0.25)) - 0.5)
    # if sigma2 >= 0:
    return h4 + sigma2
    # else:
    # return 10


def drift(index1, index2, param, h3=0, h4=0):
    miu = 0
    integration = 0
    for j in range(index1, index2):
        integration += s1(index1, j, param, h3) * period
    miu += (s1(index1, index2, param, h3) * integration)

    integration = 0
    for j in range(index1, index2):
        integration += s2(index1, j, param, h4) * period
    miu += (s2(index1, index2, param, h4) * integration)
    return miu


grid = np.zeros((no_of_per, no_of_per))  # a grid of expectations of instantaneous forward prices starting from time 0 to time 1.75.
# f(TTMs[j], TTMs[i]) will be in the location (i,j)
# same T in the same row, same t in the same column

calibrated_caplet = np.zeros(no_of_per - 1)
correlation_data = np.ones((4, 2))  # correlation data, there are 4 pairs of 3 month and 1 year rate changes


def calibration(param):

    discount = np.ones(n) / (1 + current_fwd[0] / 4)  # a vector that stores the pathwise discount factors of the 10000 paths for maturity i
    grid[0, 0] = current_fwd[0] # this will be the instantaneous forward rate from f(0,0) to f(0,0.25)

    for i in np.arange(1, no_of_per):  # i is the index of time T in TTMs
        f = np.ones(n) * current_fwd[i]  # a vector that stores the initial/current values of the 10000 paths for maturity i
        grid[i, 0] = current_fwd[i]

        for j in range(0, i):
            f= f + drift(j,i, param) * period + s1(j, i, param) * z[:, j, 0] * np.sqrt(period) + s2(j,i,param) * z[:, j, 1] * np.sqrt(period)
            ##xx[k,i, j + 1]=f
            grid[i, j + 1] = np.mean(f)  # the mean of f is the instantaneous forward rate at t = TTM[j+1] for maturity i

        discount = discount/(1 + (f / 4)) # at f(T,T), a new one period rate is realised, update the discount factor accordingly
        calibrated_caplet[i - 1] = np.mean(np.maximum(f - current_fwd[i], np.zeros(n)) * discount[0:n]) * 1000

    correlation_data[0, 0] = grid[1, 1] - grid[0, 0]  # change of 3 months rate in one period
    correlation_data[0, 1] = (1 + grid[1, 1] / 4) * (1 + grid[2, 1] / 4) * (1 + grid[3, 1] / 4) * (
                1 + grid[4, 1] / 4) - (1 + grid[0, 0] / 4) * (1 + grid[1, 0] / 4) * (1 + grid[2, 0] / 4) * (
                                         1 + grid[3, 0] / 4) # calculate changes in 1 year spot rate (1 yr rate is calculated with forward rates)
    for i in range(1, 4):
        correlation_data[i, 0] = grid[i + 1, i + 1] - grid[i, i]
        correlation_data[i, 1] = (1 + grid[i + 1, i + 1] / 4) * (1 + grid[i + 2, i + 1] / 4) * (
                    1 + grid[i + 3, i + 1] / 4) * (1 + grid[i + 4, i + 1] / 4) - (1 + grid[i, i] / 4) * (
                                             1 + grid[i + 1, i] / 4) * (1 + grid[i + 2, i] / 4) * (
                                             1 + grid[i + 3, i] / 4)
    corr = np.corrcoef(correlation_data[:, 0], correlation_data[:, 1])[0, 1]
    mse = np.dot(calibrated_caplet - caplet_price, calibrated_caplet - caplet_price)
    print(mse)
    return mse


def corr_constraint(param):
    corr = np.corrcoef(correlation_data[:, 0], correlation_data[:, 1])[0, 1]
    return corr


constraint = optimize.NonlinearConstraint(fun=corr_constraint, lb=0.81, ub=0.81)

result = optimize.minimize(fun=calibration,x0=[ 9.71375234e-03,  8.17162187e-02, -3.36126869e-02, -1.24135605e-02,
       -4.22680060e-04,  6.08222872e-05, -4.13074927e-04,  2.95759249e-02],tol=1e-06,method='trust-constr',\
                           options={'xtol':1e-06,'gtol':1e-06,'maxiter':30000},constraints=constraint)

print(result)
param = result.x

import os
outfile = open("67184800.csv", "w", newline="")
outfile.write("a(t) and phi\n")
np.savetxt(outfile, param)
outfile.write("\nSimulated Forward Curve\n")
forward_output = pd.DataFrame(grid)
forward_output.to_csv(outfile, index=0)
outfile.write("\nCalibrated Caplet Price\n")
np.savetxt(outfile, calibrated_caplet)
outfile.write("\nCorrelation\n")
outfile.write(str(corr_constraint(param)))
outfile.write("\nMSE\n")
outfile.write(str(result.fun))
outfile.close()

# In[6]:
z2 = np.random.normal(size=(n, no_of_per - 1, 2)) # we need a set of common random variables for resimulation


def forward(param, h1=0, h2=0, h3=0, h4=0):
    # h1 is a small change in 3 month and 1 year spread, h2 is a small change in initial forward value
    # h3 is a small change in sigma1, h4 is a small change in sigma2
    fwd_rates = current_fwd
    fwd_rates[1:4] += h1/3
    # change in initial spread between 3-month and 1-year rate:
    # suppose the change in initial spread is equal across time from 3 months onwards to 1 year
    # the rate over the immediate next 3 months from now is already fixed
    # so average h1/3 change for each 3 month period after the next
    fwd_rates[0] += h2
    grid2 = np.zeros((n, no_of_per, no_of_per))
    discount = np.ones(n) / (1 + fwd_rates[0] / 4)  # a vector that stores the pathwise discount factors of the 10000 paths for maturity i
    grid2[:,:,0] = fwd_rates
    for i in np.arange(1, no_of_per):
        f = np.ones(n) * fwd_rates[i]
        for j in range(0, i):
            f = f + drift(j, i, param, h3, h4) * period + s1(j, i, param, h3) * z2[:, j, 0] * np.sqrt(period) + \
                s2(j, i, param, h4) * z2[:, j, 1] * np.sqrt(period)
            grid2[:, i, j + 1] = f  # the mean of f is the instantaneous forward rate at t = TTM[j+1] for maturity i
        if i < 4:
            discount = discount/(1 + (f / 4)) # at f(T,T), a new one period rate is realised, update the discount factor accordingly
    return grid2, discount

cur_spread = abs(sum(current_fwd[0:4]) * 0.25 - current_fwd[0])
print(cur_spread)#Calculate current spread and

spread_sigma = np.zeros(1) # this is used to keep track of sigma of the spread (which will be used later)
grid, disfac = forward(param, 0, 0, 0, 0)
required_para = grid[:, 4:, 4].transpose()
new_spread = abs(sum(required_para) * 0.25 - required_para[0])
sigma = np.std(new_spread)

short_month_rate = np.zeros((7,10000))
vol_rate = np.zeros((7,1))
for i in range(1,8):
    short_month_rate[i-1,:] = grid[:,i,i]
    vol_rate[i-1] = np.std(short_month_rate[i-1,:])

spred = np.zeros((4,10000))
vol_spread= np.zeros((4,1))
for i in range(1,5):
    required_para = grid[:, i:i+4, i].transpose()
    spred[i-1,:] =abs(sum(required_para) * 0.25 - required_para[0])
    vol_spread[i-1] = np.std(spred[i-1:])

def pricing(B1, B2, param, curspread, curr1, h1=0, h2=0, h3=0, h4=0):#B1 for f(1,1) #B2 for spread
    grid, disfac = forward(param, h1, h2, h3, h4)
    required_para = grid[:, 4:, 4].transpose()
    new_spread = abs(sum(required_para) * 0.25 - required_para[0])
    sigma = np.std(new_spread)
    spread_sigma[0] = sigma #stores this sigma so that it can be accessed elsewhere
    condition1 = np.logical_and(required_para[0] > curr1, curr1 * (1 + B1) > required_para[0])
    condition2 = np.logical_and(new_spread > 0.5 * curspread * (1 - B2) , new_spread < 0.5 * curspread * (1 + B2))
    payoff = np.logical_and(condition1, condition2)
    price = payoff * disfac
    mean = np.mean(price)
    stderr = np.std(price) / np.sqrt(n)
    return (mean, stderr, price)


#mean, stderr, price = pricing(0.2, 2.33, param, cur_spread, current_fwd[0])


R1 = [0.1,0.15,0.2,0.25,0.3]
R2 = [0.1,0.15,0.2,0.25,0.3]

option_price = np.zeros((len(R1),len(R2)))
for i in range(len(R1)):
    for j in range(len(R2)):
        print(R1[i],R2[j],pricing(R1[i],R2[j],param,cur_spread,current_fwd[0])[0:2])
        option_price[i,j] = pricing(R1[i],R2[j],param,cur_spread,current_fwd[0])[0]


print("DELTA")
def delta(B1, B2):
    h1 = cur_spread * 0.01
    h2 = current_fwd[0] * 0.01
    p11 = pricing(B1, B2, param, cur_spread, current_fwd[0] , h1, 0, 0, 0)[2]
    p12 = pricing(B1, B2, param, cur_spread, current_fwd[0] , -h1, 0, 0, 0)[2]
    p21 = pricing(B1, B2, param, cur_spread, current_fwd[0] + h2, 0, h2, 0, 0)[2]
    p22 = pricing(B1, B2, param, cur_spread, current_fwd[0] - h2, 0,-h2, 0, 0)[2]
    delta1 = (p11 - p12) / (2 * h1)
    delta2 = (p21 - p22) / (2 * h2)
    mean1 = np.mean(delta1)
    mean2 = np.mean(delta2)
    stderr1 = np.std(delta1) / np.sqrt(n)
    stderr2 = np.std(delta2) / np.sqrt(n)
    return mean1, stderr1, mean2, stderr2


for i in range(len(R1)):
    for j in range(len(R2)):
        print(R1[i],R2[j],delta(R1[i],R2[j]))

cur_spread = abs(sum(current_fwd[0:4]) * 0.25 - current_fwd[0])  #Calculate current spread

# In[43]:

print("VEGA")
def vega(B1,B2):
    h3 = 0.00001
    h4 = 0.00001
    p11 = pricing(B1, B2, param, cur_spread, current_fwd[0], 0, 0, h3, 0)[2]
    p12 = pricing(B1, B2, param, cur_spread, current_fwd[0], 0, 0, -h3, 0.)[2]
    vega1 = (p11 - p12)/(2*h3)
    p21 = pricing(B1, B2, param, cur_spread, current_fwd[0] + h4, 0, 0, 0, h4)[2]
    p22 = pricing(B1, B2, param, cur_spread, current_fwd[0] - h4, 0, 0, 0, -h4)[2]
    vega2 = (p21 - p22)/(2*h4)
    mean1 = np.mean(vega1)
    mean2 = np.mean(vega2)
    stderr1 = np.std(vega1) / np.sqrt(n)
    stderr2 = np.std(vega2) / np.sqrt(n)
    return mean1, stderr1, mean2, stderr2


# In[45]:


for i in range(len(R1)):
    for j in range(len(R2)):
        print(R1[i],R2[j],vega(R1[i],R2[j]))

def digital_put_price(param, strike): # price of the digital put spread option used as hedging instruments
    grid, disfac = forward(param)
    required_para = grid[:, 4:, 4].transpose()
    new_spread = abs(sum(required_para) * 0.25 - required_para[0])
    #sigma = np.std(new_spread)
    put_payoff = np.int8(new_spread <= strike)
    #condition1 = np.logical_and(required_para[0] > curr1,curr1 * (1 + B1) > required_para[0])
    #condition2 = np.logical_and(new_spread > 0.5 * curspread - (sigma) * B2 , new_spread < 0.5 * curspread + (sigma) * B2)
    #payoff = np.logical_and(condition1,condition2)
    price = put_payoff * disfac
    mean = np.mean(price)
    stderr = np.std(price) / np.sqrt(n)
    return (mean, stderr, price)


def floorlet_price(param, strike):
    grid, disfac = forward(param)
    required_para = grid[:, 4, 4].transpose()
    floorlet_payoff = np.int8(required_para <= strike)
    #condition1 = np.logical_and(required_para[0] > curr1,curr1 * (1 + B1) > required_para[0])
    #condition2 = np.logical_and(new_spread > 0.5 * curspread - (sigma) * B2 , new_spread < 0.5 * curspread + (sigma) * B2)
    #payoff = np.logical_and(condition1,condition2)
    price = floorlet_payoff * disfac
    mean = np.mean(price)
    stderr = np.std(price) / np.sqrt(n)
    return (mean, stderr, price)


# R1 is possible configurations for the band
# R2 is possible configurations for the spread
hedge_costs = np.zeros((len(R1),len(R2),2))
for i in range(len(R1)):
    lb1 = current_fwd[0]
    ub1 = current_fwd[0]*(1+R1[i])
    for j in range(len(R2)):
        lb2 = 0.5 * cur_spread - spread_sigma[0] * R2[j]
        ub2 = 0.5 * cur_spread + spread_sigma[0] * R2[j]
        hedge_cost = 0.5*(floorlet_price(param, ub1)[2] - floorlet_price(param, lb1)[2]) + \
                     0.5*(digital_put_price(param, ub2)[2] - digital_put_price(param, lb2)[2])
        hedge_costs[i,j,0] = np.mean(hedge_cost)
        hedge_costs[i,j,1] = np.std(hedge_cost) / np.sqrt(n)
        print("hedge_cost")
        print(R1[i], R2[j], hedge_costs[i,j,0], hedge_costs[i,j,1])
