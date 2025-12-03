import numpy as np 
from scipy import stats
from scipy.stats import poisson
import matplotlib.pyplot as plt


# x = [1,0]
# p = 0.7 
# q = 1-p 
# print(stats.bernoulli.pmf(x,p))

# print(print(stats.bernoulli.cdf(x,p)))  #this will calculates the cmf


# p = 0.75

# rs = stats.bernoulli.rvs(p, size = 100)
# print(rs)

# n = 30 
# p = 0.2 
# x = 10 
# prob = stats.binom.pmf(x,n,p)
# print(prob)

# print(stats.binom.cdf(x,n,p)*100)

# print(stats.binom.cdf(9,n,p)*100)

# print(stats.binom.pmf(8,n,p)*100)
# print(stats.binom.stats(30, 0.2 , moments = 'mvsk'))


#2> 
# p = 0.4 
# n = 25 
# print(stats.binom.stats(25,0.4, moments='mvsk'))


#poison distrubution 

# n = 25 
# x = 30 
# x = stats.poisson.pmf(30,25)
# print(f"pmf : {x}")

# y = 1 - stats.poisson.cdf(30,25)
# print(f'CDF : {y}') 

