import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import skew , poisson ,zscore
from scipy import stats


# data = [-300,65,56,64,70,78,90,674]
# series = pd.Series(data)


# sns.boxplot(series)
# plt.show()

# print(type(series))
# print(type(series.values))

# Q1 = series.quantile(0.25)
# Q3 = series.quantile(0.75)

# IQR = Q3 - Q1
# print(f"IQR : {IQR}")

# lower_bound = Q1-1.5*IQR
# upper_bound = Q3+1.5*IQR 
# print(f'Lower_bound {lower_bound}')
# print(f'Upper_bound {upper_bound}')


# Outliers_free = series[(series.values > lower_bound) & (series.values < upper_bound)]
# print(list(Outliers_free))

# print(series.mean())
# print(f'median is :- {series.median()}')
# print(f'mode is :- {series.mode()}')
# print(f'series is :- {series.quantile()}')
# print(f"10% of data :- {series.quantile(0.25)}")

# print(f"Statistical Summary :- {series.describe()}")



#OUTLIERS in pandas 

#Boxplot is used to get to know the outliers

#two types of outliers 

#1>univariate and multivariate

 
# types
# data entry errors 
# meausurement error 
# exprimental errors 
# intentional errors 



# data = [-300,-280 ,65,56,64,70,78,90,674]
# data1 = pd.Series   (data)

# method 1 Visual method /boxplot/histogram/scatterplot
# sns.boxplot(data1)
# plt.show()
# method 2  IQR INTER QUARTILE RANGE 


# q1 = data1.quantile(0.25)
# q3 = data1.quantile(0.75)

# IQR = q3 - q1 
# print(IQR)
# upper_lim = q3 + (1.5*IQR)
# lower_lim = q1 - (1.5*IQR)
# print(upper_lim)
# print(lower_lim) 

# data_outfree = data1[(data1.values < upper_lim)  & (data1.values > lower_lim)]
# print(data_outfree)

# sns.displot(data_outfree, kde = True )
# plt.show()


#2. replace the outliers by upper  and lower values  
#3. transformation or binnging technique  
# outliers left


#methods to treat skewness

# log_treat = np.log(data1)
# print(log_treat)

# sqr = np.sqrt(data1)
# print(sqr) 

# cbr = np.cbrt(data1)
# print(cbr)

# box = boxcox(data1)
# print(box)


# data = [-300,-280 ,65,56,64,70,78,90,674]
# data1 = pd.Series   (data)
# print(data1.skew())

# data2 = np.array(data1)
# print(skew(data2))

# # Example data


# data = np.array([-300,-280,65,56,64,70,78,90,674])  # Right-skewed
# # Calculate skewness
# skewness = skew(data)
# print(f"Skewness : {skewness}")

#______________________________Working with permutations and combinations

import itertools 
from itertools import permutations , combinations , combinations_with_replacement 

#BY DEFAULT PERMUTAIONS AND COMBINATIONS IN PYTHON ARE WITHOUT REPLACEMENT , WHICH ONCE THE ITEM IS USED , I CAN'T BE REUSED AGAIN 

#PERMUTATIONS :- ARRANGEMENT OF ITEMS WHERE ORDER MATTERS 
seq = 'CAT'

# perm = permutations(seq , 3)

for permutations_ in permutations(seq):
    print(permutations_)

#calculating the number of permutations count 
seq = ['Ajith','Bhavani','Rushikesh',"Madhu"]

count = 0
for perm in permutations(seq):
    print(perm)
    count += 1 
print(count)

# Generate permutations of 2 names taken from the list

names = ['Alice','Ajith','Bhavani','Rushikesh',"Madhu"]
count = 0 
for perm in permutations(names, 2):
    print(perm)
    count += 1 
print(count)


#COMBINATIONS :- ARRANGEMENT OF ITEMS WHERE ORER DOESN'T MATTERS 

# Generate permutations of 2 names taken from the list
names = ['Alice','Ajith','Bhavani','Rushikesh',"Madhu"]
count = 0 
for comb in combinations(names , 2 ):
    print(comb)
    count += 1 
print(f'Combinations Count : {count}')

#Combinations with replacement 
count = 0 
for comb in combinations_with_replacement(names , 2 ):
    print(comb)
    count += 1 
print(f"Combinations With_replacement count : {count}")



#PERMUTATIONS 
sequence = ['p', 'y', 't', 'h', 'o', 'n']
count = 0
for perm in permutations(sequence , 3):
    print(perm)
    count += 1
print(f"Total Permutaions Count :- {count}")


name_seq = ['prachi', 'yogesh', 'tina', 'hetal','ovi', 'riya']
count = 0
for comb in combinations(name_seq , 2):
    print(comb)
    count += 1 
print(f'Combinations Count {count}')


comb_seq = combinations_with_replacement([1, 2, 3, 4, 5, 6], 3)
count = 0
for comb in comb_seq:
    print(comb)
    count += 1
print(f"Combination in the seq  {count}")



#___________________________________________Binomial Distribution

""""
Huge Fruit Basket (HFB) is a grocery shop that sells fruits. It is observed that 20% of their customers complain about the fruits 
purchased by them for many reasons (bad quality, foul smell and less quantity). On Friday, 30 customers purchased fruits from HFB.

1. Calculate the probability that exactly 10 customers will complain about the purchased products.
2. Calculate the probability that upto 10 customers will complain about the purchased products.
3.Calculate the probability that atleast 10 customers will complain about the purchased products.
"""
sample_size = 30 
probability_of_sucess = 0.2 

x = 10 

binomial_probabilty_distrubution =  stats.binom.pmf(x , sample_size , probability_of_sucess)*100
print(f"1 SOL : probability that exactly 10 customers will complain about the purchased products : {binomial_probabilty_distrubution}")

# x <= 10 or x >= 10
binomial_probabilty_distrubution2 =  stats.binom.cdf(x , sample_size , probability_of_sucess)*100
print(f"2 SOL : probability that upto 10 customers will complain about the purchased products : {binomial_probabilty_distrubution2}")

binomial_probabilty_distrubution3 = 1 - stats.binom.cdf(x , sample_size , probability_of_sucess)
print(f"3 SOL : probability that atleast 10 customers will complain about the purchased products : {binomial_probabilty_distrubution3*100}")

#Calculate the probability that maximum of 8 customers will complain about the fruits purchased by them. 
binomial_probabilty_distrubution =  stats.binom.cdf(8 , sample_size , probability_of_sucess)*100
print(f"4 SOL : probability that max of 8 customers will complain about the purchased products : {binomial_probabilty_distrubution}")

#TO CALCULATE MEAN/VARIANCE/SKEWNESS/KURTOSIS WE USE A METHOD CALLED {"MOMENTS"}

mean , variance , skewness , kurtosis = stats.binom.stats(sample_size , probability_of_sucess, moments = 'mvsk')
print(f"Mean     : {mean:.4f}")
print(f"Variance : {variance:.4f}")
print(f"Skewness : {skewness:.4f}")
print(f"Kurtosis : {kurtosis:.4f}")


"""2>
From the experience, it is seen that 3% of the tyres produced by the machine are defective. 
Out of the 15 tyres produced, find the probability that at most 2 are defective.
"""
p = 3/100
n = 15
x = 2 
prob = stats.binom.cdf(x , n , p)
print(f"probability that at most 2 are defective : {prob*100}")



"""
Big Basket is a grocery shopping app that sells groceries & food materials.
 It is observed that 40% of their customers complain about the vegetables purchased by them for many reasons 
 (bad quality, foul smell, and less quantity). On Sunday, 90 customers purchased vegetables from Big Basket.

(i). Calculate the probability that more than 25 customers will complain about the vegetables purchased by them.
(ii). Find the average number of customers who are likely to complain about the vegetables. Also, find the variance of the number of complaints.
"""

p = 40/100
n = 90 
x = 25 

prob = 1 -  stats.binom.cdf(x , n , p)
print(f'the probability that more than 25 customers will complain {prob*100}')

mean , variance,skewness,kurtosis = stats.binom.stats(n ,p , moments = 'mvsk')
print(f"Mean     : {mean:.4f}")
print(f"Variance : {variance:.4f}") 



#______________________________________________#Poisson Distribution
"""
The number of customer returns in a retail chain per day follows a poisson distribution at a rate of 25 returns per day.

1.Calculate the probability that the number of returns exact 30 in a day.

2.Calculate the probability that the number of returns exceeds 30 in a day.
"""


Poisson = 25 
n = 30 

prob = stats.poisson.pmf(n,Poisson)
print(f"the probability that the number of returns exact 30 in a day : {prob*100}")

prob = 1 - stats.poisson.cdf(n,Poisson)
print(f"the probability that the number of returns exceeds 30 in a day : {prob*100}")


"""
The number of road accidents on the day follow Poisson distribution with mean equals to 3. 
What is the probability that on a day exactly 1 accident will happen?
""" 
psn_dtr = 3 
x = 1 
prob = stats.poisson.pmf(x,psn_dtr)
print(f"probability that on a day exactly 1 accident will happen : {prob*100}")

"""
The number of trucks crossing a bridge during the day follow a Poisson distribution with mean 22. 
What is the probability that on a randomly selected day 14 trucks would have crossed the bridge?
"""
psn_dtr = 22
x = 14
prob = stats.poisson.pmf(x,psn_dtr)
print(f"Probability of exactly 14 trucks {prob}")


#_________________________________________NORMAL DISTRIBUTION 

"""
#### Normal Distribution
Assume a normal distribution where the mean clotting time of blood is 7.35 seconds, with a standard deviation of 0.35 seconds. 
What is the probability that blood clotting time will be less than 7 seconds
"""

mean = 7.35 
std = 0.35 
x = 7 # X<7 
prob = stats.norm.cdf(x , mean , std )
print(f"probability that blood clotting time will be less than 7 seconds : {prob*100}")


"""
Assume a normal distribution where the average size of the bass in a lake is 11.4 inches, with a standard deviation of 3.2 inches. 
Find the probability of catching a bass longer than 17 inches.
"""
mean = 11.4 
std = 3.2 
x = 17 # x > 17 

prob = 1 - stats.norm.cdf(x , mean,std)
print(f"probability of catching a bass longer than 17 inches : {prob*100}")

"""
The IQ of students follows normal distribution with mean 95 and variance 10. 
What is the probability that the any student selected at random will have IQ more than 102?
"""
mean = 95 
var = 10 
std = np.sqrt(var)
x = 120 
prob = 1 - stats.norm.cdf(x , mean , std)
print(f"probability that the any student selected at random will have IQ more than 102 {prob*100}")


"""
calculate the mean , std , z scores fro the give data
"""

data = [16.0, 16.0, 30.0, 37.0, 25.0, 22.0, 19.0, 35.0, 27.0, 32.0,
34.0, 28.0, 24.0, 35.0, 24.0, 21.0, 32.0, 29.0, 24.0, 35.0,
28.0, 29.0, 18.0, 31.0, 28.0, 33.0, 32.0, 24.0, 25.0, 22.0,
21.0, 27.0, 41.0, 23.0, 23.0, 16.0, 24.0, 38.0, 26.0, 28.0]

mean = np.mean(data)
print(f'Mean : {mean}')


std_ = np.std(data, ddof=1) #sample std
print(f'STD : {std_}')

# list_ = []
# for x in data:
#     z_score = (x - mean) / std 
#     list_ += [z_score]
# print(f"Z-scores : {np.round(list_, 2)}")

zscores = zscore(data , ddof=1)
print(f"Z-scores (scipy) : {np.round(zscores, 2)}")

max , min = np.max(zscores), np.min(zscores) 
print(f'Max Z-score : {max}')   
print(f'Min Z-score : {min}')   

"""
🔹 1. Presence of Outliers

A few extreme values (very large or very small compared to the rest) pull the tail of the distribution.

Example: Incomes in a city — most people earn between ₹20k–₹60k, but a few earn crores → positive skew.

🔹 2. Natural Distribution of Data

Some processes are inherently skewed.

Right-skewed (positive skew): waiting times, incomes, rainfall amounts.

Left-skewed (negative skew): exam marks (if most students score very high, but a few score very low).

🔹 3. Bounded Data

When data has a minimum o


"""







































