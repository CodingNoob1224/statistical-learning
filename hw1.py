#Parameter estimation by a Confidence interal and by MLE
###第一行先寫出名字＆學號。必須將"must"的中間結果印出來
##**Section One. 90% Confidence Iterval**
### **Example.** Determine the true defective rate by random chosen n = 100 samples and find out there are k defectives. Assume k is between 3~6.
# 郭佩芸 411410029

# The number of defective items k in 100 products would be randomly generated between 3~6
import numpy as np
# make sure we can generate random integers between 3 to 6
print(np.random.randint(3,6,size=10))

# Simulate the experiment where k is the number of successes in n trials
k = np.random.randint(3,6)
n = 100
# must-1: print the experiment result
print("1. There are",k,"successes in",n,"trials.")

# The point estimator p^= k/n
p_hat= k/n
# must-2: print the point estimator according to the experiment
print("2. The point estimator is", p_hat)

# Next for CI, we need to find out Margin of Error E
# E = z_crit * sigma_of_p (i.e., sqrt(npq))
# Since true p is unknown, we use sigma_of_p^ instead (i.e., sqrt(p^q^/n))
# Need to import math in order to do the square root operation, i.e., sqrt
import math
z_crit = 1.645  # 90% confidence: z_0.05
sigma = math.sqrt(p_hat*(1-p_hat)/n)
E = z_crit * sigma
# must-3: show the margin of error
print("3. The margin of error is",E)

# 90% CI about the population defective rate is
# the interval (left, right) where
# left = max(0, p_hat - E); right = min(1, p_hat + E)
left = max(0, p_hat - E)
right = min(1, p_hat + E)
# must-4: show the final result of CI
# use  "%.{}f".format(3) % x   to make sure x is in the form of three decimals
print("4. The 90% confidence interval of defective rate is (", "%.{}f".format(3)%left, ",","%.{}f".format(3)%right,")")

# **第一部分。求出 95% confidence interval of population proportion**。
##We want to estimate the true proportion of red M&Ms from a big bag of M&Ms. Randomly choose 60 candies and find out there are k red ones. Use this data to find a 95% Confidence interval of the true population proportion.
###* Use random number to simulate k so that it would be 12 ~ 18.
###* z_0.025 = 1.96, Keep the values of CI in three decimal places.

# 郭佩芸 411410029
# Write your code here.
# The number of red M&Ms k from 60 candies would be randomly generated between 12 to 18
import numpy as np
# Simulate the experiment where k is the number of successes in n trials
k = np.random.randint(12,18)
n = 60
# must-1: print the experiment result
print("1. There are",k,"successes in",n,"trials.")
# The point estimator p^= k/n
p_hat= k/n
# must-2: print the point estimator according to the experiment
print("2. The point estimator is", p_hat)
# Next for CI, we need to find out Margin of Error E
# E = z_crit * sigma_of_p (i.e., sqrt(npq))
# Since true p is unknown, we use sigma_of_p^ instead (i.e., sqrt(p^q^/n))
# Need to import math in order to do the square root operation, i.e., sqrt
import math
z_crit = 1.96  # 95% confidence: z_0.025
sigma = math.sqrt(p_hat*(1-p_hat)/n)
E = z_crit * sigma
# must-3: show the margin of error
print("3. The margin of error is",E)
# 90% CI about the population defective rate is
# the interval (left, right) where
# left = max(0, p_hat - E); right = min(1, p_hat + E)
left = max(0, p_hat - E)
right = min(1, p_hat + E)
# must-4: show the final result of CI
# use  "%.{}f".format(3) % x   to make sure x is in the form of three decimals
print("4. The 95% confidence interval of population proportion is (", "%.{}f".format(3)%left, ",","%.{}f".format(3)%right,")")

##**Section Two**. Maximum Likelihood Estimation for population proportion p.
###**Example**. Determine the true rate of getting a "6" (success) in a die tossing. Randomly roll a die n times and find out there are k successes. Assume we believe the true rate of getting a success is among {0.05,0.10,0.15,0.20,0.25,0.30}. Find an estimation p^ by MLE.
#### Let n = 50, k would be an integer among 12~16.

# Roll a die and interested in getting how many "6" ==> binomial
# The module scipy.stats contains many probability distributions including binomial
import numpy as np
from scipy.stats import binom

# For example, evaluate the probability of getting k=4 successes in 10 rolls and rate_of_success=1/6,
# i.e.,k=4 successes in bin(n=10, p=1/6) ==>  use binom.pmf(k,n,p)

binom.pmf(4,10,1/6)

# We can evaluate several probabilities together based on different success rates
binom.pmf(4, 10, [1/4,1/5,1/6])

# We are ready for the toy example given above.
# Fist define k and n.
k = np.random.randint(12,16)
n = 50
# must-1: print the experiment result, and the point estimator
print("1. There are",k,"successes in",n,"trials. The point estimator would be", k/n)


# We can use an array to produce multiple probabilities referring to different success rates
p_many = [0.05,0.10,0.15,0.20,0.25,0.30]
likelihood = binom.pmf(k,n,p_many)
# must-2: print the probabilities based on different rate of success
print("2. The likelihoods based on rates",p_many,"are\n", likelihood)

# We are interested to know which is the largest
m = max(likelihood)
# must-3: print the largest likelihood
print("3. The largest likelihood is", m)

# We are interested to which p gives the largest probability
# Use "enumerate" to find out the location i of the m in likelihood array
# Next, find out the corresponding p in p_many and this p is exactly what we want
for i in [i for i, x in enumerate(likelihood)if x==m]:
  print(i)
  p_hat = p_many[i]


# must-4: Print the final result, p^ by MLE, also in 3-decimal places
print("4. The estimated proportion p^ is","%.{}f".format(3)%p_hat,"according to the MLE.")

#**第二部分。以 MLE 求出 population proportion 的估計值**。
##Determine the true defective rate by random chosen n = 100 samples and find out there are k defectives. Assume we believe that the true rate is among {0.15, 0.20, 0.25, 0.30, 0.35, 0.40} and k is a random integer between 20 to 30.

# 郭佩芸 411410029
# Write your code here.
import numpy as np
from scipy.stats import binom

# Fist define k and n.
k = np.random.randint(20,30)
n = 100
# must-1: print the experiment result, and the point estimator
print("1. There are",k,"successes in",n,"trials. The point estimator would be", k/n)
# We can use an array to produce multiple probabilities referring to different success rates
p_many = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
likelihood = binom.pmf(k,n,p_many)
# must-2: print the probabilities based on different rate of success
print("2. The likelihoods based on rates",p_many,"are\n", likelihood)
# We are interested to know which is the largest
m = max(likelihood)
# must-3: print the largest likelihood
print("3. The largest likelihood is", m)
# We are interested to which p gives the largest probability
# Use "enumerate" to find out the location i of the m in likelihood array
# Next, find out the corresponding p in p_many and this p is exactly what we want
for i in [i for i, x in enumerate(likelihood)if x==m]:
  p_hat = p_many[i]
# must-4: Print the final result, p^ by MLE, also in 3-decimal places
print("4. The estimated proportion p^ is","%.{}f".format(3)%p_hat,"according to the MLE.")