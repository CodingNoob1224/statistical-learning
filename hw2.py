# Batting Rate Estimation
### We use MLP and MAP to estimate the batting rate of a player
### 名詞：batting_rate打擊率; hits安打數, at_bats有效打擊數, batting_rate = hits / at_bats

#### 請寫名字＆學號：郭佩芸 411410029

### import necessary functions and initialization

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior beta distribution based on the average batting rate of all players (0.25)
# assuming 5 hits in 20 (so miss 15), alpha=hits+1=6, beta=miss+1=16
prior_alpha = 6
prior_beta = 16

## Define a function to calculate Mode(X) where
### X is a Beta RV with distribution f(x)
### input (alpha,beta)
### output Mode(X) which is also argmax f(x)

# we use two methods to write a function which can output Mode
# for example, input (a,b) and output a*b
# Method 1. use "def"
def toy1(a,b):
  return a*b

# Method 2. use "lambda" when it is a very simple function involving only one expression
toy2 = lambda a,b: a*b

print(" Method 1:",toy1(2,3),"\n","Method 2:", toy2(2,3)) #兩者都得到同樣結果

# Q1.現在利用上述兩種方法 method1 & method2 計算 mode, with input alpha, beta, and output the mode
# 並測試你的結果印出 mode(6,16) & mode(8,5) (答案應該是0.25, 0.6364) ＊注意，mode(1,1)是不存在的。
# 後續程式中，使用其中一個方法計算 mode 即可

def mode(a,b):
  return (a-1)/((a-1)+(b-1))

mode2 = lambda a,b: (a-1)/((a-1)+(b-1))

print ("mode(6,16) using method 1 is", mode(6,16))
print ("mode(8,5) using method 1 is", mode(8,5))
print ("mode(6,16) using method 2 is", mode2(6,16))
print ("mode(8,5) using method 2 is", mode2(8,5))

## To estimate the season batting rate of a player

### three observations at different times are made
####   * the first observation is after (k1) at-bats and it has (h1) hits

####   * the second observation is after additional (k2) at-bats and it has (h2) hits
####   * the third observation is after additional (k3) at-bats and it has (h3) hits

#### 美國大聯盟平均打擊率約0.25，標準差約0.01124
#### 因此我們模擬球員安打數=round(at_bats*(0.25±在一個標準差內的隨機數))

k1 = 2   #the first observation made after (k1) at bats
k2 = 13   #the second observation after additional (k2) at bats
k3 = 185  #the third observation made after additional (k3) at bats
at_bats_observed = [k1, k1+k2, k1+k2+k3]
print("Total number of at bats of three observations", at_bats_observed)

# 隨機(高斯）模擬一個擊球率 大聯盟的平均&標準差 r=0.25 sigma=0.01124
rate = lambda r,sigma: r + np.random.normal(0,sigma)
# Q2. 以下請印出：以同樣輸入值(0.25, 0.01124)但是呼叫3次，確定rate的確可以產生不同的隨機打擊率。
print(rate(0.25,0.01124))
print(rate(0.25,0.01124))
print(rate(0.25,0.01124))

# 模擬安打數：利用模擬得出的擊球率＊有效的擊球數，最後印出結果
h1 = round(k1 * rate(0.25, 0.01124))    # number of hits after the first observation
h2 = round(k2 * rate(0.25, 0.01124))    # number of additional hit at the second observation
h3 = round(k3 * rate(0.25, 0.01124))    # number of additional hit at the third observation

hits_observed = [h1, h1+h2, h1+h2+h3]
print("The player has the following hits:")
for i in range(len(hits_observed)):
  print(hits_observed[i],"hits after",at_bats_observed[i],"at bats has observed")


## MLE estimate
#### it is a bernoulli: hit or miss for every swing
#### with P(hit) = batting_rate = p

#### 注意，我們將會比較 MLE & MAP 的估計結果。

#介紹zip可以將相對的兩個list逐一配對而成序對,再加上 for loop 可以很有效率的做運算
#在 add_two 範例中，稱 zip 的序對為 (a, b) 然後 for loop 做 a+b+2 運算
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
add_two = [a + b + 2 for a, b in zip(x, y)]
print(add_two)   #應該是 [8, 10, 12, 14]

# Q3.使用 zip 將 hits_observed ＆ at_bats_observed 依序配成序對，
# 如果稱 zip 的序對為 (a, b)，以 for loop 做 a/b 就可以得到 hits_observed / at_bats_observed,即為batting rate 的 mle 估計值
# 將計算所得的結果存在 mle_estimate
mle_estimate = [ht / bt for ht, bt in zip(hits_observed, at_bats_observed)]
print (mle_estimate)

##MAP with Beta distribution
####Assume batting rate is a Beta(alpha, beta) random variable
####where (alpha-1) & (beta-1) representing the number of hits (success) and misses (failure)
###When using MAP with Beta prior,
#### p^ of MAP = argmax f(x) = Mode(X)
#### which will call the mode function defined earlier     


# Given prior distribution of batting rate to be Beta(alpha, beta)
# If there are addition experiments with k hits and m misses (total k+m at bats)
# The posterior distribution of batting rate would be Beta(alpha+k, beta+m)
# The following is a toy example, given prior is Beta(6,16), and after (toy_at_bats) observations with (toy_hits) hits

toy_at_bats = [10, 20, 30] # 老師的範例 (3次的有效擊球數)
toy_hits = [1, 2, 3] # 老師的範例(分別的安打數)
toy_miss = np.array(toy_at_bats) - np.array(toy_hits) # 要先改為"array"才能運算喔！ (3次的揮空數)應該是[9,18,27]
print("miss", toy_miss)

# 建立 parameters alpha, beta of the posterior Beta distribution Beta後驗機率分佈
post_alpha = np.array(prior_alpha) + np.array(toy_hits) # 老師的小範例 alpha <-- alpha+hits
post_beta = np.array(prior_beta) + np.array(toy_miss) # 老師的小範例 beta <-- beta+miss
# 再度利用 zip 將 alpha & beta 逐一配對再呼叫 mode 即可算出 map 的估計結果
map_estimate = [mode(a,b) for a,b in zip(post_alpha,post_beta)]
# 印出結果，應該是 [0.2, 0.175, 0.16]
print("map",map_estimate)

# Q4. write your code here to calculate the MAP estimation of batting rate
# 應先建立 miss, 接著以 hits & miss 得到 post_alpha, post_beta
# 利用 zip 將 post_alpha & post_beta 逐一配對再呼叫 mode 即可算出 map 的估計結果
# 將計算所得的結果存在 map_estimate
miss_observed = np.array(at_bats_observed) - np.array(hits_observed)
post_alpha = np.array(prior_alpha) + np.array(hits_observed)
post_beta = np.array(prior_beta) + np.array(miss_observed)

map_estimate = [mode(a,b) for a,b in zip(post_alpha,post_beta)]
print("map_estimate",map_estimate)



## 請將寫好的程式多跑個幾次，每次都觀察三個不同時間點的打擊率估計值(by MLE & MAP)。當增加實驗次數時是否prior的效果會被"wash out"? 有或沒有，請說明為何你做如此結論。（另外以 PDF 檔案將此程式執行3次，觀察結果並回答問題。每次執行都應該註明 experiment 1(or 2, or 3) & 並截圖顯示 results of Q2 & Q5)

#Q5.將剛才運算所得的 mle & map 結果印出，並另外以檔案回答上述問題。

print("The Batting rate estimation by two methods:")

print('Maximum Likelihood Estimation')
for i in range(len(at_bats_observed)):
  print('batting rates after has obsersved',at_bats_observed[i], 'at bats is',"{:.4f}".format(mle_estimate[i]))

print('Maximum a Posterior with beta distribution prior')
for i in range(len(at_bats_observed)):
  print('batting rates after has obsersved',at_bats_observed[i], 'at bats is',"{:.4f}".format(map_estimate[i]))