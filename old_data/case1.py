import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import minmax_scale 
from dask.array.linalg import norm

random.seed ( 10 )

epoch = 10000                    #最多跑的回合數
learning_rate = 0.7           #學習率
tau = 0.01
path1 = "train_case1.txt"
initw = np.array([random.random(), random.random(), random.random()])

#處理檔案
#data = genfromtxt(path1, delimiter=',', names=(0, 1, 2))
data = np.array([[0,0,0],
                [0,1,0],
                [1,0,0],
                [1,1,1]])

def sigmoid(data):
    #return 1 / ( 1 + math.exp(np.sum(data)))
  if np.sum(data) >= 0:
    return  1 / ( 1 + math.exp(-np.sum(data)))
  else:
    return math.exp(np.sum(data)) / (1 + math.exp(np.sum(data)) )

def CrossEntropy(y_hat, y):
    if y == 1:
        return -math.log(y_hat + 1e-15)
    else :
        return -math.log(1 - y_hat + 1e-15)

def minmax_scale(w, min, max):
    return (w-min) / (max-min)
def re_minmax_scale(w):
    return w * (max1-min1) + min1

def to_unitVector(w):
    if w[1] < 1 and w[2] < 1:
        return w
    sqrtw = math.sqrt(w[1]*w[1] + w[2]*w[2])
    w[1] = w[1] / sqrtw
    w[2] = w[2] / sqrtw
    return w

def logistic_regression(dataset):
    w = initw                                                   #initialize weight and w0 is bias
    #print(w)
    for ep in range(epoch):
        flag = True
        delta_w = np.zeros(len(w))
        loss = 0
        for cur in dataset:
            y = cur[2]
            tmp = np.array(  (1, minmax_scale(cur[0], min1, max1), minmax_scale(cur[1], min2, max2) ) )    # minmax_scale with feature 
            y_hat = sigmoid(w * tmp )                                                    # calcute y hat which means prediction of cur's label = 1
            loss = CrossEntropy(y_hat, y)                                                # calcute loss by cross entropy
            delta_w = delta_w + learning_rate * (y - y_hat ) * tmp                       # update delta_w

            # when all of loss in this epoch is lower than tau, break the for-loop
            if loss > tau:
                flag = False
        w = w + delta_w
        if flag :
            print("loss is low enough. End in epoch : " + str(ep+1))
            break
        if ep == epoch-1:
            print("Epoch has been exceeded. End in epoch: " + str(ep+1))
    #w[0] = re_minmax_scale(w[0])
    #w[0] *= 200
    return w
    
def print_graphic(w, dataset):
    plt.xlim(min1-(max1-min1)/2, max1+(max1-min1)/2)                #畫布大小
    plt.ylim(min2-(max2-min2)/2, max2+(max2-min2)/2)
    #plt.xlim(-20,20)                #畫布大小
    #plt.ylim(-20,20)
    plt.axhline(0.5, color= 'gray')   #橫軸
    plt.axvline(0.5, color= 'gray')   #直軸

    plt.xlabel("x1")
    plt.ylabel("x2")

    x1 = np.linspace(-max1*1.5,max1*1.5,1000)
    x2 = -w[1]*x1 /w[2] - w[0]/w[2]   
    plt.plot(x1,x2)                 #畫出學習完的分隔線
    x2 = -initw[1]*x1 / initw[2] - initw[0]/ initw[2]
    plt.plot(x1,x2, color='r')   

    for i in data:                   #畫出訓練data的分布
        if i[2] == 0:
            plt.plot(i[0],i[1],"x", color='r', markersize=4 )
        else:
            plt.plot(i[0],i[1],"o", color='black', markersize=4)

    #建造LABEL
    plt.plot(-1000,-1000,"x", color='r', markersize=4 ,label='train_y=0')
    plt.plot(-1000,-1000,"o", color='black', markersize=4,label='train_y=1')
    plt.plot(-1000,-1000,"^", color='r',label='test_y=0')
    plt.plot(-1000,-1000,"^", color='black' ,label='test_y=1')
    plt.legend(loc='lower left')
    plt.show()                      




#find min and max of dataset
min1 = 1E9
min2 = 1E9
max1 = 0
max2 = 0
for cur in data:
    if cur[0] > max1:
        max1 = cur[0]
    if cur[1] > max2:
        max2 = cur[1]
    if cur[0] < min1:
        min1 = cur[0]
    if cur[1] < min2:
        min2 = cur[1]
weight = logistic_regression(data)
print("w0 = " + str(weight[0]))
print("w1 = " + str(weight[1]))
print("w2 = " + str(weight[2]))

print_graphic(weight, data)