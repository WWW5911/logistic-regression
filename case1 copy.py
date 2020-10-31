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
path1 = "train_case4.txt"
initw = np.array([random.random(), random.random(), random.random()])

#處理檔案
data = genfromtxt(path1, delimiter=',', names=('x1', 'x2', 'y'))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_data():
    data = np.loadtxt('train_case1.csv')
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  #特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels

def grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #(m,n)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))  #初始化回归系数（n, 1)
    alpha = 0.001 #步长
    maxCycle = 500  #最大循环次数

    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)  #sigmoid 函数
        weights = weights + alpha * dataMatrix.transpose() * (labelMat - h)  #梯度
    return weights
    
def print_graphic(w, dataset):
    plt.xlim(min1-(max1-min1)/2, max1+(max1-min1)/2)                #畫布大小
    plt.ylim(min2-(max2-min2)/2, max2+(max2-min2)/2)
    #plt.xlim(-20,20)                #畫布大小
    #plt.ylim(-20,20)
    plt.axhline(0.5, color= 'gray')   #橫軸
    plt.axvline(0.5, color= 'gray')   #直軸

    plt.xlabel("x1")
    plt.ylabel("x2")

    x = np.arange(-3, 3, 0.1)
    y = (-weight[0, 0] - weight[1, 0] * x) / weight[2, 0]  #matix
    plt.plot(x, y)


    for i in data:                  #畫出訓練data的分布
        if i['y'] == 0:
            plt.plot(i['x1'],i['x2'],"x", color='r', markersize=4)
        else:
            plt.plot(i['x1'],i['x2'],"o", color='black', markersize=4)

    plt.show()                      

#find min and max of dataset
min1 = 1E9
min2 = 1E9
max1 = 0
max2 = 0
for cur in data:
    if cur['x1'] > max1:
        max1 = cur['x1']
    if cur['x2'] > max2:
        max2 = cur['x2']
    if cur['x1'] < min1:
        min1 = cur['x1']
    if cur['x2'] < min2:
        min2 = cur['x2']
dataMatIn, classLabels = init_data()
weight = grad_descent(dataMatIn, classLabels)
weight = np.array( (weight[0], weight[1], weight[2]) )
print("w0 = " + str(weight[0]))
print("w1 = " + str(weight[1]))
print("w2 = " + str(weight[2]))

print_graphic(weight, data)